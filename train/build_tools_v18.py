#!/usr/bin/env python3
"""Tokenise the v2 tool corpus into loss-masked shards.

Separate from build_sft_v17.py on purpose. v17's toolcall path builds each example's menu by taking
the correct tool plus three distractors drawn from ONE GLOBAL POOL scanned out of the corpus
(`scan_tools` / `tool_menu`). That is the exact defect docs/44 measured: every example shares the
same ~15 tool names, so the model learns that vocabulary rather than learning to read a schema. On a
menu built from the real Paystack OpenAPI spec it scored 9%.

Here every record carries its OWN menu, generated per job across five naming conventions
(paystack / verbnoun / camel / dotted / terse), 20,000 distinct menus for 20,000 examples. The menu
is used verbatim and never rebuilt. Nothing is shared between examples except the format.

Four types, and the SHAPE of each one is the lesson:

  toolcall_api   read an unfamiliar schema, call, then answer from the result
  toolpatch      emit ONLY the changed fields. docs/44: asked to change one field, the released
                 model re-emitted the whole call and corrupted an untouched amount 6 times out of 6
  toolrefuse     answer in prose and call nothing (over-calling, and 41% benign over-refusal)
  tooldisambig   ask one clarifying question first, THEN call

The newline after <reserved_1> is load-bearing and is the same convention v17 uses. `<reserved_1>\\n`
opens prose; `<reserved_1><reserved_3>` opens a call. If a no-call example were written in the
call shape the model would read the format itself as the instruction to call and the contrast that
teaches restraint would be lost.

usage: build_tools_v18.py --raw $B/synth_raw --out-dir $B/data/sft18_tools
"""
import argparse, glob, json, os, random, re, sys, time

ROOT = "/leonardo_scratch/large/userexternal/imisi000/morena"
TOKENIZER = f"{ROOT}/data/tokenizer/tokenizer.json"
SHARD_TOKENS = 100_000_000
TYPES = ("toolcall_api", "toolact", "seedance", "toolpatch", "toolrefuse",
         "tooldisambig", "toolsession")

USER, ASSISTANT, TOOLS = "<reserved_0>", "<reserved_1>", "<reserved_2>"
TOOL_CALL, TOOL_RESULT = "<reserved_3>", "<reserved_5>"

BLOCK = re.compile(r'^(USER2|USER|CALL|PATCH|RESULT|ANSWER|DECISION|ASK)\s*:\s*(.*)$')
TBLOCK = re.compile(r'^T(\d+)_(USER|ACT|RESULT|ANSWER)\s*:\s*(.*)$')


def parse_blocks(text):
    out, cur, key = {}, [], None
    for line in (text or "").splitlines():
        m = BLOCK.match(line.strip())
        if m:
            if key:
                out[key] = "\n".join(cur).strip()
            key, cur = m.group(1), [m.group(2)]
        elif key:
            cur.append(line)
    if key:
        out[key] = "\n".join(cur).strip()
    return out


def jload(s):
    """First JSON object in a block. The generator sometimes wraps a line in ``` or prose."""
    if not s:
        return None
    i = s.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[i:])[0]
    except Exception:
        return None


def j1(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def parse_turns(text):
    """T<n>_USER / _ACT / _RESULT / _ANSWER -> an ordered list of turn dicts.

    Turns are keyed by their stated number rather than by order of appearance, because the
    generator occasionally repeats a block, and sorting by the number is the only way to tell a
    repeat from a new turn.
    """
    turns, cur, key = {}, [], None
    for line in (text or "").splitlines():
        m = TBLOCK.match(line.strip())
        if m:
            if key:
                turns.setdefault(key[0], {})[key[1]] = "\n".join(cur).strip()
            key, cur = (int(m.group(1)), m.group(2)), [m.group(3)]
        elif key:
            cur.append(line)
    if key:
        turns.setdefault(key[0], {})[key[1]] = "\n".join(cur).strip()
    return [turns[k] for k in sorted(turns)]


def none_ish(v):
    return (not v) or v.strip().upper() in ("NONE", "NULL", "N/A", "-")


def session_segments(rec, menu):
    """A whole conversation as one training example: the thread only carries if the earlier
    turns are in the context window when the later ones are predicted. Splitting it into
    independent pairs would teach exactly the statelessness this type exists to remove."""
    turns = parse_turns(rec.get("output"))
    if len(turns) < 3:
        return None
    segs = [(f"{USER}\n{TOOLS}{j1(menu)}", False)]
    trained_any = False
    for i, t in enumerate(turns):
        u, act, res, ans = (t.get("USER"), t.get("ACT"), t.get("RESULT"), t.get("ANSWER"))
        if not u or not ans:
            return None
        if i > 0:
            segs.append((USER, False))
        segs.append(("\n" + u.strip() + "\n", False))
        if none_ish(act):
            # No tool. The newline after <reserved_1> is the model's own decision and has to
            # be inside the trained span, same as every other type here.
            segs.append((ASSISTANT, False))
            segs.append(("\n" + ans.strip(), True))
        else:
            call = jload(act)
            if not call:
                return None
            segs.append((ASSISTANT, False))
            segs.append((TOOL_CALL + j1(call), True))
            if not none_ish(res):
                r = jload(res)
                if r is not None:
                    segs.append((TOOL_RESULT + j1(r), False))
            segs.append((ASSISTANT, False))
            segs.append(("\n" + ans.strip(), True))
        trained_any = True
    return segs if trained_any else None


def segments(rec):
    """-> [(text, trainable), ...] or None if the record cannot be rendered.

    Rendering only. Everything that decides whether a record DESERVES to be training data
    (undeclared arguments, a patch restating unchanged fields, a call where there should be none)
    is validate_tools_v2.py's job and has already run.
    """
    typ = rec.get("type")
    menu = rec.get("menu")
    if not menu:
        return None

    if typ == "toolsession":
        return session_segments(rec, menu)

    # The single- and two-turn types share one USER block; sessions do not, and their T<n>_USER
    # lines do not match this pattern, so the session branch has to come first.
    b = parse_blocks(rec.get("output"))
    if not b.get("USER"):
        return None
    head = f"{USER}\n{TOOLS}{j1(menu)}\n{b['USER'].strip()}\n"

    if typ in ("toolcall_api", "toolact", "seedance"):
        call, ans = jload(b.get("CALL")), (b.get("ANSWER") or "").strip()
        if not call or not ans:
            return None
        segs = [(head + ASSISTANT, False), (TOOL_CALL + j1(call), True)]
        res = jload(b.get("RESULT"))
        if res is None:
            return segs
        # The answer turn is trained too: reading a result back into the user's language is half
        # of what a tool-calling assistant does, and it is where a fintech demo is actually judged.
        segs += [(TOOL_RESULT + j1(res) + ASSISTANT, False), ("\n" + ans, True)]
        return segs

    if typ == "toolpatch":
        cur = (rec.get("input") or {}).get("current")
        patch, ans = jload(b.get("PATCH")), (b.get("ANSWER") or "").strip()
        if not cur or not patch or not ans:
            return None
        # The pending call is replayed as the assistant's PRIOR turn, which is where it came from in
        # any real client. The user turn that produced it was never generated, so the exchange opens
        # on an assistant call. That is a small format irregularity accepted deliberately: the
        # alternative is inventing a first user turn, and an invented turn that does not match the
        # pending call teaches a false correspondence between request and arguments.
        opening = f"{USER}\n{TOOLS}{j1(menu)}\n{ASSISTANT}{TOOL_CALL}{j1(cur)}"
        return [(opening + f"{USER}\n{b['USER'].strip()}\n" + ASSISTANT, False),
                (TOOL_CALL + j1(patch), True),
                (ASSISTANT, False),
                ("\n" + ans, True)]

    if typ == "toolrefuse":
        ans = (b.get("ANSWER") or "").strip()
        if not ans:
            return None
        # No DECISION in the target. DECISION is the generator's self-report, used by the validator
        # to confirm the example really is a no-call one; training on it would teach the model to
        # narrate a decision object before every answer.
        return [(head + ASSISTANT, False), ("\n" + ans, True)]

    if typ == "tooldisambig":
        ask, u2, call = (b.get("ASK") or "").strip(), (b.get("USER2") or "").strip(), jload(b.get("CALL"))
        if not ask or not u2 or not call:
            return None
        return [(head + ASSISTANT, False),
                ("\n" + ask, True),
                (f"{USER}\n{u2}\n" + ASSISTANT, False),
                (TOOL_CALL + j1(call), True)]

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=f"{ROOT}/synth_raw")
    ap.add_argument("--out-dir", default=f"{ROOT}/data/sft18_tools")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--seed", type=int, default=18)
    ap.add_argument("--dup", default="toolcall_api=2,toolact=3,seedance=2,toolpatch=2,toolrefuse=1,tooldisambig=1,toolsession=2",
                    help="type=factor,... see the note below before changing these")
    # DUP HISTORY, because two rounds of guessing cost two training runs.
    #   v1  api=1 patch=3 refuse=2 disambig=3  ->  free-choice tool accuracy 64%
    #   v2  api=2 patch=3 refuse=1 disambig=3  ->  18% (nano) / 9% (mini). WORSE.
    # The v2 reasoning was that refusals outweighed calls. That was the wrong culprit. Reading the
    # actual completions: every single decline is a CLARIFYING QUESTION, not a refusal, and several
    # ask for a field the prompt already supplied ("wetin be the transaction reference?" when the
    # reference was given). One asks which bank a refund should use, which is not a field refunds
    # have. disambig at x3 was a quarter of the corpus teaching "when unsure, ask", and terse real
    # requests always look under-specified to a model holding that prior.
    # v3 cut disambig to x1 and raised straightforward calls to x3. The sessions carry several
    # immediate calls each, in context, as a further counterweight.
    #   v3  api=3 patch=2 refuse=1 disambig=1 session=2  ->  free-choice 50% (nano 27%)
    # Better, and still only half. Reading the live model: it stalls on TERSE requests, asking for
    # a field the tool does not have ("which bank?" on a transfer with no bank argument). Nothing
    # in the corpus showed a short request being acted on, because every generated request was
    # fully specified. v4 adds `toolact` at x3: deliberately terse requests, answered by calling
    # immediately with only the arguments the user supplied and nothing invented. Those records
    # also pin the reply to the user's own language, which is the second complaint: it drifts into
    # English on the answer even when the whole conversation is Yoruba.
    ap.add_argument("--max-per-type", type=int, default=200000)
    a = ap.parse_args()

    import numpy as np
    from tokenizers import Tokenizer

    dup = {}
    for kv in a.dup.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            dup[k.strip()] = int(v)

    tok = Tokenizer.from_file(a.tokenizer)
    eos = tok.token_to_id("<eos>")
    for m in (USER, ASSISTANT, TOOLS, TOOL_CALL, TOOL_RESULT):
        ids = tok.encode(m).ids
        assert len(ids) == 1, f"{m} is not one token: {ids}"

    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    examples, stats, dropped = [], {}, {}
    for typ in TYPES:
        for ldir in sorted(glob.glob(f"{a.raw}/{typ}/*")):
            lang, got, bad = os.path.basename(ldir), 0, 0
            for p in sorted(glob.glob(f"{ldir}/part-*.jsonl")):
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if not line or got >= a.max_per_type:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        bad += 1
                        continue
                    if r.get("degenerate") or r.get("empty"):
                        bad += 1
                        continue
                    if str(r.get("id", "")).startswith("t6-hold-"):
                        # A held-out record has no business in a training shard. Failing loudly
                        # beats a quiet skip: if these are reaching the builder at all, the
                        # staging step is wrong and every number downstream is contaminated.
                        sys.exit(f"HELD-OUT record {r['id']} found in {p}. Staging is wrong.")
                    segs = segments(r)
                    if not segs:
                        bad += 1
                        continue
                    examples += [segs] * dup.get(typ, 1)
                    got += 1
            if got or bad:
                stats[f"{typ}/{lang}"] = got
                dropped[f"{typ}/{lang}"] = bad
                print(f"  {typ}/{lang}: {got:,} kept, {bad:,} unrenderable (x{dup.get(typ,1)})",
                      flush=True)

    if not examples:
        sys.exit("no renderable records; check --raw layout is <type>/<lang>/part-*.jsonl")
    rng.shuffle(examples)
    print(f"[tools] {len(examples):,} examples after duplication", flush=True)

    def encode_segments(segs):
        """Tokenise with a per-segment loss mask. BPE does not guarantee encode(p+t) starts with
        encode(p), so the prefix is verified and the two halves encoded independently when it does
        not. Getting this wrong shifts the mask by a token and trains on the wrong side of it."""
        ids, mask, prev = [], [], ""
        for text, train in segs:
            enc = tok.encode(prev + text).ids
            base = tok.encode(prev).ids if prev else []
            if enc[:len(base)] != base:
                enc = base + tok.encode(text).ids
            new = enc[len(base):]
            ids += new
            mask += [1 if train else 0] * len(new)
            prev += text
        ids.append(eos)
        mask.append(1)
        return ids, mask

    t0 = time.time()
    buf, mbuf, n, k, manifest, ntok, nresp = [], [], 0, 0, [], 0, 0

    def flush():
        nonlocal buf, mbuf, n, k
        if not buf:
            return
        arr, msk = np.concatenate(buf), np.concatenate(mbuf)
        assert arr.size == msk.size, (arr.size, msk.size)
        fn, mn = f"shard_tools_{k:03d}.bin", f"mask_tools_{k:03d}.bin"
        for data, name in ((arr, fn), (msk, mn)):
            tmp = os.path.join(a.out_dir, name + ".tmp")
            data.tofile(tmp)
            os.replace(tmp, os.path.join(a.out_dir, name))
        manifest.append({"file": fn, "n_tokens": int(arr.size), "mask": mn})
        k += 1
        buf, mbuf, n = [], [], 0

    skipped = 0
    for e in examples:
        i_, m_ = encode_segments(e)
        if len(i_) <= 1 or not any(m_):
            skipped += 1
            continue
        buf.append(np.asarray(i_, dtype=np.uint16))
        mbuf.append(np.asarray(m_, dtype=np.uint8))
        n += len(i_)
        ntok += len(i_)
        nresp += sum(m_)
        if n >= SHARD_TOKENS:
            flush()
    flush()

    sha = ""
    try:
        import hashlib
        sha = hashlib.sha256(open(a.tokenizer, "rb").read()).hexdigest()[:16]
    except Exception:
        pass
    index = {"dtype": "uint16", "eos_id": eos, "vocab_size": tok.get_vocab_size(),
             "tokenizer_sha": sha, "mask_dtype": "uint8",
             "n_response_tokens": int(nresp), "response_frac": round(nresp / max(ntok, 1), 4),
             "shards": manifest, "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "counts": stats, "unrenderable": dropped, "dup": dup}
    with open(os.path.join(a.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"[tools] {ntok:,} tokens, {nresp:,} trained ({100*nresp/max(ntok,1):.1f}%), "
          f"{len(manifest)} shard(s), {skipped:,} skipped, {time.time()-t0:.0f}s -> {a.out_dir}")


if __name__ == "__main__":
    main()
