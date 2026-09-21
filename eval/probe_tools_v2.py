#!/usr/bin/env python3
"""Measure what the v18 tool fine-tune actually bought, against the released checkpoints.

Runs on Leonardo, not Modal:
    MORENA_LOCAL=1 python probe_tools_v2.py --out $B/eval/tools_v2.json

The comparison is base vs fine-tuned on the SAME harness. The 9% in docs/44 was measured on the
1.5B through the Modal lab, so it is not a number the nano or the mini can be scored against
directly; both released checkpoints are re-measured here instead.

The menu is the real Paystack OpenAPI spec and is HELD OUT: it appears in no training record, and
none of the 20,000 generated menus reuse its tool names. A model that improves here improved at
reading a schema, not at recognising one.

Four suites, one per behaviour the fine-tune trained:

  A schema     22 Nigerian-language requests. Did it pick the right tool and use only the
               arguments that tool declares? Prefilled with <reserved_3>, which is how docs/44
               measured it, so "opened a call" is not what is being scored: the choice is.
  B patch      A call is pending and the user changes ONE field. Two separate things are scored.
               minimal: did it emit only the changed field. preserved: whatever it emitted, are
               the fields the user never mentioned still carrying their original values. The
               second is the docs/44 corruption bug, which hit 6 times out of 6.
  C no-call    Requests that need no tool. Not prefilled, because prefilling a call marker would
               decide the question being asked.
  D disambig   Requests missing something the tool requires. Did it ask, or did it guess.
  F session    A five-turn conversation that SWITCHES LANGUAGE partway. Two things are scored
               separately: does the thread survive to turn five (a terse "make that one 3000"
               resolves only against turn one), and does the reply follow the user into the new
               language instead of carrying on in the old one.
  G language   One terse request per language, then the assistant's spoken answer after the tool
               result. Scores whether that answer is in the language the user wrote in. Separate
               from F, which is about MOVING between languages; this is about STAYING put, which
               is the thing that actually drifts.
  E free       Suite A again with NO prefill. A is scored on which tool it picks, having already
               been told to pick one. E is what the demo actually does: the model decides whether
               to call at all. The first tool run scored 90.9% on A and 63.6% on E, and the whole
               gap was over-abstention, so A alone is the flattering half of the story.
"""
import argparse, json, os, re, sys

USER, ASSISTANT = "<reserved_0>", "<reserved_1>"
TOOLS_MARKER, TOOL_CALL, TOOL_RESULT = "<reserved_2>", "<reserved_3>", "<reserved_5>"
H = "/leonardo/home/userexternal/imisi000/morena"
B = "/leonardo_scratch/large/userexternal/imisi000/morena"
TOK = f"{B}/data/tokenizer/tokenizer.json"


def j1(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def first_json(text):
    """The model still emits the legacy <|tool_call|> string after a reserved prefill, so the
    parser looks for the first brace rather than requiring a marker."""
    body = text
    for stop in (TOOL_RESULT, "<|tool_result|>", ASSISTANT, "<|assistant|>", USER):
        body = body.split(stop)[0]
    i = body.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(body[i:])[0]
    except Exception:
        return None


# ---- suite B: pending calls and the single field each user changes -----------------------------
# Every `current` here uses plausible values, not placeholders, because the question is whether an
# untouched field survives, and a field that was nonsense to begin with cannot answer it.
PATCH_CASES = [
    {"lang": "Pidgin", "current": {"name": "transfer_initiate",
        "arguments": {"amount": 500000, "recipient": "RCP_8812", "reason": "school fees", "currency": "NGN"}},
     "say": "No, make am 250000 instead.", "field": "amount", "value": 250000},
    {"lang": "English", "current": {"name": "transfer_initiate",
        "arguments": {"amount": 75000, "recipient": "RCP_3390", "reason": "rent", "currency": "NGN"}},
     "say": "Change the reason to deposit, please.", "field": "reason", "value": "deposit"},
    {"lang": "Hausa", "current": {"name": "transfer_initiate",
        "arguments": {"amount": 120000, "recipient": "RCP_5521", "reason": "kayan gini", "currency": "NGN"}},
     "say": "A tura wa RCP_7788 maimakon haka.", "field": "recipient", "value": "RCP_7788"},
    {"lang": "Yoruba", "current": {"name": "transfer_initiate",
        "arguments": {"amount": 45000, "recipient": "RCP_1204", "reason": "owo ile", "currency": "NGN"}},
     "say": "Je ki o je 60000.", "field": "amount", "value": 60000},
    {"lang": "Igbo", "current": {"name": "transfer_initiate",
        "arguments": {"amount": 30000, "recipient": "RCP_9001", "reason": "ugwo ulo", "currency": "NGN"}},
     "say": "Gbanwee ihe kpatara ya ka o buru ego nri.", "field": "reason", "value": None},
    {"lang": "Pidgin", "current": {"name": "transfer_initiate",
        "arguments": {"amount": 15000, "recipient": "RCP_4417", "reason": "transport", "currency": "NGN"}},
     "say": "Abeg change the recipient to RCP_2200.", "field": "recipient", "value": "RCP_2200"},
]

NO_CALL = [
    ("Pidgin", "Thank you well well, you don try."),
    ("English", "Good morning. How are you today?"),
    ("Pidgin", "Wetin be the difference between savings and current account?"),
    ("Hausa", "Na gode sosai da taimako."),
    ("English", "Who won the match last night?"),
    ("Yoruba", "E se pupo, o ti to."),
    ("English", "Can you explain what a bank code is?"),
    ("Igbo", "Daalu, ihe niile adigo mma."),
]

DISAMBIG = [
    ("Pidgin", "Abeg send money give Chidi."),                 # no amount
    ("English", "Transfer 20000 naira."),                       # no recipient
    ("Hausa", "Ka duba sunan mai asusun nan."),                 # no account number or bank
    ("English", "Verify the payment for me."),                  # no reference
    ("Yoruba", "Fi owo ranse si mi ore."),                      # no amount, no recipient
    ("Pidgin", "Buy airtime."),                                 # nothing at all
]


# Language identification here is a word-list heuristic, not a classifier. It is checking one
# thing: did the reply move to the language the user just moved to. High-frequency function words
# separate these four well enough for that, and the alternative is shipping a langid model to a
# compute node with no internet.
LANG_WORDS = {
    "yor": ["ni", "ti", "mo", "je", "ki", "owo", "se", "yin", "fun", "won", "pe", "awon"],
    "hau": ["na", "da", "ka", "ba", "yi", "wannan", "zan", "ne", "in", "sosai", "yin",
            "bidiyo", "bidiyon", "ku", "take", "gode"],
    "ibo": ["na", "ya", "ka", "maka", "gi", "nke", "ihe", "biko", "ndi", "anyi"],
    "pcm": ["abeg", "dey", "na", "make", "wetin", "don", "am", "sabi", "go", "fit", "una"],
    "eng": ["the", "your", "a", "is", "we", "will", "video", "clip", "making"],
    # These three were missing, and the corpus contains all of them. Every Zulu, Shona and
    # Swahili answer therefore scored as a language failure while being perfectly correct:
    # "Siyakwenza i-video yakho yomdlalo webhola emigwaqweni" is exactly right, and the detector
    # simply could not see it. A missing word list reads as a model defect, which is the worst
    # kind of measurement bug because it sends the next fix in the wrong direction.
    "zul": ["siyakwenza", "yakho", "i", "le", "kanye", "ngo", "uku", "isi", "wena", "futhi"],
    "sna": ["tiri", "kuita", "yako", "ne", "iyi", "uye", "kwa", "vanhu", "ndi", "pa"],
    "swh": ["tunatengeneza", "yako", "na", "ya", "wa", "kwa", "hii", "video", "ili", "katika"],
}


DIACRITIC = [
    ("yor", "\u1eb9\u1eb8\u1ecd\u1ecc\u1e63\u1e62"),        # e o s with under-dot
    ("ibo", "\u1ee5\u1ee4\u1ecb\u1eca\u1e45\u1e44"),        # u i with under-dot, n with over-dot
    ("hau", "\u0199\u0198\u0257\u018a\u0253\u0181"),        # hooked k d b
]


def lang_guess(t):
    raw = t or ""
    for lg, marks in DIACRITIC:
        if any(c in raw for c in marks):
            # Yoruba and Igbo share the o-dot, so a tie is broken on the marks unique to each.
            if lg == "yor" and any(c in raw for c in "\u1ee5\u1ee4\u1ecb\u1eca") \
               and not any(c in raw for c in "\u1eb9\u1eb8\u1e63\u1e62"):
                return "ibo"
            return lg
    t = " " + "".join(c.lower() if c.isalnum() else " " for c in raw) + " "
    # Two languages need a marker pass before counting. Pidgin shares most of its vocabulary with
    # English, so "I dey work on dat video for your website" loses on raw counts while carrying an
    # unmistakable "dey"; and undiacriticised Hausa was landing on Yoruba because "yin" appears in
    # both lists. A single strong marker settles either case.
    for lg, marks in (("pcm", (" dey ", " wetin ", " abeg ", " sabi ", " una ", " o ", " don ")),
                      ("hau", (" ana ", " yanzu ", " haka ", " kuma ", " nan ", " muna ", " naka "))):
        if any(mk in t for mk in marks):
            return lg
    best, sc = None, 0
    for lg, ws in LANG_WORDS.items():
        n = sum(t.count(" " + w + " ") for w in ws)
        if n > sc:
            best, sc = lg, n
    return best



# Six turns, one thread, switching language at turn 4.
#
# The first version scored the switch on turn 4, which asks for money to be sent. The model
# correctly answers that with a tool call, and a JSON object has no language, so the detector read
# English field names and called it a failure. That produced "switch 25%" for a model that was
# behaving perfectly. The language test now lands on turn 6, which is a thank-you and can only be
# answered in words, and the harness generates the assistant's spoken answer after each tool
# result instead of fabricating one, so the history the model sees is its own.
SESSIONS = [
    {"switch_to": "yor", "want": 3000, "turns": [
        "Abeg send 20000 naira give Chidi for school fees.",
        "Wetin be my balance now?",
        "Thank you o.",
        "Fi 5000 naira ranse si Chidi lekan si.",
        "Rara, je ki o je 3000.",
        "O se, o ti to."]},
    {"switch_to": "hau", "want": 2500, "turns": [
        "Abeg send 15000 naira give Musa for rent.",
        "How much dey my account?",
        "Thank you well well.",
        "Ka tura 4000 naira zuwa ga Musa kuma.",
        "A'a, ka mai da shi 2500.",
        "Na gode sosai."]},
    {"switch_to": "ibo", "want": 3500, "turns": [
        "Abeg send 12000 naira give Ada for market.",
        "Check my balance abeg.",
        "Thanks.",
        "Zipu 6000 naira nye Ada ozo.",
        "Mba, mee ka o buru 3500.",
        "Daalu nke ukwuu."]},
    {"switch_to": "eng", "want": 2000, "turns": [
        "Abeg send 9000 naira give Emeka.",
        "Wetin remain for my account?",
        "Nice one.",
        "Please send another 4000 to Emeka.",
        "No, make it 2000 instead.",
        "Thank you, that is all."]},
]


# One short request per language, phrased the way someone would type it.
STAY = [
    ("pcm", "Abeg send 5000 give Chidi."),
    ("yor", "Fi 5000 ranse si Chidi."),
    ("ibo", "Zipu 5000 nye Chidi."),
    ("hau", "Ka tura 5000 zuwa ga Chidi."),
]


def latest_ckpt(run):
    """Newest step_* under a run, ignoring half-written .tmp dirs.

    Hardcoding the step number meant the eval silently SKIPPED both fine-tunes the moment a run
    used a different number of steps, and printed a clean table of base models as though that were
    the answer. A skip must not be mistakable for a result.
    """
    import glob as _g
    cks = sorted(d for d in _g.glob(f"{B}/runs/{run}/ckpt/step_*") if not d.endswith(".tmp"))
    return cks[-1] if cks else f"{B}/runs/{run}/ckpt/MISSING"


def build_specs(which):
    cfg_nano = f"{H}/train/configs/morena0p2b_nano.json"
    cfg_mini = f"{H}/train/configs/morena0p5b_mini.json"
    all_specs = [
        {"kind": "morena", "name": "nano-base", "config": cfg_nano, "tokenizer": TOK,
         "ckpt": f"{B}/runs/morena0p2b-nano/ckpt/step_00010000"},
        {"kind": "morena", "name": "nano-tools", "config": cfg_nano, "tokenizer": TOK,
         "ckpt": latest_ckpt("morena0p2b-nano-tools")},
        {"kind": "morena", "name": "mini-base", "config": cfg_mini, "tokenizer": TOK,
         "ckpt": f"{B}/runs/morena0p5b-mini-sft/ckpt/step_00011500"},
        {"kind": "morena", "name": "mini-tools", "config": cfg_mini, "tokenizer": TOK,
         "ckpt": latest_ckpt("morena0p5b-mini-tools")},
    ]
    if which == "all":
        return all_specs
    want = set(which.split(","))
    return [s for s in all_specs if s["name"] in want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu", default=f"{H}/eval/paystack_menu.json")
    ap.add_argument("--set", dest="jobset", default=f"{H}/eval/paystack_ng.json")
    ap.add_argument("--out", default=f"{B}/eval/tools_v2.json")
    ap.add_argument("--models", default="all")
    ap.add_argument("--max-new", type=int, default=160)
    a = ap.parse_args()

    sys.path.insert(0, f"{H}/eval")
    from modelio import build_runner

    menu = json.load(open(a.menu, encoding="utf-8"))["tools"]
    # Aliases exist in the menu file to study name recall; they are stripped here so the model has
    # nothing to match on but the description and the argument names.
    menu = [{k: v for k, v in t.items() if k != "aliases"} for t in menu]
    by_name = {t["name"]: t for t in menu}
    menu_json = j1(menu)
    jobs = json.load(open(a.jobset, encoding="utf-8"))["jobs"]

    head = f"{USER}\n{TOOLS_MARKER}{menu_json}\n"
    p_schema = [f"{head}{j['question'].strip()}\n{ASSISTANT}{TOOL_CALL}" for j in jobs]
    p_patch = [f"{USER}\n{TOOLS_MARKER}{menu_json}\n{ASSISTANT}{TOOL_CALL}{j1(c['current'])}"
               f"{USER}\n{c['say'].strip()}\n{ASSISTANT}{TOOL_CALL}" for c in PATCH_CASES]
    p_free = [f"{head}{j['question'].strip()}\n{ASSISTANT}" for j in jobs]
    p_nocall = [f"{head}{q}\n{ASSISTANT}" for _, q in NO_CALL]
    p_disamb = [f"{head}{q}\n{ASSISTANT}" for _, q in DISAMBIG]

    results, missing = {}, []
    for spec in build_specs(a.models):
        name = spec["name"]
        if not os.path.isdir(spec["ckpt"]):
            print(f"!! MISSING {name}: no checkpoint at {spec['ckpt']}", flush=True)
            missing.append(name)
            continue
        print(f"\n[probe] === {name} ===", flush=True)
        runner = build_runner(spec)
        gen = lambda ps: [o["text"] if isinstance(o, dict) else o for o in
                          runner.generate(ps, max_new_tokens=a.max_new, temperature=0.0, top_p=1.0)]

        # --- A: schema reading -------------------------------------------------------------
        A = []
        for j, raw in zip(jobs, gen(p_schema)):
            c = first_json(raw)
            nm = (c or {}).get("name")
            declared = set(by_name.get(nm, {}).get("arguments", {}))
            got = set(((c or {}).get("arguments") or {}))
            A.append({"lang": j["lang"], "want": j["expect"], "got": nm,
                      "json_ok": isinstance(c, dict),
                      "right": nm == j["expect"],
                      "args_clean": bool(nm in by_name) and not (got - declared),
                      "raw": raw[:200]})

        # --- E: same requests, free to decline -------------------------------------------
        E = []
        for j, raw in zip(jobs, gen(p_free)):
            c = first_json(raw)
            nm = (c or {}).get("name")
            E.append({"lang": j["lang"], "want": j["expect"], "got": nm,
                      "right": nm == j["expect"],
                      "declined": not (TOOL_CALL in raw or "<|tool_call|>" in raw or c is not None),
                      "raw": raw[:160]})

        # --- F: long session with a language switch --------------------------------------
        F = []
        for sess in SESSIONS:
            hist = f"{USER}\n{TOOLS_MARKER}{menu_json}"
            edit_call, prose_after_switch = None, None
            for i, turn in enumerate(sess["turns"]):
                hist += ("" if i == 0 else USER) + "\n" + turn + "\n" + ASSISTANT
                raw = gen([hist])[0]
                call = first_json(raw)
                if call is not None:
                    # A call: append it, hand back a result, then let the model say its own
                    # sentence about it. That sentence is what carries the language.
                    hist += TOOL_CALL + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
                    hist += TOOL_RESULT + '{"status":"success"}' + ASSISTANT
                    said = gen([hist])[0].split(USER)[0].strip()
                    hist += "\n" + said[:200]
                else:
                    said = raw.split(TOOL_RESULT)[0].strip()
                    hist += "\n" + said[:200]
                if i == 4:
                    edit_call = call
                if i == 5:
                    prose_after_switch = said
            amt = None
            if isinstance(edit_call, dict):
                src = edit_call.get("set") or edit_call.get("arguments") or {}
                for k, v in src.items():
                    if "amount" in k.lower() or (isinstance(v, (int, float)) and v > 100):
                        amt = v
            F.append({"switch_to": sess["switch_to"],
                      "followed": lang_guess(prose_after_switch) == sess["switch_to"],
                      "kept_thread": amt == sess["want"],
                      "amt": amt, "want": sess["want"],
                      "guessed": lang_guess(prose_after_switch),
                      "reply6": (prose_after_switch or "")[:140]})

        # --- G: answering in the language you were asked in ------------------------------
        G = []
        for lg, q in STAY:
            base = f"{USER}\n{TOOLS_MARKER}{menu_json}\n{q}\n{ASSISTANT}"
            raw = gen([base + TOOL_CALL])[0]          # prefilled: this suite is about the ANSWER
            call = first_json(TOOL_CALL + raw)
            if call is None:
                G.append({"lang": lg, "stayed": False, "answer": "", "guessed": None})
                continue
            hist = (base + TOOL_CALL
                    + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
                    + TOOL_RESULT + '{"status":"success"}' + ASSISTANT)
            said = gen([hist])[0].split(USER)[0].strip()
            G.append({"lang": lg, "stayed": lang_guess(said) == lg,
                      "guessed": lang_guess(said), "answer": said[:120]})

        # --- B: patch --------------------------------------------------------------------
        Bres = []
        for c_, raw in zip(PATCH_CASES, gen(p_patch)):
            o = first_json(raw)
            cur = c_["current"]["arguments"]
            minimal = False
            emitted = {}
            if isinstance(o, dict):
                if o.get("op") == "patch":
                    emitted = dict(o.get("set") or {})
                    for k in (o.get("unset") or []):
                        emitted[k] = None
                    minimal = set(emitted) == {c_["field"]}
                elif isinstance(o.get("arguments"), dict):
                    emitted = o["arguments"]          # a full call: the old behaviour
            # preserved: every field the user did NOT mention still holds its original value.
            # A full call that repeats them correctly still counts as preserved; only a CHANGED
            # value is the corruption docs/44 measured.
            untouched = [k for k in cur if k != c_["field"]]
            preserved = all(k not in emitted or emitted[k] == cur[k] for k in untouched)
            Bres.append({"lang": c_["lang"], "field": c_["field"], "minimal": minimal,
                         "preserved": preserved, "json_ok": isinstance(o, dict),
                         "emitted": emitted, "raw": raw[:200]})

        # --- C: no-call ------------------------------------------------------------------
        C = []
        for (lang, q), raw in zip(NO_CALL, gen(p_nocall)):
            called = (TOOL_CALL in raw) or ("<|tool_call|>" in raw) or (first_json(raw) is not None)
            C.append({"lang": lang, "q": q, "abstained": not called, "raw": raw[:160]})

        # --- D: disambiguation -----------------------------------------------------------
        D = []
        for (lang, q), raw in zip(DISAMBIG, gen(p_disamb)):
            called = (TOOL_CALL in raw) or ("<|tool_call|>" in raw) or (first_json(raw) is not None)
            asked = ("?" in raw or "?" in raw) and not called
            D.append({"lang": lang, "q": q, "asked": asked, "called": called, "raw": raw[:160]})

        pct = lambda rows, k: 100.0 * sum(bool(r[k]) for r in rows) / max(len(rows), 1)
        summary = {
            "schema_right": pct(A, "right"), "schema_args_clean": pct(A, "args_clean"),
            "patch_minimal": pct(Bres, "minimal"), "patch_preserved": pct(Bres, "preserved"),
            "nocall_abstained": pct(C, "abstained"), "disambig_asked": pct(D, "asked"),
            "free_right": pct(E, "right"), "free_declined": pct(E, "declined"),
            "switch_followed": pct(F, "followed"), "session_thread": pct(F, "kept_thread"),
            "stayed_in_language": pct(G, "stayed"),
        }
        results[name] = {"summary": summary, "schema": A, "free": E, "patch": Bres,
                         "nocall": C, "disambig": D, "session": F, "language": G}
        print(f"  schema right {summary['schema_right']:5.1f}%  FREE right {summary['free_right']:5.1f}% "
              f"(declined {summary['free_declined']:4.1f}%)  args clean {summary['schema_args_clean']:5.1f}%  "
              f"patch minimal {summary['patch_minimal']:5.1f}%  preserved {summary['patch_preserved']:5.1f}%  "
              f"no-call {summary['nocall_abstained']:5.1f}%  asked {summary['disambig_asked']:5.1f}%  "
              f"switch {summary['switch_followed']:5.1f}%  thread {summary['session_thread']:5.1f}%  "
              f"stays {summary['stayed_in_language']:5.1f}%", flush=True)
        del runner
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    if missing:
        print(f"\n!! {len(missing)} checkpoint(s) missing and NOT evaluated: {', '.join(missing)}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {a.out}\n")
    hdr = (f"{'model':12s} {'schema':>8s} {'FREE':>7s} {'declnd':>7s} {'args':>7s} "
           f"{'patch min':>10s} {'no-call':>8s} {'asked':>7s} {'switch':>7s} {'thread':>7s} {'stays':>7s}")
    print(hdr); print("-" * len(hdr))
    for n, r in results.items():
        s = r["summary"]
        print(f"{n:12s} {s['schema_right']:7.1f}% {s['free_right']:6.1f}% {s['free_declined']:6.1f}% "
              f"{s['schema_args_clean']:6.1f}% {s['patch_minimal']:9.1f}% "
              f"{s['nocall_abstained']:7.1f}% {s['disambig_asked']:6.1f}% "
              f"{s['switch_followed']:6.1f}% {s['session_thread']:6.1f}% {s['stayed_in_language']:6.1f}%")


if __name__ == "__main__":
    main()
