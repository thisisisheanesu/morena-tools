#!/usr/bin/env python3
"""Score a fine-tune on HELD-OUT Seedance briefs.

    MORENA_LOCAL=1 python probe_seedance.py --holdout <clean_holdout.jsonl> --ckpt <dir> --config <cfg>

Every brief here was generated with an id of its own (`t6-hold-`) and was never staged into a
training shard; build_tools_v18.py exits with an error if one turns up in the corpus, so a leak
fails the build rather than quietly inflating this score.

Nothing here is scored against the generator's wording. A video prompt has no single right answer,
so what is checked is whether the instruction could be POSTED TO SEEDANCE AND WORK:

  tool        did it choose seedance_generate over the five other endpoints
  formula     does the prompt carry the parts the model's own guide asks for
  vocab       are the camera terms from the set Seedance understands, not invented
  ratio       does the aspect ratio match what the clip is for (TikTok 9:16, TV 16:9, feed 1:1)
  duration    an integer within the documented 4 to 30 second range
  audio       generate_audio true when the prompt names speech, music or a sound
  fixed       camera_fixed true only for a static shot
  language    is the spoken reply in the language the user wrote in
"""
import argparse, json, os, re, sys

H = "/leonardo/home/userexternal/imisi000/morena"
B = "/leonardo_scratch/large/userexternal/imisi000/morena"
USER, ASSISTANT = "<reserved_0>", "<reserved_1>"
TOOLS_MARKER, TOOL_CALL, TOOL_RESULT = "<reserved_2>", "<reserved_3>", "<reserved_5>"

SHOT = {"extreme close-up", "close-up", "medium shot", "wide shot", "extreme wide shot",
        "two shot", "over-the-shoulder", "insert", "point of view"}
MOVE = {"dolly in", "dolly out", "tracking shot", "crane up", "pan", "tilt", "zoom",
        "steadicam follow shot", "handheld", "rack focus", "orbit shot", "whip pan", "static shot"}
ANGLE = {"low angle", "high angle", "overhead", "eye level", "dutch angle"}
ALLOWED = SHOT | MOVE | ANGLE

VERTICAL = ("tiktok", "reels", "shorts", "status", "phone")
WIDE = ("youtube", "television", "tv", "cinema", "screen", "website", "advert")
SQUARE = ("instagram", "feed", "square")
SOUND = re.compile(r"\bsound includes\b", re.I)
# Any populated "Sound includes ..." clause means the clip has audio. The earlier version required
# a word from a short list, so "Sound includes the rustle of fabric and distant traffic" counted as
# silent and the model was marked wrong for correctly setting generate_audio.
HAS_SOUND = re.compile(r"sound includes\s+\S+(\s+\S+){2,}", re.I)

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


# Orthography first, word lists second.
#
# A ten-word function list cannot classify one short sentence, and this one produced three separate
# false failures before that was obvious: Zulu and Shona had no list at all, and among the
# languages it did cover, "A n se fidio" (we are making a video) scored as non-Yoruba because the
# list held "mo" but not the single letter "a". Both readings were perfect Yoruba.
#
# These languages are separable by their marks long before their vocabulary. Yoruba writes the
# under-dot on e and o and s; Igbo dots u, o and i; Hausa uses hooked letters. Those are decisive
# when present, so they are checked first and the word lists only break ties.
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


def first_json(text):
    s = text
    for st in (TOOL_RESULT, "<|tool_result|>", ASSISTANT, USER):
        s = s.split(st)[0]
    i = s.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(s[i:])[0]
    except Exception:
        return None


def blocks(text):
    out, cur, key = {}, [], None
    for line in (text or "").splitlines():
        m = re.match(r"^(USER|CALL|RESULT|ANSWER)\s*:\s*(.*)$", line.strip())
        if m:
            if key:
                out[key] = "\n".join(cur).strip()
            key, cur = m.group(1), [m.group(2)]
        elif key:
            cur.append(line)
    if key:
        out[key] = "\n".join(cur).strip()
    return out


def wanted_ratio(brief):
    b = (brief or "").lower()
    if any(w in b for w in VERTICAL):
        return "9:16"
    if any(w in b for w in SQUARE):
        return "1:1"
    if any(w in b for w in WIDE):
        return "16:9"
    return None


def score(call, rec, answer):
    a = (call or {}).get("arguments") or {}
    p = str(a.get("prompt", ""))
    low = p.lower()
    terms = [t for t in ALLOWED if t in low]
    # Anything after "The camera uses" that is not an allowed term is an invented camera word.
    tail = low.split("the camera uses", 1)[1][:160] if "the camera uses" in low else ""
    invented = bool(re.search(r"\b(bokeh pan|snap zoom|hyperlapse|drone swoop|vertigo|parallax)\b", tail))
    want = wanted_ratio((rec.get("input") or {}).get("brief"))
    dur = a.get("duration")
    mv = [t for t in MOVE if t in low]
    return {
        "tool": (call or {}).get("name") == "seedance_generate",
        "formula": ("the camera uses" in low) and ("the image is" in low or bool(SOUND.search(p))),
        "vocab": bool(terms) and not invented,
        "ratio": (want is None) or (a.get("aspect_ratio") == want),
        "duration": isinstance(dur, int) and 4 <= dur <= 30,
        "audio": (a.get("generate_audio") is True) == bool(HAS_SOUND.search(p)),
        "fixed": (a.get("camera_fixed") is True) == ("static shot" in mv),
        "language": lang_guess(answer) == rec.get("lang"),
        "prompt": p[:200],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokenizer", default=f"{B}/data/tokenizer/tokenizer.json")
    ap.add_argument("--name", default="model")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", default=f"{B}/eval/seedance_holdout.json")
    a = ap.parse_args()

    sys.path.insert(0, f"{H}/eval")
    from modelio import build_runner

    recs = []
    for line in open(a.holdout, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not str(r.get("id", "")).startswith("t6-hold-"):
            continue
        b = blocks(r.get("output"))
        if b.get("USER") and r.get("menu"):
            r["_user"] = b["USER"]
            recs.append(r)
        if len(recs) >= a.limit:
            break
    if not recs:
        sys.exit(f"no held-out records in {a.holdout}")
    print(f"[seedance] {len(recs)} held-out briefs", flush=True)

    runner = build_runner({"kind": "morena", "name": a.name, "ckpt": a.ckpt,
                           "config": a.config, "tokenizer": a.tokenizer})

    def gen(ps):
        return [o["text"] if isinstance(o, dict) else o
                for o in runner.generate(ps, max_new_tokens=320, temperature=0.0, top_p=1.0)]

    rows = []
    for r in recs:
        mj = json.dumps(r["menu"], ensure_ascii=False, separators=(",", ":"))
        head = f"{USER}\n{TOOLS_MARKER}{mj}\n{r['_user'].strip()}\n{ASSISTANT}"
        raw = gen([head + TOOL_CALL])[0]
        call = first_json(TOOL_CALL + raw)
        answer = ""
        if call:
            hist = (head + TOOL_CALL + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
                    + TOOL_RESULT + '{"job_id":"j_1","status":"processing"}' + ASSISTANT)
            answer = gen([hist])[0].split(USER)[0].strip()
        rows.append(dict(score(call, r, answer), lang=r.get("lang"), answer=answer[:120]))

    n = len(rows)
    keys = ["tool", "formula", "vocab", "ratio", "duration", "audio", "fixed", "language"]
    summary = {k: 100.0 * sum(bool(x[k]) for x in rows) / n for k in keys}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"name": a.name, "n": n, "summary": summary, "rows": rows},
              open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"\nHELD-OUT Seedance, {n} briefs, {a.name}")
    for k in keys:
        print(f"  {k:9s} {summary[k]:6.1f}%")
    ok = [x for x in rows if all(x[k] for k in keys)]
    print(f"  {'POSTABLE':9s} {100*len(ok)/n:6.1f}%   (every check passing on the same brief)")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
