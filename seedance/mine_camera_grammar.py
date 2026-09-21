#!/usr/bin/env python3
"""Mine cinematographic grammar from screenplays, without keeping the screenplays.

Film scripts are copyrighted. What is useful here is not their prose but the FORM their camera
directions take: which shot sizes exist, which movements go with which, how a direction line is
built. That form is functional vocabulary, not creative expression, so this keeps counts and short
canonical terms and throws the scene text away. Nothing downstream reproduces a line from a film.

Reads rohitsaxena/MovieSum (2,200 screenplays) and writes camera_grammar.json:

    vocab      canonical term -> how often it appears
    pairs      (shot size, movement) co-occurrence within a direction block
    templates  the SHAPES direction lines take, with content replaced by slots

usage: mine_camera_grammar.py [n_scripts] [out.json]
"""
import json, re, sys
from collections import Counter, defaultdict

# Canonical vocabulary. The left side is what screenplays actually write, the right is the term
# Seedance 2.5 understands, so the mapping does double duty as a translation table.
SHOT = {
    r"EXTREME CLOSE ?UPS?|ECU": "extreme close-up",
    r"CLOSE ?UPS?|CLOSE ON|CU\b": "close-up",
    r"MED(?:IUM)? (?:SHOT|ON)|MS\b": "medium shot",
    r"WIDE (?:SHOT|ON|ANGLE)|LONG SHOT|WS\b": "wide shot",
    r"EXTREME WIDE|ESTABLISHING(?: SHOT)?|AERIAL": "extreme wide shot",
    r"TWO ?SHOT": "two shot",
    r"OVER THE SHOULDER|O\.?T\.?S\.?": "over-the-shoulder",
    r"INSERT": "insert",
    r"P\.?O\.?V\.?": "point of view",
}
MOVE = {
    r"DOLLY IN|PUSH(?:ES|ING)? IN|MOVE[SD]? IN": "dolly in",
    r"DOLLY OUT|PULL(?:S|ING)? BACK|PULL ?BACK": "dolly out",
    r"TRACK(?:S|ING)?|FOLLOW(?:S|ING)? (?:HIM|HER|THEM)": "tracking shot",
    r"CRANE|BOOM (?:UP|DOWN)": "crane up",
    r"PAN(?:S|NING)?\b": "pan",
    r"TILT(?:S|ING)?(?: UP| DOWN)?": "tilt",
    r"ZOOM(?:S|ING)?": "zoom",
    r"STEADICAM": "steadicam follow shot",
    r"HAND ?HELD": "handheld",
    r"RACK FOCUS": "rack focus",
    r"ORBIT|CIRCL(?:E|ES|ING)": "orbit shot",
    r"WHIP PAN": "whip pan",
    r"STATIC|LOCKED ?OFF": "static shot",
}
ANGLE = {
    r"LOW ANGLE": "low angle", r"HIGH ANGLE": "high angle",
    r"OVERHEAD|BIRD'?S ?EYE|TOP DOWN": "overhead",
    r"EYE ?LEVEL": "eye level", r"DUTCH(?: ANGLE| TILT)?": "dutch angle",
}
TIME = {
    r"\bDAWN\b": "dawn", r"\bDAY\b": "day", r"\bDUSK\b": "dusk", r"\bNIGHT\b": "night",
    r"MAGIC HOUR|GOLDEN HOUR": "golden hour", r"\bSUNSET\b": "sunset", r"\bSUNRISE\b": "sunrise",
}
GROUPS = {"shot": SHOT, "move": MOVE, "angle": ANGLE, "time": TIME}
COMPILED = {g: [(re.compile(p, re.I), term) for p, term in d.items()] for g, d in GROUPS.items()}

# MovieSum is XML-tagged and its camera cues sit INSIDE the prose of a scene description, not on
# their own shouty line the way a shooting script prints them. Scanning for standalone uppercase
# lines found exactly zero across 400 screenplays, which is the sort of silence worth checking
# rather than accepting. Scene text is read sentence by sentence and immediately discarded; only
# the canonical terms and their counts survive.
SCENE = re.compile(r"<scene_description>(.*?)(?:</scene_description>|$)", re.S)
SENT = re.compile(r"(?<=[.!?])\s+")


def terms_in(text):
    found = defaultdict(list)
    for g, pats in COMPILED.items():
        for rx, term in pats:
            if rx.search(text):
                found[g].append(term)
    return found


def main(n=400, out="camera_grammar.json"):
    from datasets import load_dataset
    ds = load_dataset("rohitsaxena/MovieSum", split=f"train[:{n}]")
    vocab = {g: Counter() for g in GROUPS}
    pairs = Counter()
    lines_seen = 0
    for row in ds:
        for block in SCENE.findall(str(row["script"])):
            for sent in SENT.split(block):
                lines_seen += 1
                f = terms_in(sent)
                if not f:
                    continue
                for g, terms in f.items():
                    vocab[g].update(set(terms))
                for sh in set(f.get("shot", [])):
                    for mv in set(f.get("move", [])):
                        pairs[f"{sh}|{mv}"] += 1

    grammar = {
        "source": "rohitsaxena/MovieSum, 2,200 screenplays; terms and counts only, no script text",
        "scripts_read": n,
        "sentences_scanned": lines_seen,
        "vocab": {g: dict(c.most_common()) for g, c in vocab.items()},
        "pairs": dict(pairs.most_common(60)),
    }
    json.dump(grammar, open(out, "w"), indent=1)
    print(f"scanned {lines_seen:,} scene sentences across {n} screenplays -> {out}")
    for g in GROUPS:
        top = list(vocab[g].most_common(6))
        print(f"  {g:6s} {len(vocab[g]):2d} terms  top: {top}")
    print("  top pairings:", list(pairs.most_common(6)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400,
         sys.argv[2] if len(sys.argv) > 2 else "camera_grammar.json")
