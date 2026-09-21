#!/usr/bin/env python3
"""Second wave: bring every type up to roughly the same size, so the mixture is balanced by RAW
COUNT rather than by duplication.

Duplication was doing the balancing until now, and it is a poor substitute: x3 on a slice means
the model sees the same 3,400 conversations three times, which teaches those conversations rather
than the behaviour behind them. Every type here goes to about 8,000 real records so the dup
factors can all come down.

`tooldisambig` is deliberately NOT scaled. It is the one type that was measured doing harm: at x3
it taught the model to ask when unsure, and terse real requests always look under-specified to a
model holding that prior. It stays where it is, at roughly half the others.

Sessions in this wave carry the corrected prompt, where turn one ACTS instead of opening with a
clarifying question.

usage: seed_balance_v5.py [out.json]
"""
import json, itertools, random, sys

sys.path.insert(0, ".")
from seed_tools_v2 import make_menu, LANGS, NG, CONCEPTS, STYLES, EDITS, AMBIG, MODES
from seed_sessions_v3 import SHAPES, lang_sequence

# type -> how many more, chosen to land each around 8,000 raw once the existing wave is included.
WANT = {"toolcall_api": 3200, "toolrefuse": 3400, "toolpatch": 4000, "toolsession": 4200}


def main(out="jobs/tool_balance_v5.json"):
    rng = random.Random(20260921)
    def lang(): return rng.choice(NG + NG + LANGS)
    weights = list(itertools.chain.from_iterable([[c] * w for c, (_, _, w) in CONCEPTS.items()]))
    jobs, mid = [], 100000

    for i in range(WANT["toolcall_api"]):
        mid += 1; style, concepts, menu = make_menu(rng)
        tgt = rng.choice([c for c in weights if c in concepts]) if any(c in concepts for c in weights) else concepts[0]
        jobs.append({"type": "toolcall_api", "lang": lang(), "id": f"t5-api-{i:05d}", "phase": 8,
                     "menu": menu, "target": dict(STYLES)[style][tgt],
                     "menu_id": f"{style}-{mid}", "domain": "Nigerian fintech"})

    for i in range(WANT["toolpatch"]):
        mid += 1; style, concepts, menu = make_menu(rng)
        tname = dict(STYLES)[style]["transfer"] if "transfer" in concepts else menu[0]["name"]
        tool = next(t for t in menu if t["name"] == tname)
        # Plausible values, not placeholders: the question a patch example answers is whether an
        # untouched field survives, and a field that was nonsense to begin with cannot answer it.
        filler = {"number": 25000, "string": "RCP_4417"}
        cur = {"name": tname, "arguments": {k: filler.get(v, "NGN") for k, v in tool["arguments"].items()}}
        for k in list(cur["arguments"]):
            if "curr" in k or "ccy" in k: cur["arguments"][k] = "NGN"
            elif "reason" in k or "note" in k or "narration" in k: cur["arguments"][k] = "school fees"
        jobs.append({"type": "toolpatch", "lang": lang(), "id": f"t5-patch-{i:05d}", "phase": 8,
                     "menu": menu, "current": cur, "edit_kind": rng.choice(EDITS),
                     "menu_id": f"{style}-{mid}"})

    for i in range(WANT["toolrefuse"]):
        mid += 1; style, concepts, menu = make_menu(rng)
        jobs.append({"type": "toolrefuse", "lang": lang(), "id": f"t5-ref-{i:05d}", "phase": 8,
                     "menu": menu, "mode": MODES[i % len(MODES)], "menu_id": f"{style}-{mid}"})

    for i in range(WANT["toolsession"]):
        mid += 1; style, concepts, menu = make_menu(rng, n=rng.choice([5, 6, 6, 7]))
        turns = rng.choice([4, 4, 5, 5, 6, 6, 7])
        langs = lang_sequence(rng, turns)
        jobs.append({"type": "toolsession", "id": f"t5-sess-{i:05d}", "phase": 8,
                     "lang": langs[0], "langs": langs, "menu": menu,
                     "menu_id": f"{style}-{mid}", "shape": rng.choice(SHAPES)})

    rng.shuffle(jobs)
    json.dump(jobs, open(out, "w"), ensure_ascii=False)
    from collections import Counter
    print(f"wrote {len(jobs)} jobs -> {out}")
    print("  by type:", dict(Counter(j["type"] for j in jobs)))
    print("  distinct menus:", len({j["menu_id"] for j in jobs}))
    sess = [j for j in jobs if j["type"] == "toolsession"]
    print("  session switches:", dict(sorted(Counter(
        sum(1 for a, b in zip(j["langs"], j["langs"][1:]) if a != b) for j in sess).items())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "jobs/tool_balance_v5.json")
