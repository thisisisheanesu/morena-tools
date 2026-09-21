#!/usr/bin/env python3
"""Seed ACT-IMMEDIATELY examples, and reinforce answering in the user's language.

Two measured failures, one file.

ACTING. Left to decide for itself whether to call a tool, the fine-tune manages 50%. The rest of
the time it asks a clarifying question, and the question is often about a field the tool does not
have ("which bank?" on a transfer whose schema has no bank). The corpus taught it to ask: between
tooldisambig and the many session turns that open by checking a detail, asking is well represented
and acting on a terse request is not. Every example here is a short request that is already
actionable, answered by acting.

LANGUAGE. It drifts into English on the reply. The sessions taught switching BETWEEN languages but
comparatively little about staying put, so every example here pins the answer to the language the
user wrote in and says so in the prompt.

Deliberately skewed to the tools that stall worst in practice: transfers and refunds, which have
the most optional arguments and therefore the most for a nervous model to ask about.

usage: seed_act_v4.py [n] [out.json]
"""
import json, random, sys

sys.path.insert(0, ".")
from seed_tools_v2 import make_menu, LANGS, NG, CONCEPTS, STYLES

# Weighted toward the concepts the live model hesitates on. balance is included because it takes no
# arguments at all and still drew "I no fit check your balance directly".
WEIGHTS = {"transfer": 5, "refund": 4, "recipient": 3, "verify": 3, "balance": 3,
           "resolve": 2, "vaccount": 2, "customer": 2, "charge": 2, "banks": 1}


def main(n=8000, out="jobs/tool_act_v4.json"):
    rng = random.Random(20260920)
    pool = [c for c, w in WEIGHTS.items() for _ in range(w)]
    jobs, skipped = [], 0
    for i in range(n):
        style, concepts, menu = make_menu(rng, n=rng.choice([5, 6, 6, 7]))
        avail = [c for c in pool if c in concepts]
        if not avail:
            skipped += 1
            continue
        concept = rng.choice(avail)
        name = dict(STYLES)[style][concept]
        # Nigerian languages tripled here: these are the ones the demo is judged in.
        lang = rng.choice(NG * 3 + LANGS)
        jobs.append({"type": "toolact", "id": f"t4-act-{i:05d}", "phase": 7,
                     "lang": lang, "menu": menu, "target": name,
                     "menu_id": f"{style}-a{i}"})
    json.dump(jobs, open(out, "w"), ensure_ascii=False)
    from collections import Counter
    print(f"wrote {len(jobs)} jobs -> {out}  ({skipped} menus had no weighted concept)")
    print("  langs:", dict(Counter(j["lang"] for j in jobs).most_common()))
    print("  distinct menus:", len({j["menu_id"] for j in jobs}))
    print("  example target:", jobs[0]["target"], "| lang", jobs[0]["lang"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8000,
         sys.argv[2] if len(sys.argv) > 2 else "jobs/tool_act_v4.json")
