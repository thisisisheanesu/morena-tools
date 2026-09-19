#!/usr/bin/env python3
"""Seed LONG multi-turn, language-switching tool sessions.

Everything shipped so far is short. toolcall_api is one turn, toolpatch and tooldisambig are two.
So the model has never seen a fourth turn, and it has never seen the user change language without
starting over. Both are the normal case in Lagos, and both are what the demo does within about
fifteen seconds of anyone touching it.

Two things are being taught here that the existing types cannot teach:

  length      4 to 7 turns holding one thread, where "send am again" and "make that one 3000" only
              resolve against earlier turns. The existing patch type is a single edit against a
              call that was handed to it in the prompt, so nothing has to be remembered.

  switching   the user starts in Pidgin, moves to Yoruba, maybe ends in English, and the assistant
              follows each time WITHOUT restarting or commenting on it. Sequences are built so the
              switch always lands mid-conversation, never on turn one, because a switch on the
              first turn is just a monolingual conversation in a different language.

usage: seed_sessions_v3.py [n] [out.json]
"""
import json, random, sys

sys.path.insert(0, ".")
from seed_tools_v2 import make_menu, LANGS, NG

SHAPES = [
    "a transfer that gets corrected twice before it is confirmed",
    "check the balance, then send money, then ask what the reference was",
    "look up an account name, save the person as a recipient, then pay them",
    "start a refund, change the amount, then cancel it and say why",
    "pay one person, then say 'send the same to my brother' with a different name",
    "a transfer where the user changes their mind about the reason, then the amount",
    "verify a payment, find it failed, then start a fresh payment",
    "buy airtime, then ask an ordinary question the tools cannot answer, then send money",
    "send money, thank the assistant, then ask it to do the same again tomorrow",
    "create a customer, then a dedicated account for them, then read the number back",
]


def lang_sequence(rng, n):
    """A sequence that switches at least once, never on the first turn.

    Weighted to Nigerian languages and to English, because English is what people switch INTO when
    they want to be precise about an amount, and Pidgin is what they switch back into. A run of the
    same language between switches is deliberate: switching every single turn is not how anyone
    speaks and would teach the model that a new language means a new topic.
    """
    a = rng.choice(NG)
    b = rng.choice([x for x in NG + ["eng", "eng"] if x != a])
    switch = rng.randint(1, n - 1)          # index of the first turn in the new language
    seq = [a] * switch + [b] * (n - switch)
    # A third language needs a turn left after the first switch to land on.
    if n >= 5 and switch + 1 <= n - 1 and rng.random() < 0.45:
        c = rng.choice([x for x in NG + ["eng"] if x != b] + [a])
        back = rng.randint(switch + 1, n - 1)
        seq = seq[:back] + [c] * (n - back)
    return seq


def main(n=4000, out="jobs/tool_sessions_v3.json"):
    rng = random.Random(20260920)
    jobs = []
    for i in range(n):
        style, concepts, menu = make_menu(rng, n=rng.choice([5, 6, 6, 7]))
        turns = rng.choice([4, 4, 5, 5, 6, 6, 7])
        langs = lang_sequence(rng, turns)
        jobs.append({"type": "toolsession", "id": f"t3-sess-{i:05d}", "phase": 6,
                     "lang": langs[0], "langs": langs, "menu": menu,
                     "menu_id": f"{style}-s{i}", "shape": rng.choice(SHAPES)})
    json.dump(jobs, open(out, "w"), ensure_ascii=False)
    from collections import Counter
    print(f"wrote {len(jobs)} sessions -> {out}")
    print("  turns:", dict(sorted(Counter(len(j['langs']) for j in jobs).items())))
    print("  distinct menus:", len({j["menu_id"] for j in jobs}))
    print("  switches per session:",
          dict(sorted(Counter(sum(1 for a, b in zip(j["langs"], j["langs"][1:]) if a != b)
                              for j in jobs).items())))
    print("  first-turn langs:", dict(Counter(j["langs"][0] for j in jobs)))
    print("  example:", jobs[0]["langs"], "|", jobs[0]["shape"][:50])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4000,
         sys.argv[2] if len(sys.argv) > 2 else "jobs/tool_sessions_v3.json")
