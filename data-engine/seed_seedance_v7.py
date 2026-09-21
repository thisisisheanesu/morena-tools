#!/usr/bin/env python3
"""Seedance wave two: Kinyarwanda, Swahili, Shona and English, and multi-shot sequences.

Two gaps in wave one.

LANGUAGES. It was weighted three to one toward the four Nigerian languages, and Kinyarwanda was
absent entirely: 0 records. MORENA has mt_kin pretraining data so the base model knows the
language, but nothing in the tool corpus ever asked it to write a shot in it. This wave inverts
the weighting.

SEQUENCES. Every wave-one record was a single shot. Seedance 2.5 divides a clip with "Shot N" and
handles up to 30 seconds, which is three or four beats, and that is what anyone actually wants:
the market waking up, then the trader setting out her tomatoes, then the first customer. A
sequence needs a second and third camera framing that DIFFER from the first, so each job carries
extra terms drawn from the mined grammar and the prompt is told not to reuse one.

usage: seed_seedance_v7.py [n] [out.json] [holdout.json]
"""
import json, random, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
GRAMMAR = os.path.join(HERE, "..", "seedance", "camera_grammar.json")
sys.path.insert(0, HERE)
from seed_seedance_v6 import menu_for, weighted

# The four asked for, plus a thin tail of the wave-one languages so they do not decay.
FOCUS = ["kin", "swh", "sna", "en"]
TAIL = ["pcm", "yor", "ibo", "hau"]

# Sequences need briefs with more than one beat in them.
SEQ_BRIEFS = [
    "a market trader wants to show her stall waking up, setting out, then the first customer",
    "a musician wants a teaser that moves from the empty stage to the crowd to the first chord",
    "a tour guide wants the coastline from far away, then closer, then a detail of the water",
    "a restaurant wants the pot, then the plating, then someone eating",
    "a tailor wants the cloth, then the machine, then the finished garment on someone",
    "a farmer wants the field at dawn, the harvest, then the load going to market",
    "a church wants the empty hall, the choir arriving, then the congregation standing",
    "a barber wants the empty chair, the cut in progress, then the customer's face at the end",
    "a school wants the gate opening, the classroom filling, then a child reading aloud",
    "a football club wants the street outside, the tunnel, then the first touch of the ball",
]
SINGLE_BRIEFS = [
    "a trader wants a short clip for her WhatsApp status",
    "a musician wants a vertical teaser for TikTok",
    "a bank wants a fifteen second advert for television",
    "a student filmmaker wants an establishing shot for a short film",
    "a cooperative wants a square clip for Instagram",
    "a tour operator wants a wide cinematic view for YouTube",
    "a grandmother wants a clip of the compound at dawn to send to her children",
    "a tech startup wants a thirty second explainer for its website",
]


def main(n=7000, out="jobs/seedance_v7.json", holdout="jobs/seedance_v7_holdout.json", n_hold=400):
    g = json.load(open(GRAMMAR, encoding="utf-8"))
    V, PAIRS = g["vocab"], g["pairs"]
    rng = random.Random(20260923)
    pair_keys = list(PAIRS.keys()); pair_w = [PAIRS[k] for k in pair_keys]

    jobs = []
    for i in range(n + n_hold):
        if pair_keys and rng.random() < 0.72:
            shot, move = rng.choices(pair_keys, weights=pair_w, k=1)[0].split("|")
        else:
            shot, move = weighted(rng, V["shot"])[0], weighted(rng, V["move"])[0]
        # Six in ten are sequences. Enough to teach the shape without drowning the single shot,
        # which is still what a status update or a teaser wants.
        shots = rng.choice([1, 1, 2, 3, 3, 4]) if rng.random() < 0.62 else 1
        job = {
            "type": "seedance", "id": f"t7-sd-{i:05d}", "phase": 10,
            "lang": rng.choice(FOCUS * 4 + TAIL),
            "menu": menu_for(rng), "target": "seedance_generate",
            "menu_id": f"sd7-{i}", "shots": shots,
            "brief": rng.choice(SEQ_BRIEFS if shots > 1 else SINGLE_BRIEFS),
            "shot": shot, "move": move,
            "angle": weighted(rng, V["angle"])[0],
            "time": weighted(rng, V["time"])[0],
        }
        if shots > 1:
            # A second framing that is not the first, so the sequence actually changes.
            alt_s = [t for t in V["shot"] if t != shot] or [shot]
            alt_m = [t for t in V["move"] if t != move] or [move]
            job["shot2"] = rng.choice(alt_s)
            job["move2"] = rng.choice(alt_m)
        jobs.append(job)

    rng.shuffle(jobs)
    hold, train = jobs[:n_hold], jobs[n_hold:]
    for j in hold:
        j["id"] = j["id"].replace("t7-sd-", "t7-hold-")
    json.dump(train, open(out, "w"), ensure_ascii=False)
    json.dump(hold, open(holdout, "w"), ensure_ascii=False)
    from collections import Counter
    print(f"train {len(train):,} -> {out}")
    print(f"HELD OUT {len(hold):,} -> {holdout}")
    print("  langs:", dict(Counter(j["lang"] for j in train).most_common()))
    print("  shots per clip:", dict(sorted(Counter(j["shots"] for j in train).items())))
    ex = next(j for j in train if j["shots"] > 2)
    print(f"  example: {ex['lang']} {ex['shots']}-shot | {ex['shot']}+{ex['move']} then {ex['shot2']}+{ex['move2']}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7000,
         sys.argv[2] if len(sys.argv) > 2 else "jobs/seedance_v7.json",
         sys.argv[3] if len(sys.argv) > 3 else "jobs/seedance_v7_holdout.json")
