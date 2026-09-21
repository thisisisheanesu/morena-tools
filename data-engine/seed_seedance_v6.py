#!/usr/bin/env python3
"""Seed African-language video briefs that come out as Seedance 2.5 calls.

Input: somebody describing, in Yoruba or Pidgin or Hausa or Igbo, the video they want.
Output: a tool call whose `prompt` argument can be posted to Seedance 2.5 unedited.

The camera vocabulary is not invented. It is mined from 400 screenplays by
seedance/mine_camera_grammar.py, which reads 520,378 scene sentences and keeps only canonical terms
and their frequencies, never the script text: the scripts are copyrighted, their camera grammar is
functional vocabulary, and only the second thing is useful here anyway. Terms are sampled in
proportion to how often real films use them, and shot/movement pairs are drawn from pairs that
actually co-occur, so a wide shot gets a tracking move far more often than a rack focus.

The API surface is the real one: aspect_ratio, resolution, duration, seed, camera_fixed,
generate_audio, and the 4-to-30-second limit.

A HELD-OUT slice is written to a separate file and never enqueued for training. Without it there
is no honest way to say whether the model learned to write a shot or learned these briefs.

usage: seed_seedance_v6.py [n] [out.json] [holdout.json]
"""
import json, random, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))
GRAMMAR = os.path.join(HERE, "..", "seedance", "camera_grammar.json")

# Real Seedance 2.5 tools. The distractors are the model's other endpoints, so choosing between
# them is a real choice rather than a formality.
def menu_for(rng):
    gen = {"name": "seedance_generate",
           "description": "Generate a video clip from a text prompt with Seedance 2.5",
           "arguments": {"prompt": "string", "aspect_ratio": "string", "resolution": "string",
                         "duration": "number", "generate_audio": "boolean",
                         "camera_fixed": "boolean", "seed": "number"}}
    others = [
        {"name": "seedance_image_to_video",
         "description": "Animate a still image into a video clip",
         "arguments": {"image_url": "string", "prompt": "string", "duration": "number",
                       "aspect_ratio": "string"}},
        {"name": "seedance_extend",
         "description": "Extend an existing generated clip by a few more seconds",
         "arguments": {"video_id": "string", "prompt": "string", "duration": "number"}},
        {"name": "seedance_video_edit",
         "description": "Edit an existing clip following an instruction",
         "arguments": {"video_id": "string", "instruction": "string"}},
        {"name": "seedance_job_status",
         "description": "Check whether a generation job has finished",
         "arguments": {"job_id": "string"}},
        {"name": "asset_upload",
         "description": "Upload a reference image or audio file for use in a generation",
         "arguments": {"file_url": "string", "kind": "string"}},
    ]
    rng.shuffle(others)
    menu = [gen] + others[:rng.choice([3, 4, 5])]
    rng.shuffle(menu)
    return menu


# What the clip is for. Carries the aspect ratio and duration implicitly, which is the inference
# the model has to make without being told the numbers.
BRIEFS = [
    "a market trader wants a short clip for her WhatsApp status",
    "a musician wants a vertical teaser for TikTok",
    "a church wants an announcement clip for the screen at the front",
    "a football fan wants a Reels clip of a street match",
    "a jollof restaurant wants a square clip for Instagram",
    "a bank wants a fifteen second advert for television",
    "a student filmmaker wants an establishing shot for a short film",
    "a fashion label wants a lookbook clip for YouTube",
    "a farmer wants to show his harvest to a buyer abroad",
    "a wedding planner wants a teaser for the couple's status",
    "a radio station wants a visual for a song, cinema aspect",
    "a barber wants a clip of the shop for Shorts",
    "a tech startup wants a thirty second explainer for its website",
    "a grandmother wants a clip of the compound at dawn to send to her children",
    "a tour guide wants a wide cinematic view of the coastline",
    "a dancer wants a slow motion clip for Reels",
]

NG = ["pcm", "yor", "ibo", "hau"]
OTHER = ["swh", "sna", "zul", "eng"]


def weighted(rng, counter, k=1):
    terms = list(counter.keys())
    if not terms:
        return ["medium shot"] * k
    weights = [counter[t] for t in terms]
    return rng.choices(terms, weights=weights, k=k)


def main(n=6000, out="jobs/seedance_v6.json", holdout="jobs/seedance_v6_holdout.json", n_hold=400):
    g = json.load(open(GRAMMAR, encoding="utf-8"))
    V, PAIRS = g["vocab"], g["pairs"]
    rng = random.Random(20260922)
    pair_keys = list(PAIRS.keys())
    pair_w = [PAIRS[k] for k in pair_keys]

    jobs = []
    for i in range(n + n_hold):
        # Shot and movement together, drawn from combinations real films actually use.
        if pair_keys and rng.random() < 0.72:
            shot, move = rng.choices(pair_keys, weights=pair_w, k=1)[0].split("|")
        else:
            shot = weighted(rng, V["shot"])[0]
            move = weighted(rng, V["move"])[0]
        jobs.append({
            "type": "seedance", "id": f"t6-sd-{i:05d}", "phase": 9,
            "lang": rng.choice(NG * 3 + OTHER),
            "menu": menu_for(rng), "target": "seedance_generate",
            "menu_id": f"sd-{i}",
            "brief": rng.choice(BRIEFS),
            "shot": shot, "move": move,
            "angle": weighted(rng, V["angle"])[0],
            "time": weighted(rng, V["time"])[0],
        })
    rng.shuffle(jobs)
    hold, train = jobs[:n_hold], jobs[n_hold:]
    # Distinct ids, so a held-out record can never be swept into a training build by a glob or a
    # careless join. The split has to be impossible to lose by accident, not merely intended.
    for j in hold:
        j["id"] = j["id"].replace("t6-sd-", "t6-hold-")
    json.dump(train, open(out, "w"), ensure_ascii=False)
    json.dump(hold, open(holdout, "w"), ensure_ascii=False)
    from collections import Counter
    print(f"train {len(train):,} -> {out}")
    print(f"HELD OUT {len(hold):,} -> {holdout}   (never enqueued with the training batch)")
    print("  langs:", dict(Counter(j["lang"] for j in train).most_common()))
    print("  shots:", dict(Counter(j["shot"] for j in train).most_common(5)))
    print("  moves:", dict(Counter(j["move"] for j in train).most_common(5)))
    print("  example:", train[0]["lang"], "|", train[0]["shot"], "+", train[0]["move"],
          "|", train[0]["brief"][:46])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000,
         sys.argv[2] if len(sys.argv) > 2 else "jobs/seedance_v6.json",
         sys.argv[3] if len(sys.argv) > 3 else "jobs/seedance_v6_holdout.json")
