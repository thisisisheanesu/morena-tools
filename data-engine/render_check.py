#!/usr/bin/env python3
"""Render a few joined records with the cluster builder's own segments() and eyeball them.

The builder runs on Leonardo where there is no way to look at what it produced without reading
binary shards back through a tokenizer. This runs the identical function locally on the identical
records, so a format mistake is caught before 20,000 examples are baked into a shard.

What to look for, in order of how badly each one breaks training:
  * the loss mask. `>` marks trained text. A menu or a user turn inside a `>` segment means the
    model is being trained to write the prompt.
  * `<reserved_1>\n` before prose and `<reserved_1><reserved_3>` before a call, never swapped.
  * a patch that restates a field the user did not mention. That is the docs/44 corruption bug
    reproduced in the training data itself.
"""
import importlib.util, json, sys, os

spec = importlib.util.spec_from_file_location(
    "b18", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "leonardo", "build_tools_v18.py"))
b18 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b18)

SHOW = {"toolcall_api": 1, "toolpatch": 2, "toolrefuse": 1, "tooldisambig": 1}
shown, stats = {}, {}
for path in sys.argv[1:]:
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = r.get("type")
        segs = b18.segments(r)
        s = stats.setdefault(t, [0, 0])
        s[0 if segs else 1] += 1
        if segs and shown.get(t, 0) < SHOW.get(t, 1):
            shown[t] = shown.get(t, 0) + 1
            print("=" * 78); print(f"{t}   lang={r.get('lang')}   id={r.get('id')}")
            print("=" * 78)
            for text, train in segs:
                mark = ">" if train else " "
                body = text if len(text) < 700 else text[:340] + f"\n   ...[{len(text)-680} chars]...\n" + text[-340:]
                for ln in body.split("\n"):
                    print(f" {mark} {ln}")
            print()

print("-" * 78)
for t, (ok, bad) in sorted(stats.items()):
    tot = ok + bad
    print(f"{t:15s} renderable {ok:5,}/{tot:5,}  ({100*ok/max(tot,1):.1f}%)")
