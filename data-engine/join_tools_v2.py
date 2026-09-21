#!/usr/bin/env python3
"""Join generated tool records back to the menu each was generated against, and lay them out for
the cluster builder.

The worker stores only `menu_id` in the record, not the menu, because a 20,000-menu corpus embedded
once per record is ~23MB of duplicated JSON in the output bucket. The menus live in the seed file
that produced the jobs, so they are joined back here on the job `id`, which is unique per record.

Emits $RAW/<type>/<lang>/part-000.jsonl, the layout build_tools_v18.py globs for.

usage: join_tools_v2.py --seeds jobs/tools_v2_pilot.json --out staged/synth_raw outputs/*.jsonl
"""
import argparse, collections, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    # Six seed files now, one per wave. The menus live in the seeds and nowhere else, so a
    # record whose seed file is missing cannot be rendered at all, which is why the count of
    # unmatched records is printed rather than swallowed.
    ap.add_argument("--seeds", nargs="+", default=["jobs/tools_v2_pilot.json"])
    ap.add_argument("--out", default="staged/synth_raw")
    ap.add_argument("inputs", nargs="+", help="jsonl files of generated records (clean.jsonl)")
    a = ap.parse_args()

    seeds = {}
    for f in a.seeds:
        n0 = len(seeds)
        for j in json.load(open(f, encoding="utf-8")):
            seeds[j["id"]] = j
        print(f"[join]   {f}: {len(seeds)-n0:,} jobs")
    print(f"[join] {len(seeds):,} seed jobs across {len(a.seeds)} files")

    groups, miss, dup, bad = collections.defaultdict(list), 0, 0, 0
    seen = set()
    for path in a.inputs:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1          # a truncated final line from an interrupted dump
                continue
            s = seeds.get(r.get("id"))
            if not s:
                miss += 1
                continue
            if r["id"] in seen:          # a requeued job can be written twice
                dup += 1
                continue
            seen.add(r["id"])
            r["menu"] = s["menu"]
            # `current` is the pending call a patch edits. It lives in the seed, and the record's
            # own input.current is the same value; take the seed's so a truncated record cannot
            # carry a half-written one.
            if s.get("current") is not None:
                r.setdefault("input", {})["current"] = s["current"]
            groups[(r["type"], r.get("lang", "eng"))].append(r)

    total = 0
    for (typ, lang), recs in sorted(groups.items()):
        d = os.path.join(a.out, typ, lang)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "part-000.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(recs)
        print(f"  {typ}/{lang}: {len(recs):,}")
    print(f"[join] {total:,} records -> {a.out}   ({miss:,} no seed, {dup:,} duplicate ids, {bad:,} unparseable)")


if __name__ == "__main__":
    main()
