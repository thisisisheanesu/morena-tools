#!/usr/bin/env python3
"""Filter generated tool data before it becomes training data.

The generator is good but not perfect: it sometimes puts a bank NAME in a bank_code field, invents
an argument the menu never declared, or restates fields in a patch that the user never mentioned.
The last one is fatal, because restating unchanged fields is the exact behaviour docs/44 measured
as corrupting amounts 6 times out of 6. Anything that does it must never reach the model.

usage: validate_tools_v2.py pilot_outputs/  ->  writes clean.jsonl and reports the reject reasons
"""
import json, os, re, sys
from collections import Counter

def parse_blocks(text):
    out, cur, key = {}, [], None
    for line in (text or "").splitlines():
        m = re.match(r'^(USER2|USER|CALL|PATCH|RESULT|ANSWER|DECISION|ASK)\s*:\s*(.*)$', line.strip())
        if m:
            if key: out[key] = "\n".join(cur).strip()
            key, cur = m.group(1), [m.group(2)]
        elif key:
            cur.append(line)
    if key: out[key] = "\n".join(cur).strip()
    return out

def jload(s):
    if not s: return None
    i = s.find("{")
    if i < 0: return None
    try: return json.JSONDecoder().raw_decode(s[i:])[0]
    except Exception: return None

def check(rec):
    t = rec.get("type"); b = parse_blocks(rec.get("output"))
    menu = {x["name"]: x for x in (rec.get("menu") or [])}
    if t == "toolcall_api":
        if not b.get("USER") or not b.get("ANSWER"): return "missing_block"
        c = jload(b.get("CALL"))
        if not c or not c.get("name"): return "no_call"
        tgt = (rec.get("input") or {}).get("target")
        if tgt and c["name"] != tgt: return "wrong_tool_name"
        if menu:
            decl = set((menu.get(c["name"], {}).get("arguments") or {}))
            extra = set(c.get("arguments") or {}) - decl
            if extra: return "undeclared_args"
        return None
    if t == "toolpatch":
        p = jload(b.get("PATCH"))
        if not p or p.get("op") not in ("patch", "cancel", "confirm"): return "bad_op"
        if p["op"] == "patch" and not (p.get("set") or p.get("unset")): return "empty_patch"
        cur = ((rec.get("input") or {}).get("current") or {}).get("arguments") or {}
        setf = p.get("set") or {}
        # the fatal one: restating a field at its existing value teaches whole-call re-emission
        unchanged = [k for k, v in setf.items() if k in cur and str(cur[k]) == str(v)]
        if unchanged: return "restates_unchanged"
        if len(setf) > 3: return "patch_too_wide"
        return None
    if t == "toolrefuse":
        d = jload(b.get("DECISION"))
        if not d or d.get("call") is not None: return "emitted_a_call"
        if d.get("reason") not in ("no_tool","benign","unsafe","missing_info"): return "bad_reason"
        if not b.get("ANSWER"): return "no_answer"
        return None
    if t == "tooldisambig":
        if not b.get("ASK") or not b.get("USER2"): return "missing_turn"
        if b["ASK"].count("?") != 1: return "not_one_question"
        c = jload(b.get("CALL"))
        if not c or not c.get("name"): return "no_call"
        if menu:
            extra = set(c.get("arguments") or {}) - set((menu.get(c["name"], {}).get("arguments") or {}))
            if extra: return "undeclared_args"
        return None
    return "unknown_type"

def main(d):
    rej, keep = Counter(), []
    for root, _, files in os.walk(d):
        for f in files:
            if not f.endswith(".jsonl"): continue
            for line in open(os.path.join(root, f), encoding="utf-8"):
                if not line.strip(): continue
                r = json.loads(line)
                why = check(r)
                if why: rej[f"{r.get('type')}:{why}"] += 1
                else: keep.append(r)
    out = os.path.join(d, "clean.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in keep: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = len(keep) + sum(rej.values())
    print(f"kept {len(keep)}/{tot} ({100*len(keep)/max(tot,1):.1f}%) -> {out}")
    for k, v in rej.most_common(): print(f"  reject {k:38s} {v}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot_outputs")
