#!/bin/bash
#SBATCH -J tools-build
#SBATCH -A aih4a_vamboai
#SBATCH -p lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH -t 02:00:00
#SBATCH --output=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.err
#
# Tokenise the v2 tool corpus into loss-masked shards. CPU only, so it goes on the serial partition
# and does not spend GPU hours. The lrd_all_serial QOS caps a user at cpu=8, mem=30800M and 4h,
# anything else is queued there.
set -euo pipefail
H=/leonardo/home/userexternal/imisi000/morena
B=/leonardo_scratch/large/userexternal/imisi000/morena
mkdir -p $H/logs
source $B/venv/bin/activate

# DUP. Two rounds of guessing already cost two training runs, so the reasoning is written down in
# build_tools_v18.py next to the default. Short version: the model was not over-REFUSING, it was
# over-ASKING, and tooldisambig at x3 was a quarter of the corpus teaching "when unsure, ask".
# Disambig drops to x1, plain calls rise to x3, and the sessions come in at x2 because each one
# carries several in-context calls, which is the counterweight this needs.
python -u $H/data/leonardo/build_tools_v18.py \
  --raw     $B/synth_raw \
  --out-dir $B/data/sft18_tools \
  --dup "toolcall_api=2,toolact=3,seedance=2,toolpatch=2,toolrefuse=1,tooldisambig=1,toolsession=2"

python - <<'PY'
import json
d = json.load(open("/leonardo_scratch/large/userexternal/imisi000/morena/data/sft18_tools/index.json"))
n = sum(s["n_tokens"] for s in d["shards"])
print(f"[check] {n:,} tokens, {d['n_response_tokens']:,} trained ({100*d['response_frac']:.1f}%)")
assert d["vocab_size"] == 65536, d["vocab_size"]
assert d["eos_id"] == 2, d["eos_id"]
# A response fraction this far from the SFT corpus's 64.7% would mean the mask is inverted or the
# prompts are being trained on. Tool data is prompt-heavy (the menu is most of it), so the expected
# range is low, but not near zero and not near one.
assert 0.05 < d["response_frac"] < 0.60, d["response_frac"]
print("[check] ok")
PY
echo "[tools-build] done at $(date)"
