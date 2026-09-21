#!/bin/bash
#SBATCH -J sd-eval
#SBATCH -A aih4a_vamboai
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=8 --mem=80G
#SBATCH -t 01:30:00
#SBATCH --output=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.err
#
# Held-out Seedance briefs, both fine-tunes. These 400 were generated with their own id prefix and
# never staged into a shard; the corpus builder aborts if one turns up, so this score cannot be
# quietly inflated by a leak.
set -uo pipefail
H=/leonardo/home/userexternal/imisi000/morena
B=/leonardo_scratch/large/userexternal/imisi000/morena
source $B/venv/bin/activate
export MORENA_LOCAL=1 HF_HUB_OFFLINE=1 HF_HOME=$B/hf_cache PYTHONUNBUFFERED=1
cd $H/eval
latest () { ls -d $B/runs/$1/ckpt/step_* 2>/dev/null | grep -v '\.tmp$' | sort -V | tail -1; }
for combo in "nano-tools morena0p2b-nano-tools morena0p2b_nano" "mini-tools morena0p5b-mini-tools morena0p5b_mini"; do
  set -- $combo
  CK=$(latest $2)
  [ -n "$CK" ] || { echo "!! no checkpoint for $2"; continue; }
  echo "===== $1  ($CK)"
  python -u $H/eval/probe_seedance.py --holdout $B/eval/holdout/seedance_holdout.jsonl \
    --ckpt "$CK" --config $H/train/configs/$3.json --name $1 --limit 120 \
    --out $B/eval/seedance_$1.json
done
echo "[sd-eval] done at $(date)"
