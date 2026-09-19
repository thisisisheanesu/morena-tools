#!/bin/bash
#SBATCH -J tools-eval
#SBATCH -A aih4a_vamboai
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 01:00:00
#SBATCH --output=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.err
#
# Base vs fine-tuned, both sizes, on the held-out Paystack menu. One GPU, four checkpoints loaded
# one at a time. MORENA_LOCAL=1 keeps eval/common.py from reaching for Modal, which is gone.
set -euo pipefail
H=/leonardo/home/userexternal/imisi000/morena
B=/leonardo_scratch/large/userexternal/imisi000/morena
mkdir -p $H/logs $B/eval
source $B/venv/bin/activate
cd $H/eval
export MORENA_LOCAL=1 HF_HUB_OFFLINE=1 HF_HOME=$B/hf_cache PYTHONUNBUFFERED=1
python -u $H/eval/probe_tools_v2.py --out $B/eval/tools_v2.json --models "${EVAL_MODELS:-all}"
echo "[tools-eval] done at $(date)"
