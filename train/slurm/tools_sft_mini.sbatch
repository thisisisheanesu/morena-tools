#!/bin/bash
#SBATCH -J tools-mini
#SBATCH -A aih4a_vamboai
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH --output=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.err
#
# MORENA 0.5B mini, tool-calling fine-tune on ONE A100.
#
# WHY. docs/44 measured the released lineage at 9% correct tool against a menu built from the real
# Paystack OpenAPI spec. The cause is in the data, not the model: build_sft_v17.py assembles every
# toolcall example's menu from one global pool of ~15 tool names scanned out of the corpus, so the
# model memorised that vocabulary instead of learning to read a schema. Two further failures came
# out of the same probe: asked to change one field of a pending call it re-emitted the whole call
# and corrupted an untouched amount 6 times out of 6, and it calls tools on requests that need none.
#
# WHAT CHANGED. 20,000 records over 20,000 DISTINCT menus across five naming conventions, plus a
# patch type that emits only changed fields, refusals including benign-answered, and disambiguation.
#
# ONE GPU on purpose. 503M params at bf16 is ~1.0GB of weights; optimiser state and activations fit
# an A100 64GB several times over, so there is nothing to shard.
#
# FSDP is nonetheless left ON at world=1. --no-fsdp leaves the parameters in fp32 and flash-attn
# then refuses outright: "FlashAttention only support fp16 and bf16 data type". The bf16 cast comes
# from FSDP's MixedPrecisionPolicy, not from the autocast around the forward. The alternatives are
# both worse: --attn sdpa silently drops cross-document masking on packed sequences, and sdpa_mask
# materialises a (B,1,T,T) mask at roughly twice the cost. At world=1 fully_shard has no one to
# communicate with, so it costs nothing and the numerics match how these models were trained.
#
# GEOMETRY. The config's global_batch_seqs=256 is a PRETRAINING batch: 1.05M tokens per step, which
# on one GPU is ~50s/step and puts an 800-step fine-tune past 40 hours. Overridden to 32 seqs
# (131k tokens/step). Fine-tuning a capability into an already-converged checkpoint does not need a
# million-token batch and is better off with more steps at a smaller one.
#
# STEPS. The built corpus is 11,718,448 tokens, of which 11.5% are trained (tool data is
# prompt-heavy: the menu is most of every example). At 70% of a 131k-token step that is ~128 steps
# per epoch, so 350 steps is ~2.7 epochs. The first draft said 800, which is 6.2 epochs and, with
# the x3 duplication on patch and disambiguation, roughly eighteen exposures of each unique record.
# A small model picks up a format in two or three passes; the rest is memorisation.
#
# SCHEDULE. --anneal is the right mechanism for grafting onto a finished run: it starts the WSD
# decay at the resumed step and sets total_steps to step+N, so the LR lands at zero exactly when
# the fine-tune ends. Without it the resumed step (11,500) already equals the config's total_steps
# and the process would exit having trained nothing.
set -euo pipefail
H=/leonardo/home/userexternal/imisi000/morena
B=/leonardo_scratch/large/userexternal/imisi000/morena
mkdir -p $H/logs
source $B/venv/bin/activate

DATA_ROOT=$B/data
OUT=$B/runs/morena0p5b-mini-tools
BASE=${MINI_CKPT:-$B/runs/morena0p5b-mini-sft/ckpt/step_00011500}
STEPS=${TOOL_STEPS:-350}

[ -s "$DATA_ROOT/sft18_tools/index.json" ] || { echo "[mini-tools] missing $DATA_ROOT/sft18_tools/index.json; run tools_sft_build.sbatch first"; exit 1; }
[ -s "$DATA_ROOT/sft17/index.json" ]       || { echo "[mini-tools] missing the sft17 replay source"; exit 1; }

# On a requeue, continue from this run's own latest checkpoint rather than restarting from the
# released one. "auto" looks inside --out; the explicit base is only correct on the first launch.
if compgen -G "$OUT/ckpt/step_*" > /dev/null; then RESUME=auto; else RESUME=$BASE; fi
[ "$RESUME" = auto ] || [ -d "$RESUME" ] || { echo "[mini-tools] base checkpoint missing: $RESUME"; exit 1; }

export OMP_NUM_THREADS=8
export MORENA_WALLTIME=08:00:00

echo "[mini-tools] resume=$RESUME out=$OUT anneal=$STEPS"
srun --kill-on-bad-exit=1 torchrun --standalone --nnodes 1 --nproc_per_node 1 $H/train/train.py \
  --config $H/train/configs/morena0p5b_mini.json \
  --mix    $H/train/configs/mix_sft18_tools.json \
  --data-root $DATA_ROOT \
  --out    $OUT \
  --resume "$RESUME" \
  --anneal "$STEPS" \
  --loss-mask \
  --ckpt-minutes 20 \
  --override '{"train":{"lr":7e-5,"global_batch_seqs":32,"micro_batch":4,"fsdp_shard_size":0}}'
echo "[mini-tools] exit $? at $(date)"
