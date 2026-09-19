#!/bin/bash
#SBATCH -J tools-smoke
#SBATCH -A aih4a_vamboai
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 00:20:00
#SBATCH --output=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/imisi000/morena/logs/%x-%j.err
#
# Three steps, thrown away. Proves the things the real run would otherwise discover six hours in:
# that --no-fsdp works on one GPU, that micro_batch 8 at seq_len 4096 fits an A100 64GB, that the
# released nano checkpoint loads and its fingerprint matches, and that --anneal moves total_steps
# past the resumed step instead of exiting immediately. Uses sft17, which is already on disk, so it
# can run before the tool corpus lands.
#
# FSDP is kept at world=1 rather than --no-fsdp. --no-fsdp leaves the parameters in fp32 and
# flash-attn then refuses: "FlashAttention only support fp16 and bf16 data type". The bf16 cast
# comes from FSDP's MixedPrecisionPolicy, not from the autocast around the forward. The
# alternatives are worse: --attn sdpa drops cross-document masking on packed sequences, and
# sdpa_mask materialises a (B,1,T,T) mask at roughly twice the cost. At world=1 fully_shard has
# no one to communicate with, so it costs nothing and the numerics match how these models were
# actually trained.
set -euo pipefail
H=/leonardo/home/userexternal/imisi000/morena
B=/leonardo_scratch/large/userexternal/imisi000/morena
mkdir -p $H/logs
source $B/venv/bin/activate
OUT=$B/runs/smoke-tools-geometry
rm -rf $OUT
export OMP_NUM_THREADS=8
srun --kill-on-bad-exit=1 torchrun --standalone --nnodes 1 --nproc_per_node 1 $H/train/train.py \
  --config $H/train/configs/morena0p2b_nano.json \
  --mix    $H/train/configs/mix_sft17.json \
  --data-root $B/data \
  --out    $OUT \
  --resume $B/runs/morena0p2b-nano/ckpt/step_00010000 \
  --anneal 800 \
  --loss-mask \
  --max-steps 3 \
  --ckpt-minutes 999 \
  --override '{"train":{"lr":1e-4,"global_batch_seqs":32,"micro_batch":8,"fsdp_shard_size":0}}'
echo "[smoke] exit $? at $(date)"
