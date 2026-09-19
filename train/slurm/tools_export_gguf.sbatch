#!/bin/bash
#SBATCH -J tools-gguf
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH -A aih4a_vamboai
#SBATCH --gres=gpu:1
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH -t 01:30:00
#SBATCH -o /leonardo/home/userexternal/imisi000/morena/logs/tools-gguf_%j.out
#SBATCH -e /leonardo/home/userexternal/imisi000/morena/logs/tools-gguf_%j.out
#
# Tool fine-tunes -> GGUF Q4_K_M, for the laptop demo.
#
# This follows edge/gguf_nano.sbatch step for step rather than inventing a shorter route. The first
# attempt went straight from export_hf_mini.py to convert_hf_to_gguf.py and died on "Failed to
# detect model architecture": that exporter writes MORENA's own config, and llama.cpp only knows
# architectures it has a loader for. export_llama_hf.py is the missing step. MORENA is Llama-3
# shaped (GQA, SwiGLU, RoPE, RMSNorm, tied embeddings) so the reshape is a rename, and PYTHONPATH
# has to point at the in-tree gguf-py or the converter picks up the wrong gguf package.
set -uo pipefail
H=/leonardo/home/userexternal/imisi000/morena
B=/leonardo_scratch/large/userexternal/imisi000/morena
source $B/venv/bin/activate
export HF_HUB_OFFLINE=1 MORENA_LOCAL=1 HF_HOME=$B/hf_cache TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$B/llama.cpp/gguf-py
mkdir -p $B/edge

one () {
  local TAG=$1 RUN=$2 CFG=$3
  echo "===== $TAG ====="
  local CK=$(ls -d $B/runs/$RUN/ckpt/step_* 2>/dev/null | grep -v "\.tmp$" | sort -V | tail -1)
  [ -n "$CK" ] && [ -d "$CK" ] || { echo "FATAL: no checkpoint for $RUN" >&2; return 1; }
  echo "[gguf] $TAG checkpoint = $CK"
  cd $H/eval || return 1
  env -u MORENA_CKPT python export_hf.py --ckpt "$CK" --config "$CFG" --out $B/hf_export/$TAG || return 1
  python export_llama_hf.py --src $B/hf_export/$TAG --out $B/hf_export/$TAG-llama || return 1
  cd $B/llama.cpp || return 1
  python convert_hf_to_gguf.py $B/hf_export/$TAG-llama \
      --outfile $B/edge/$TAG-f16.gguf --outtype f16 || return 1
  ./build/bin/llama-quantize $B/edge/$TAG-f16.gguf $B/edge/$TAG-Q4_K_M.gguf Q4_K_M 16 || return 1
  rm -f $B/edge/$TAG-f16.gguf
  ls -la $B/edge/$TAG-Q4_K_M.gguf
  # Greedy check in the shape these models are actually used in: a tool menu and a request. A
  # conversion that quantised to noise still prints a file listing, so the file is not the proof.
  echo "== greedy $TAG"
  ./build/bin/llama-completion -m $B/edge/$TAG-Q4_K_M.gguf \
      -p '<reserved_0>
<reserved_2>[{"name":"transfer_initiate","description":"Send money to a saved recipient","arguments":{"amount":"number","recipient":"string","reason":"string","currency":"string"}},{"name":"balance_fetch","description":"Check the available balance","arguments":{}}]
Abeg send 5000 naira give Chidi.
<reserved_1>' -e -n 60 --temp 0 -no-cnv -t 16 2>/dev/null | tail -4
}

one nano-tools morena0p2b-nano-tools  $H/train/configs/morena0p2b_nano.json
one mini-tools morena0p5b-mini-tools  $H/train/configs/morena0p5b_mini.json
cd $B/edge && sha256sum nano-tools-Q4_K_M.gguf mini-tools-Q4_K_M.gguf > SHA256SUMS.tools 2>/dev/null
echo "TOOLS GGUF DONE"
