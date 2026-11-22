#!/usr/bin/env bash
# Run from the repository root. RESUME=1 continues only the same fingerprinted build.
set -euo pipefail
PYTHON=${PYTHON:-.venv/bin/python}
MODEL=${MODEL:-.runtime/models/Qwen3-TTS-12Hz-1.7B-Base}
OUTPUT=${OUTPUT:-data/qwen_holdlistenstream_v1}
DEVICES=${DEVICES:-6,7}
export CUDA_VISIBLE_DEVICES="$DEVICES" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
IFS=',' read -ra gpu_ids <<< "$DEVICES"
for id in "${gpu_ids[@]}"; do
    used=$(nvidia-smi -i "$id" --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( used > 1024 )); then
        echo "GPU $id already uses ${used} MiB; select idle devices with DEVICES." >&2
        exit 1
    fi
done
mkdir -p "$OUTPUT/logs"
rebuild=all
if [[ ${RESUME:-0} == 1 ]]; then rebuild=resume; fi
args=(--model "$MODEL" --output "$OUTPUT" --codecs semantic_gan vae_vq)
if [[ -n ${SOURCE_MANIFEST:-} ]]; then args+=(--source-manifest "$SOURCE_MANIFEST"); fi
"$PYTHON" -m qwen_tts.holdlistenstream.prepare "${args[@]}" --stage index --rebuild "$rebuild" 2>&1 | tee "$OUTPUT/logs/index.log"
"$PYTHON" -m qwen_tts.holdlistenstream.prepare "${args[@]}" --stage references --device cuda:0 2>&1 | tee "$OUTPUT/logs/references.log"
trap 'jobs -pr | xargs -r kill' EXIT
for ((i=0; i<${#gpu_ids[@]}; i++)); do
    "$PYTHON" -m qwen_tts.holdlistenstream.prepare "${args[@]}" --stage encode --workers "${#gpu_ids[@]}" --worker "$i" --device "cuda:$i" > "$OUTPUT/logs/encode-$i.log" 2>&1 &
done
for ((i=0; i<${#gpu_ids[@]}; i++)); do
    wait -n
done
"$PYTHON" -m qwen_tts.holdlistenstream.prepare "${args[@]}" --validate-only 2>&1 | tee "$OUTPUT/logs/validate.log"
