#!/bin/sh
# The E011 runner: sh experiments/E011/run.sh LLAMA_BUILD_BIN QREF
# LLAMA_BUILD_BIN holds llama-gguf-hash; QREF is experiments/E011/qref.c compiled against
# that build's libggml-base. Writes every output into experiments/E011/ and a log to run.log.
set -u
BIN=$1
QREF=$2
OUT=experiments/E011
HUB=$HOME/.cache/huggingface/hub
QWEN=$HUB/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
QWEN_GGUF=$HUB/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67
SMOL=$HUB/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac
SMOL_GGUF=$HUB/models--bartowski--SmolLM2-135M-Instruct-GGUF/snapshots/09816acd5d99df7be770d85ea30822623dab342c
LOG=$OUT/run.log
: > "$LOG"
step() {  # step NAME OUTFILE COMMAND...: run, keep stdout, log exit status and seconds
    name=$1; out=$2; shift 2
    start=$(python3 -c 'import time; print(time.time())')
    "$@" > "$out" 2> "$out.stderr"
    code=$?
    end=$(python3 -c 'import time; print(time.time())')
    echo "$name: exit $code, $(python3 -c "print(f'{$end - $start:.1f}')") s" >> "$LOG"
    [ -s "$out.stderr" ] || rm -f "$out.stderr"
}
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ); revision $(git rev-parse --short HEAD) plus working tree; free $(df -h "$HOME" | awk 'NR==2 {print $4}')" >> "$LOG"
step reader "$OUT/reader.txt" uv run python experiments/E011/compare_reader.py \
    "$QWEN_GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf" "$SMOL_GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf" \
    "$SMOL_GGUF/SmolLM2-135M-Instruct-f16.gguf"
for pair in "$QWEN_GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf" "$SMOL_GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf" \
            "$SMOL_GGUF SmolLM2-135M-Instruct-f16.gguf"; do
    set -- $pair
    step "gguf-hash wvc $2" "$OUT/$2.wvc-gguf-hash" uv run wvc hash --format gguf-hash "$1/$2"
    (cd "$1" && "$BIN/llama-gguf-hash" --sha256 "$2" 2>/dev/null) > "$OUT/$2.llama-gguf-hash"
    if cmp -s "$OUT/$2.wvc-gguf-hash" "$OUT/$2.llama-gguf-hash"; then same=identical; else same=DIFFERENT; fi
    echo "gguf-hash $2: $(wc -l < "$OUT/$2.wvc-gguf-hash" | tr -d ' ') lines, wvc and llama-gguf-hash $same" >> "$LOG"
done
(cd spec/vectors/v0 && "$BIN/llama-gguf-hash" --sha256 quant.gguf 2>/dev/null) > "$OUT/quant.gguf.llama-gguf-hash"
if cmp -s "$OUT/quant.gguf.llama-gguf-hash" spec/vectors/v0/quant.gguf-hash; then same=identical; else same=DIFFERENT; fi
echo "gguf-hash spec vector quant.gguf: llama-gguf-hash and the committed quant.gguf-hash $same" >> "$LOG"
step dequant-c "$OUT/dequant-c.txt" uv run python experiments/E011/verify_dequant_c.py "$QREF"
step "diff qwen q4_k_m" "$OUT/qwen-q4_k_m.diff.txt" uv run wvc diff --rank "$QWEN/model.safetensors" "$QWEN_GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
step "diff qwen q4_k_m json" "$OUT/qwen-q4_k_m.diff.json" uv run wvc diff --rank --json --all "$QWEN/model.safetensors" "$QWEN_GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
step "diff smollm2 q4_k_m" "$OUT/smollm2-q4_k_m.diff.txt" uv run wvc diff --rank "$SMOL/model.safetensors" "$SMOL_GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
step "diff smollm2 q4_k_m json" "$OUT/smollm2-q4_k_m.diff.json" uv run wvc diff --rank --json --all "$SMOL/model.safetensors" "$SMOL_GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf"
step "diff smollm2 f16" "$OUT/smollm2-f16.diff.txt" uv run wvc diff "$SMOL/model.safetensors" "$SMOL_GGUF/SmolLM2-135M-Instruct-f16.gguf"
step "diff smollm2 f16 json" "$OUT/smollm2-f16.diff.json" uv run wvc diff --json --all "$SMOL/model.safetensors" "$SMOL_GGUF/SmolLM2-135M-Instruct-f16.gguf"
echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ); free $(df -h "$HOME" | awk 'NR==2 {print $4}')" >> "$LOG"
