#!/bin/sh
# E012: GGUF uploads through hf: arguments. Usage, from the repository root:
#   sh experiments/E012/run.sh CACHE_DIR
# CACHE_DIR is an empty directory used as HF_HUB_CACHE for this run only, so that the bytes
# downloaded can be measured and the cache deleted afterwards. Every step's stdout goes to
# experiments/E012/NAME, its stderr to NAME.err, and its exit code and seconds to run.log.
set -u
CACHE="$1"
OUT=experiments/E012
export HF_HUB_CACHE="$CACHE"

QWEN3=hf:Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
QWEN3_Q=hf:bartowski/Qwen_Qwen3-0.6B-GGUF@60b85c0e3d8fe0f6474f406922a26d12aca4550d
QWEN25=hf:Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775
QWEN25_Q=hf:Felladrin/gguf-Q5_K_M-Qwen2.5-0.5B-Instruct@69b6ffa31d2619f3a78b67ac661dec2b98d4e2fb
LLAMA=hf:meta-llama/Llama-3.2-1B-Instruct@9213176726f574b556790deb65791e0c5aa438b6
LLAMA_Q=hf:lmstudio-community/Llama-3.2-1B-Instruct-GGUF@199151125cf15a129ab3b548b26afeed976df066

: > "$OUT/run.log"
step() {
    name=$1
    shift
    start=$(date +%s)
    uv run --extra hf wvc "$@" > "$OUT/$name" 2> "$OUT/$name.err"
    code=$?
    echo "$name: exit $code, $(( $(date +%s) - start )) s: wvc $*" >> "$OUT/run.log"
}

step bare-listing.txt hash "$QWEN3_Q"
step qwen3-q4_k_m.diff.json diff --rank --json --all "$QWEN3" "$QWEN3_Q:Qwen_Qwen3-0.6B-Q4_K_M.gguf"
step qwen3-q4_k_m.agent.md diff --rank --format agent "$QWEN3" "$QWEN3_Q:Qwen_Qwen3-0.6B-Q4_K_M.gguf"
step qwen25-q5_k_m-split.diff.json diff --rank --json --all "$QWEN25" "$QWEN25_Q:qwen2-00002-of-00003.gguf"
step qwen25-q5_k_m-split.agent.md diff --rank --format agent "$QWEN25" "$QWEN25_Q:qwen2-00002-of-00003.gguf"
step split-vs-single.diff.json diff --json --all "$QWEN25_Q:qwen2.5-0.5b-instruct-q5_k_m-imat.gguf" "$QWEN25_Q:qwen2-00001-of-00003.gguf"
step split.gguf-hash hash --format gguf-hash "$QWEN25_Q:qwen2-00003-of-00003.gguf"
step llama-q4_k_m.diff.json diff --rank --json --all "$LLAMA" "$LLAMA_Q:Llama-3.2-1B-Instruct-Q4_K_M.gguf"
step llama-q4_k_m.agent.md diff --rank --format agent "$LLAMA" "$LLAMA_Q:Llama-3.2-1B-Instruct-Q4_K_M.gguf"
du -sk "$CACHE" | awk '{print "cache: " $1 " KiB"}' >> "$OUT/run.log"
