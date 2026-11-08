#!/usr/bin/env bash
# Run LiquidAI LFM3-1.7B (.mercan) with the Mercan CLI.
#
#   scripts/run_lfm3.sh                      # interactive chat (reasoning hidden)
#   scripts/run_lfm3.sh --no-thinking        # faster, no reasoning (less reliable)
#   scripts/run_lfm3.sh --show-thinking      # stream the reasoning before the answer
#   scripts/run_lfm3.sh -p "Merhaba!"        # single prompt
#
# Environment:
#   LFM3_MODEL    local .mercan path or Hugging Face owner/repo
#                 (default: MercanAI/LFM3-1.7B-mercan, downloaded once into ~/.cache/mercan)
#   MERCAN_BIN    mercan CLI to use (default: build one with scripts/mercan_cli.sh)
#   MERCAN_JOBS   parallel build jobs (default: nproc)
#
# Sampling defaults follow the LFM3 model card ("General": temperature 0.1,
# top_k 50, repetition_penalty 1.1). Later arguments override them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${LFM3_MODEL:-MercanAI/LFM3-1.7B-mercan}"
BIN="$("$ROOT/scripts/mercan_cli.sh")"

if ! "$BIN" arch list | grep -q '^lfm3'; then
  echo "$BIN has no lfm3 architecture provider; rebuild from the ios-lfm branch (delete build/cli)" >&2
  exit 1
fi

exec "$BIN" run "$MODEL" \
  -n 4096 \
  --temperature 0.1 --top-k 50 --repeat-penalty 1.1 \
  -t "$(( $(nproc) > 8 ? 8 : $(nproc) ))" \
  "$@"
