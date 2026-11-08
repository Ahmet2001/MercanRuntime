#!/usr/bin/env bash
# Run a Mercan / NedoLM model (Turkish chat) with the Mercan CLI.
#
#   scripts/run_mercan.sh                    # interactive chat
#   scripts/run_mercan.sh -p "Merhaba!"      # single prompt
#   scripts/run_mercan.sh --temperature 0    # greedy (any CLI option can be appended)
#
# Environment:
#   MERCAN_MODEL  local .mercan path or Hugging Face owner/repo[:file.mercan]
#                 (default: MercanAI/Mercan-0.8B-DPO-Q4, downloaded once into ~/.cache/mercan)
#   MERCAN_BIN    mercan CLI to use (default: build one with scripts/mercan_cli.sh)
#   MERCAN_JOBS   parallel build jobs (default: nproc)
#
# Sampling uses the CLI defaults, which match the iOS app (temperature 0.7,
# top_k 40, top_p 0.95, min_p 0.05, repetition penalty 1.10).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MERCAN_MODEL:-MercanAI/Mercan-0.8B-DPO-Q4:Mercan-0.8B-DPO-Q4_K_M.mercan}"
BIN="$("$ROOT/scripts/mercan_cli.sh")"

if ! "$BIN" arch list | grep -q '^nedolm'; then
  echo "$BIN has no nedolm architecture provider" >&2
  exit 1
fi

exec "$BIN" run "$MODEL" \
  -n 512 \
  -t "$(( $(nproc) > 8 ? 8 : $(nproc) ))" \
  "$@"
