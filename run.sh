#!/usr/bin/env bash
# Thin wrapper around `uv run modal run main.py`. All entries default to
# the canonical ConlangCrafter CPT/LoRA challenge — see README for details.
set -euo pipefail

# W&B is on by default (online). main.py reads WANDB_API_KEY from the local
# environment to build the Modal secret; source it from ~/.netrc when unset so
# the key never has to live in git or config.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])" 2>/dev/null || true)"
  export WANDB_API_KEY
fi

cmd=(uv run modal run main.py)

case "${1:-}" in
  "")
    exec "${cmd[@]}"
    ;;
  smoke)
    shift
    exec "${cmd[@]}" \
      --minutes 0.1 \
      --eval-blocks 2 \
      --micro-batch-size 4 \
      --grad-accum 1 \
      --no-compile-model \
      --no-compile-warmup \
      --attn-implementation sdpa \
      --wandb-mode disabled \
      "$@"
    ;;
  track1)
    shift
    exec "${cmd[@]}" --track 1 "$@"
    ;;
  track2)
    shift
    exec "${cmd[@]}" --track 2 "$@"
    ;;
  track3)
    shift
    exec "${cmd[@]}" --track 3 "$@"
    ;;
  input-bench)
    shift
    exec "${cmd[@]}" --track 2 \
      --seq-len 2048 \
      --micro-batch-size 5 \
      --grad-accum 1 \
      --eval-micro-batch-size 1 \
      --input-benchmark-batches 256 \
      --input-benchmark-warmup-batches 16 \
      --wandb-mode disabled \
      "$@"
    ;;
  # Escape hatches for reproducing pre-conlang records.
  legacy-cpt-track1)
    shift
    exec "${cmd[@]}" --track 1 \
      --dataset-id HuggingFaceTB/finemath \
      --dataset-config finemath-4plus \
      --dataset-revision e92b25a616738fe95dc186b64dfb19f9c8525594 \
      "$@"
    ;;
  legacy-sft-track1)
    shift
    exec "${cmd[@]}" --track 1 --data-mode sft "$@"
    ;;
  *)
    exec "${cmd[@]}" "$@"
    ;;
esac
