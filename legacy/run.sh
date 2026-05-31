#!/usr/bin/env bash
# Legacy reproduction wrapper. New runs should use ../run.sh, which defaults to
# the v3 hard typo-noised ConlangCrafter corpus.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY="$(python3 -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])" 2>/dev/null || true)"
  export WANDB_API_KEY
fi

cmd=(uv run modal run main.py)

case "${1:-}" in
  clean-conlang-track1)
    shift
    exec "${cmd[@]}" --track 1 \
      --dataset-id TearedModels/conlangcrafter-cpt-bd412d52 \
      --dataset-revision 5cfd047a92023011326e8383d45d97db22add909 \
      "$@"
    ;;
  clean-conlang-track2)
    shift
    exec "${cmd[@]}" --track 2 \
      --dataset-id TearedModels/conlangcrafter-cpt-bd412d52 \
      --dataset-revision 5cfd047a92023011326e8383d45d97db22add909 \
      "$@"
    ;;
  clean-conlang-track3)
    shift
    exec "${cmd[@]}" --track 3 \
      --dataset-id TearedModels/conlangcrafter-cpt-bd412d52 \
      --dataset-revision 5cfd047a92023011326e8383d45d97db22add909 \
      "$@"
    ;;
  finemath-track1)
    shift
    exec "${cmd[@]}" --track 1 \
      --dataset-id HuggingFaceTB/finemath \
      --dataset-config finemath-4plus \
      --dataset-revision e92b25a616738fe95dc186b64dfb19f9c8525594 \
      "$@"
    ;;
  ultrachat-sft-track1)
    shift
    exec "${cmd[@]}" --track 1 --data-mode sft "$@"
    ;;
  "")
    sed -n '1,80p' legacy/README.md
    exit 2
    ;;
  *)
    echo "unknown legacy target: $1" >&2
    sed -n '1,80p' legacy/README.md >&2
    exit 2
    ;;
esac
