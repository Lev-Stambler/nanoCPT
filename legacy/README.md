# Legacy Reproduction Entrypoints

The current default dataset is the v3 hard typo-noised conlang corpus:
`TearedModels/conlangcrafter-cpt-bd412d52-hard-typo2`.

This directory keeps explicit opt-in wrappers for retired datasets and modes
used by older records. Do not use these for new leaderboard submissions unless
you are intentionally reproducing old numbers.

```bash
# Retired clean conlang corpus used by v1/v2 records:
./legacy/run.sh clean-conlang-track2

# Pre-conlang CPT anchor:
./legacy/run.sh finemath-track1

# Legacy SFT path:
./legacy/run.sh ultrachat-sft-track1
```

Retired record artifacts stay under `_legacy_records/` so the active `records/`
tree only contains current-version submissions.
