# Agent guidance for modded-continued-training

Project-level conventions for AI agents (Claude Code, Codex, etc.) working
on this repo. Read this before touching the trainer or running ablations.

## Ablation runs must be on the same GPU SKU

Modal's `gpu="H100"` resolves to whichever H100 SKU is available at
schedule time — typically **H100 SXM5 80GB HBM3** but sometimes
**H100 NVL 94GB**. Even at 100 % `gpu_utilization_percent`, the NVL is
materially slower per step for our workload (compile-warmup
~234 s vs ~218 s, step throughput ~7 k vs ~9 k tokens/s for the same
configuration). That difference shows up as fewer completed steps within
the timed budget — a real handicap that has nothing to do with the
optimizer/adapter/schedule being ablated.

**Rule**: when comparing optimizers, schedules, or other training-loop
knobs at a fixed wall-clock budget, every run in the comparison must land
on the same GPU SKU. Before declaring a winner:

1. Check `gpu_name` in each run's `summary.json`. If they don't all
   match (e.g. one is `NVIDIA H100 NVL` and the others are
   `NVIDIA H100 80GB HBM3`), the comparison is not fair.
2. Re-run the off-SKU candidate(s) until you get a matching SKU.
   Modal capacity varies by region/time; retrying usually lands you on
   the more common SKU.
3. Also confirm `peak_gpu_utilization_percent ≥ ~95 %` for every run —
   a run that was util-starved (data-loader bottleneck, OOM-induced
   fallback path, etc.) is also unfair to keep in the comparison.

If you can't get matching SKUs after a reasonable number of retries,
note the SKU mismatch explicitly in the leaderboard description and the
findings doc — don't quietly publish a number that overstates the
loser.

## Don't ship records whose source path no longer exists

When `main.py` loses a code path (e.g. the LoRA strip in `edcfd4e`), any
record artifact that exercised that path becomes irreproducible. Delete
the record dirs and the corresponding `citations/records.bib` entries
in the same PR that removes the code. Stale records on the leaderboard
mislead first-time visitors about what the current trainer actually does.

## Per-group LR convention for the Muon family

The `muon`, `muon8`, and `normuon` optimizer choices each split params
into "2D hidden weights → Muon-family" and "embed/lm\_head/1D → AdamW8bit
tail" (modded-nanogpt convention). Both groups default to `--lr` if you
don't pass `--muon-lr` / `--adamw-tail-lr`, but Muon's Newton-Schulz
update is RMS-normalized to ~1 — so at a shared LR, the Muon group's
effective per-step change is much smaller than the AdamW tail's.

Empirical guidance (Track 2, full FT, Qwen3.5-4B-Base, 2026-05-28):

- AdamW-tail-only LR `2e-5` is fine.
- Muon group LR should be ~10× that (`2e-4`) to get a fair comparison
  against pure AdamW.

A naive single-LR Muon run at `2e-5` scored +0.389 vs +0.486 with the
per-group LR. Always use `--muon-lr` when ablating Muon-family
optimizers; otherwise the optimizer choice is conflated with an
unintentional LR ablation.

## Modal image rebuilds on `main.py` change

Modal's image hash includes `main.py` (the entrypoint), so any edit
invalidates the cached image and triggers a full rebuild. Cold image
build is ~5-10 min on top of the run itself. Batch unrelated `main.py`
edits before launching long ablations.

## Maximize VRAM utilization

The H100 has 80 GiB (SXM5) or 94 GiB (NVL). Modal schedules either SKU for
`gpu="H100"`, so to keep runs comparable the allocator is capped to a fixed
byte budget — `--vram-fraction` (default 0.92) of the 80 GiB reference, i.e.
**~72.8 GiB on every card** (see `compute_vram_fraction` in `main.py`). The
applied fraction and budget are recorded in `summary.json` as
`vram_fraction_applied` / `vram_budget_gib`.

A correct Track 2 config should push `peak_cuda_memory_allocated_gib` close
to that **budget** (~72.8 GiB), not the raw card. Idle VRAM is wasted
activations, which is wasted tokens-per-step, which is wasted loss drop in
the fixed budget.

If a run lands at e.g. 54 GiB, that's ~19 GiB of unused activation room —
bump `--seq-len`, `--micro-batch-size`, or enable features that trade memory
for tokens (`--lowpass --gradient-checkpointing false`, `--optimizer-name
adamw8bit` to free optimizer state, etc.) until the next run trips OOM, then
back off one notch. (To deliberately use a full 94 GiB NVL card, raise
`--vram-fraction`; the default keeps SKUs interchangeable.)

Goal: every accepted record's `summary.json` should have
`peak_cuda_memory_allocated_gib` close to `vram_budget_gib` (≈ the cap).
Note `peak_cuda_memory_allocated_fraction` is measured against the raw card,
so on a 94 GiB NVL it reads ~0.77 even at a full budget — judge against the
budget, not the raw fraction.

## Document-aware packed sequences (don't leak across boundaries)

This trainer packs multiple documents into fixed-length blocks via
`stream_concat_no_padding`. Two correctness rules to maintain:

1. **Attention must not cross document boundaries.** Pass `position_ids`
   that reset to 0 at every new document start; set `attention_mask=None`.
   HF transformers' `flash_attention_2` / `flex_attention` paths detect
   document starts from `position_ids == 0` and build per-document
   block masks internally. Never pass `attention_mask=torch.ones(...)`
   to a packed batch — that's what produced the v1 leaky-attention
   leaderboard (~0.2 lower baseline than the v2 doc-aware run).

2. **Low-pass activation compression must not cross document
   boundaries** either. When `--lowpass` is on, each document's tokens
   are padded up to a multiple of `--lowpass-chunk-size` with pad tokens
   (labels `-100`, ignored in loss). This guarantees every Hadamard
   chunk is purely within one document.

When changing the eval/packing/attention semantics in a way that alters
absolute eval-loss numbers, bump `EVAL_VERSION` in `main.py` and start a
new `records/<track>/v<N+1>/` subdirectory — only same-version records
are directly comparable.

## Iterate with `--no-compile-model` when touching new code paths

When adding a new feature (optimizer, schedule, lowpass variant, etc.)
run your first few iterations with `--no-compile-model --no-compile-warmup`
(or use `run.sh smoke`).  `torch.compile` adds minutes of autotune time
and obscures whether a bug is in your logic or in the compiled graph.
Only turn compilation back on once the plain-Python path runs correctly.

## Run wiring sanity-check after the timed boundary

`run.sh smoke` is the canonical end-to-end check that the
compile/warmup → timed train loop → final eval pipeline still works
after non-trivial `main.py` edits. It uses `--no-compile-model
--no-compile-warmup` so it skips the expensive autotune but still
exercises the timing-boundary code (`budget_start` at line ~1490,
deadline check at ~1523, final eval at ~1590). Run it after any change
that touches the trainer's control flow.
