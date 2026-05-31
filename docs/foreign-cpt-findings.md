# ConlangCrafter CPT findings (2026-05-27/28)

## Question

The Track 1 SFT default (`HuggingFaceH4/ultrachat_200k` + `Qwen3.5-4B-Base`)
produces a `baseline_eval_loss − final_eval_loss` of roughly **+0.05** over a
30-minute LoRA fine-tune. That is barely above noise. We wanted a dataset that
gives a **pronounced** loss-drop signal — large enough that experiments can
distinguish good and bad training choices clearly, and large enough that the
score number tells a meaningful story.

## Hypothesis

The drop is small because ultrachat is general English chat: very close to
Qwen's pretraining distribution, so its baseline loss is already low (~0.13)
and there is little headroom for training to improve it. Picking a dataset
that is **demonstrably outside** the pretraining distribution should give a
high baseline loss and a much larger achievable drop.

## Choice: ConlangCrafter synthetic conlang

A constructed language synthesized one-shot via Vertex AI Gemini 3.5 Flash
against a [ConlangCrafter](https://arxiv.org/abs/2508.06094) language spec
(`bd412d52`, DeepSeek-R1-generated, 131 lexicon words, polysynthetic, IPA
with tones and clicks). The original clean corpus was published as
[`TearedModels/conlangcrafter-cpt-bd412d52`](https://huggingface.co/datasets/TearedModels/conlangcrafter-cpt-bd412d52).
The current v3 default is the regenerated 2% typo-noised variant
[`TearedModels/conlangcrafter-cpt-bd412d52-hard-typo2`](https://huggingface.co/datasets/TearedModels/conlangcrafter-cpt-bd412d52-hard-typo2)
at revision `c8aae01b831778315a2522cf20f5111f4ab7b903`.

### Why this and not alternatives

| Considered | Status | Reason |
|---|---|---|
| **ConlangCrafter synthetic conlang** | **Selected** | Guaranteed novel (the language did not exist before synthesis), infinitely extensible (re-run the script for more tokens), deterministic by spec id, no Qwen pretraining exposure. |
| SumTablets ([colesimmons/SumTablets](https://huggingface.co/datasets/colesimmons/SumTablets), arXiv [2602.22200](https://arxiv.org/abs/2602.22200)) | Considered, dropped | Sumerian cuneiform transliterations, 82,452 rows / 30M chars. Gave a bigger Track 2 absolute drop (+1.09 vs the conlang's +0.51), but it's fixed-size: at Track 3 budgets (~80M tokens consumed) the model would see each tablet 5–8× and start memorizing rather than learning structure, polluting the signal. The conlang scales by re-running synthesis. |
| Linear A and other undeciphered scripts (Indus, Rongorongo, Voynich, Proto-Elamite, Phaistos Disc) | Rejected | Corpora are tiny (Linear A: ~7,400 signs across 1,427 inscriptions; others similar). Three-plus orders of magnitude too small, and without decipherment there is no grammar regularity to learn — any loss drop would be memorization. |
| Tibetan (TIBSTC, `pkupie/mc2_corpus`) | Rejected | 11B tokens, but well-represented in modern pretraining — not OOD enough. |

## Synthesis pipeline (`scripts/synthesize_conlang_cpt.py`)

- One ConlangCrafter spec (full phonology + grammar + lexicon) as the system
  prompt, ~9.4K input tokens per call.
- Rotated topic seeds across 50 prompts for content diversity (village at
  dawn, hunter and prey, harvest song, etc.).
- Per-chunk quality gate: lexicon-overlap minimum (substring match, ≥50%),
  English-word ratio maximum (≤5%), minimum length (400 chars). Failed
  chunks retried up to 2× with a different topic.
- `thinking_config=ThinkingConfig(thinking_budget=0)` — without this, Gemini
  3.5 Flash silently burns the entire output budget on thoughts and returns
  empty text. **Critical debugging finding.**
- `max_output_tokens=8192` with headroom — when the model hits MAX_TOKENS,
  the truncated final part can have `text=None` and `resp.text` returns `""`
  even after thousands of generated tokens. Always leave slack.
- Async concurrency 32 against Vertex on `gemini-3.5-flash` (global endpoint).
- Resumable JSONL output during the run; converts to parquet at the end.

### Generation stats (one-shot run, 2026-05-27)

- 3,077 chunks accepted, 0 final rejects (all gated chunks succeeded on retry).
- 10.99M output tokens, 13.26M chars.
- 1607s (27 min) wallclock at concurrency 32.
- Cost: ~$5–20 on Flash pricing.

### Hard-regeneration stats (v3 default, 2026-05-31)

- Same `bd412d52` spec and Vertex `gemini-3.5-flash` prose generator.
- `--variant hard_typo2_20260531 --typo-rate 0.02 --target-tokens 10_000_000 --concurrency 32`.
- 3,079 chunks accepted, 0 final rejects; 101 short and 58 English-ratio
  first-attempt rejects were retried.
- 13.25M chars after typo injection; 1642s wallclock.
- Published as `TearedModels/conlangcrafter-cpt-bd412d52-hard-typo2`.

## Results

Current v3 hard-typo2 Track 2 baseline: AdamW fused full fine-tune, lr
2e-5, mb 4 × ga 2, seq4096, seed 1337 scored **+0.7089** (baseline
1.2425 → final 0.5336) in 114 steps on H100 SXM5.

### Track 2 — 5 minutes

Identical hyperparameters across candidates (LoRA rank 32, AdamW fused,
lr 2e-4, micro_batch_size 8, flex-attention, `max-autotune-no-cudagraphs`,
64-block held-out eval).

| Dataset | Eval-loss drop | Baseline | Final | Steps | Tokens |
|---|---:|---:|---:|---:|---:|
| FineMath (legacy CPT, anchor) | **−0.034** ❌ | 1.431 | 1.466 | 101 | 3.31M |
| **ConlangCrafter** (selected) | **+0.510** ✅ | 0.854 | 0.345 | 101 | 3.31M |
| SumTablets (reference) | +1.092 | 1.946 | 0.855 | 99 | 3.24M |

Both foreign datasets dwarf the FineMath signal and the prior Hermes-SFT
Track 1 record of +0.052. SumTablets gives a larger absolute drop in 5 min
because Qwen's tokenizer treats Latin-with-subscripts transliteration as
many unfamiliar tokens, inflating both baseline and headroom. ConlangCrafter
gives the lowest final loss and is the chosen canonical because it scales.

### Track 1 — 30 minutes (ConlangCrafter, seed 1337, LoRA-era)

| Metric | Value |
|---|---:|
| eval_loss_drop | **+0.540** |
| baseline_eval_loss | 0.854 |
| final_eval_loss | 0.315 |
| steps | 604 |
| tokens | 19.79M |
| supervised tok/s | 10,989 |
| peak GPU util | 100% |

Snapshot: [Modal run](https://modal.com/apps/tear-labs-43657/main/ap-lv4L5notEjWXBIJhrpNcOe).
The original LoRA-era record artifact under `records/track_1_30min/` was
removed when the trainer dropped all PEFT/adapter code in favour of full
fine-tuning; see the README leaderboard for the current canonical record.

Roughly **10× the previous best Track 1 signal** (Hermes-SFT GraLoRA at
+0.052).

### Track 2 optimizer ablation (full FT, seed 1337, 2026-05-28)

After the LoRA strip we re-ran Track 2 to pick the default optimizer for
the full-fine-tune trainer. The Muon family uses modded-nanogpt's
hybrid convention (Muon/NorMuon/Muon8 on 2D hidden weights, AdamW8bit
tail on embed/lm\_head/1D params), wired via the new `--muon-lr` and
`--adamw-tail-lr` flags. The hybrid groups were run at
`muon_lr=2e-4`, `adamw_tail_lr=2e-5` (a 10× ratio — Muon's normalized
updates need a larger nominal LR).

All four runs landed on H100 SXM5 80GB HBM3 (the NorMuon run was
re-executed after an initial allocation on H100 NVL produced an unfair
~21 % step-count deficit; see `AGENTS.md` "Ablation runs must be on the
same GPU SKU").

| Rank | Optimizer            | Eval-loss drop | Baseline | Final | Steps | Compile/warmup |
|---:|---|---:|---:|---:|---:|---:|
| 1  | **adamw\_fused**     | **+0.4972**    | 0.854    | 0.357 | 77    | 238s           |
| 2  | muon8 hybrid         | +0.4868        | 0.854    | 0.368 | 82    | 219s           |
| 3  | muon hybrid          | +0.4862        | 0.854    | 0.368 | 86    | 218s           |
| 4  | normuon hybrid       | +0.4612        | 0.854    | 0.393 | 81    | 244s           |

Takeaways:
- **AdamW fused wins at the 5-min Track 2 budget**, so it is the new
  `auto` default. The Muon family closes most of the gap once the
  10× LR ratio is applied, but doesn't beat AdamW within 300 timed
  seconds at this model/data scale.
- **Muon8 ≈ Muon**: 8-bit blockwise momentum costs basically nothing in
  loss drop here. Useful if optimizer-state memory becomes a constraint
  at larger Track-3 budgets or different model sizes.
- **NorMuon lands last** even on the same SKU and comparable step count
  (81 vs 77-86 for the others). The gap is the optimizer, not the
  hardware — its normalized momentum may need different beta2/eps for
  this scale, but that's a tuning ablation for a follow-up.
- A naive Muon run at `muon_lr=2e-5` (same as AdamW) scored only
  **+0.389** — the per-group LR convention matters; without it Muon
  underutilizes its update budget.

#### Muon LR sweep (seed 1337, H100 SXM5, same data/seq/eval as above)

Because Muon almost always beats AdamW on LM training and our +0.486 was
suspiciously close to AdamW's +0.497, we did a full LR sweep across both
of our supported `lr_adjustment` modes. Two modes differ in how they
scale the post-Newton-Schulz update:

- `match_rms_adamw` (default): `lr * 0.2 * sqrt(max(rows, cols))` — the
  Moonshot Kimi convention. Designed so you can reuse AdamW-sized LRs.
- `original` (Keller Jordan): `lr * sqrt(max(1, rows/cols))` — what
  modded-nanogpt uses. Needs LRs 10-25× larger to get equivalent
  per-step magnitude.

| `lr_adjustment` | `muon_lr` | drop | final | steps | Notes |
|---|---:|---:|---:|---:|---|
| `match_rms_adamw` | 2e-5  | +0.3841 | 0.470 | 77 | Moonshot's Qwen2.5-7B SFT recipe |
| `match_rms_adamw` | **2e-4** | **+0.4862** | **0.368** | **86** | **Peak; leaderboard** |
| `match_rms_adamw` | 5e-4  | +0.4373 | 0.417 | 66 | NVL confound; still below 2e-4 |
| `original`         | 5e-3  | +0.4592 | 0.395 | 84 | Keller-Jordan scale, FT-conservative |
| `original`         | 1e-2  | +0.4084 | 0.446 | 74 | Modded-nanogpt pretraining scale; over-LR here |

Both modes have a clear peak; the peaks are within 0.03 of each other
and **neither beats AdamW fused** at this 5-min Track 2 budget. The
Muon implementation passes audit (Newton-Schulz with the standard
`(3.4445, -4.7750, 2.0315)` coefficients in bfloat16, Nesterov
momentum, fp32 momentum buffer, `torch.matmul` kernels). The
optimizer code is fine; the result is real.

**Probable explanations** for AdamW winning at this budget:

1. Track 2's 5-minute window completes only ~80 optimization steps on a
   single H100. Muon's documented LM advantages come from many more
   steps (modded-nanogpt records compare at thousands to tens of
   thousands of steps); 80 steps may be too few for the Newton-Schulz
   spectral-shaping benefit to compound. Track 3 (2 hours, ~2,000
   steps) is the natural follow-up.
2. Fine-tuning a model already converged on a different distribution
   (Qwen pretrain → synthetic conlang) may be a fundamentally
   different regime from from-scratch LM training. Moonshot's Kimi
   numbers show Muon matching but not necessarily *beating* AdamW for
   SFT in their published recipes.
3. The conlang corpus is small and lexically repetitive; AdamW's
   per-parameter adaptive LR may be more efficient for fitting a
   narrow distribution than Muon's spectral approach.

The 4-way ablation table above keeps `adamw_fused` as the canonical
default. Anyone iterating on optimizers should re-test at Track 1
(30 min) or Track 3 (2 hr) before concluding the Muon family is
strictly worse for this challenge.

## Interpretation

1. **Distribution distance is the dominant lever for loss-drop magnitude.**
   Going from in-distribution (FineMath, baseline 1.43) to fully novel
   (synthetic conlang, baseline 0.85) changes the achievable drop by 1–2
   orders of magnitude. Optimizer, adapter, and schedule choices — which
   dominated prior iteration logs — are much smaller effects than the
   dataset choice.

2. **The synthetic conlang has a counter-intuitively *low* baseline loss
   (~0.85) despite being novel.** Best hypothesis: the IPA/tone-mark
   characters tokenize into many small sub-character pieces that the
   tokenizer's distribution model can predict relatively well from local
   bigram statistics, plus the corpus has repetitive lexical structure (root
   words like `k'u`, `hun`, `wa.la` recur frequently). The model can predict
   "next sub-character within a known root" without knowing the language. The
   *relative* drop (60% of baseline eliminated) is still huge.

3. **5 minutes captures ~95% of the drop on the conlang.** Track 2
   gave +0.510; Track 1 gave +0.540 — diminishing returns after the model
   fits the lexical inventory. Future iterations should either pick a less
   compressible conlang spec (Latin-script alternatives in the
   `malper/ConlangCrafter` set may have higher baselines) or move to a
   larger corpus to delay saturation. The generator now supports these knobs
   directly: `--latin-script`, `--min-lexicon-words N`, a larger
   `--target-tokens`, and `--typo-rate` (character-level noise on the train
   corpus that breaks rote memorization and raises the loss floor — note it
   adds irreducible entropy, so use a modest rate ~1-3%). **`--typo-rate`
   now defaults to 0.02**: the canonical train corpus is generated with 2%
   typos, while the held-out eval corpus is always kept clean (typos are
   force-disabled for `--topic-set heldout`).

## Caveats

- **Single-seed Track 1.** Promoting to a record claim under the README's
  p<0.01 rule needs seeds 2027 and 4099. Commands are in the README's
  Track 1 table.
- **Tokenization sensitivity.** The conlang's low baseline loss is partly a
  tokenization artifact (sub-character IPA pieces). A different base model
  with a different tokenizer would likely show different baseline numbers
  but the same general "drop is much larger than ultrachat" story.
- **Eval set construction.** By default the eval blocks are drawn from the
  unshuffled stream's leading documents (same as the legacy FineMath path),
  so eval is from the same chunk distribution as training — the reported
  `eval_loss_drop` measures distribution-fitting, not generalization, which is
  why it saturates in ~5 min. A separate held-out generalization eval now
  exists: generate a corpus from the same grammar with disjoint topics
  (`scripts/synthesize_conlang_cpt.py --topic-set heldout --variant heldout
  --seed <s>`), push it (`--repo-suffix heldout`), and pass
  `--heldout-eval-dataset-id <repo> --heldout-eval-dataset-revision <rev>`.
  Runs then report `eval_mode: heldout` plus `ngram_reference_loss` and
  (optionally) `random_init_eval_loss` for context. See the README "Held-out
  generalization eval" section. These numbers are NOT comparable to the
  in-distribution leaderboard.
- **One specific language (`bd412d52`).** Different ConlangCrafter languages
  would give different baseline/drop numbers. We picked the longest-spec
  DeepSeek-R1 language; nothing about the choice was optimized for "easiest
  learning."
- **Catastrophic forgetting on the base task is unmeasured.** We did not
  evaluate whether the CPT-on-conlang adapter degrades English performance.

## How to reproduce

```bash
# 1. Smoke the synthesis pipeline (~3 min, ~$0.10).
source ~/.config/.env.global   # provides VERTEXAI_PROJECT etc.
uv run python scripts/synthesize_conlang_cpt.py --smoke

# 2. Full corpus generation (~30 min, ~$5-20 on Flash).
uv run python scripts/synthesize_conlang_cpt.py \
  --language-id bd412d52 --variant hard_typo2_<date> \
  --target-tokens 10_000_000 --concurrency 32 --typo-rate 0.02

# 3. Publish to HF.
uv run python scripts/push_conlang_dataset.py data/conlang_cpt/<language_id>

# 4. Train (main.py defaults to TearedModels/conlangcrafter-cpt-bd412d52-hard-typo2,
#    so the canonical commands "just work"). Override --dataset-id to
#    use your own corpus.
./run.sh track2                  # 5-min sprint
./run.sh track1                  # 30-min Track 1

./run.sh track1 --seed 1337 --record-description "ConlangCrafter CPT seed1337" --record-contributors "@you"
./run.sh track1 --seed 2027 --record-description "ConlangCrafter CPT seed2027" --record-contributors "@you"
./run.sh track1 --seed 4099 --record-description "ConlangCrafter CPT seed4099" --record-contributors "@you"
```

## v2 reset: doc-aware packed attention (2026-05-29)

After the LoRA strip and 4-way optimizer ablation we discovered the
trainer had been passing `attention_mask=torch.ones(...)` to packed
batches, so every token could attend to all preceding tokens in the
pack — including tokens from earlier documents in the same packed
block. This artificially deflated baseline eval loss by ~0.2 because the
model exploited cross-document context that doesn't exist at inference.

The fix (commit `1ad3511`): track per-token doc-relative position
offsets through packing, emit `position_ids` that reset to 0 at every
document start, pass `attention_mask=None`. HF transformers'
`flash_attention_2` / `flex_attention` paths detect document starts
from `position_ids == 0` and build per-document cu_seqlens / block
masks internally. The eval cache key was bumped to invalidate the old
caches; record dirs gained a `v<N>/` version subfolder so v1 leaky
records remain available but not comparable.

### Same config, before / after the fix

| Run | Baseline | Final | Drop | Steps |
|---|---:|---:|---:|---:|
| v1 leaky (AdamW fused, seq 4096, mb 1 × ga 8) | 0.854 | 0.357 | +0.497 | 77 |
| v2 doc-aware (same config) | 1.021 | 0.348 | **+0.673** | 103 |

The v2 baseline both has a higher (correct) starting loss AND drops it
more, because cleaner attention produces cleaner gradients and a faster
per-step throughput (11.2k vs 8.3k tokens/sec — flex_attention with
proper per-doc block masks is more efficient than dense causal masking
over a packed sequence).

## Hadamard low-pass MLP activation compression (2026-05-29)

Ported from the `nanoCPT-instant-lowpass` worktree (the GraLoRA-era
implementation) and adapted for plain `nn.Linear` in
`lowpass.py`. New CLI flag set: `--lowpass --lowpass-chunk-size 64
--lowpass-keep 32 --lowpass-target-filter mlp --lowpass-projector-kind
hadamard`.

How it works: every MLP `nn.Linear` is wrapped in `LowpassLinear`
whose backward saves `x_hat = Hadamard(x)[..., :keep, :]` (50%
activation memory at keep=32/chunk=64) instead of the full `x`. The
input gradient `grad_x = grad_output @ weight` stays exact; the
parameter gradient `grad_w = go_hat.T @ x_hat` is computed from
Hadamard-projected inputs and grad_outputs. Documents are padded to a
multiple of `chunk_size` before packing so every Hadamard chunk is
pure-within-doc (pad labels are `-100`, pads add no gradient).

### Isolation result (v2, only `--lowpass` added)

Same config as the v2 baseline above, just `--lowpass` flag added:

| Run | Baseline | Final | Drop | Steps | tokens/s |
|---|---:|---:|---:|---:|---:|
| v2 baseline | 1.021 | 0.348 | +0.673 | 103 | 11,161 |
| v2 + `--lowpass` | 1.015 | 0.555 | +0.461 | **63** | **6,881** |

**Lowpass alone cuts throughput by ~38 %** (and therefore step count by
the same fraction, since each step does the same amount of work).
Baseline eval loss is essentially identical, so the projection isn't
breaking the eval distribution — the loss is just frozen earlier
because we get 60 % as many optimizer steps.

### Why this implementation isn't a win at our scale

1. **`torch.compile` graph-break per LowpassLinear.** `dynamo` can't
   trace through `torch.autograd.Function.apply`, so each of the 144
   wrapped MLP linears breaks the compiled graph into a separate
   segment. With 144 graph breaks per forward + 144 per backward, the
   model effectively runs eager between segments. The Hadamard
   projection itself is cheap (~5 ms via the lifted Triton piecewise
   kernel); the compile fragmentation is the killer.
2. **Gradient checkpointing already eliminates the savings.** With
   `--gradient-checkpointing auto` (on for 4B full-FT on a single H100),
   activations aren't held across the full forward/backward boundary —
   they're recomputed inside each transformer block at backward time.
   The peak GPU memory is dominated by optimizer state and the
   single-block recompute window, neither of which lowpass touches.
3. **The two combine badly.** Disabling checkpointing to make lowpass's
   compression matter pushes the peak memory above 80 GiB on SXM5 even
   with adamw8bit + keep=32 — the activation tensor for a single
   forward pass at seq=4096 is ~40 GiB on its own, and a 50 %
   compression on MLP only saves a few GiB.

### Refactor: model-wide `saved_tensors_hooks`

Per-Linear wrapping turned out to be the wrong abstraction. The
`nanoCPT-hadamard-lowpass` worktree (which achieved +13% throughput at
LoRA scale) installs a single `torch.autograd.graph.saved_tensors_hooks`
context around the training step. Every `save_for_backward` call across
the model is intercepted by a pack/unpack pair — no custom autograd
Function, no graph break, fully compatible with `torch.compile`.

The refactored `lowpass.py` ships this pattern: `activation_save_context(config, seq_len)` returns the hooks context; the train and warmup loops wrap their forward+backward block with it. `pack(tensor)` filters by ndim, hidden-dim range, and sequence-axis match, then Hadamard-projects chunks and stores the compressed tuple. `unpack(packed)` inverse-projects on demand. Default filter is `8000 ≤ hidden_dim ≤ 16000` to target the MLP intermediate (Qwen3.5-4B `intermediate_size = 9216`) and skip the residual stream (`hidden_size = 2560`) and the lm_head output (`vocab_size = 248320`).

### What the hooks actually catch

Diagnostic prints (set `LOWPASS_LOG_SHAPES=N` for first-N-unique
logging) show:

- With gradient checkpointing **ON** (default): the only saved tensor
  the hook sees is `(batch, seq, hidden_size)` — the block input that
  `torch.utils.checkpoint` stores for later recompute. The MLP
  intermediate, attention K/V, and other inside-block tensors are saved
  during *recompute*, in a context the outer hook does not capture.
- With gradient checkpointing **OFF**: the hook sees the MLP
  intermediate `(batch, seq, 9216)`, the lm_head output
  `(batch, seq, 248320)`, and attention shapes. Filtering to the
  `[8000, 16000]` window isolates just the MLP intermediate.

### The two regimes neither work at 4B full FT

| Regime | Compresses | Memory effect | Quality |
|---|---|---:|---|
| ckpt on, hidden 8000-16000 | nothing (block I/O is 2560) | none — pure overhead | trains fine |
| ckpt on, hidden ≥ 64 | block I/O (residual stream) | ~3-5 GiB saved | diverges (keep=32, 48, 60); only keep=64 (lossless) trains |
| ckpt off, hidden 8000-16000 | MLP intermediate | would save ~10 GiB | OOM before first step |

The ckpt-off OOM is the killer. At 4B full FT with adamw8bit (~8 GiB
opt state), model (8 GiB), grads (8 GiB), and uncompressed activations
across all 36 blocks (~50+ GiB), no-checkpointing simply doesn't fit
on 80 GiB H100 even with MLP-intermediate compression. The
`nanoCPT-hadamard-lowpass` worktree got their +13 % at **LoRA**
scale where opt state and grad memory are tiny (~50 MB instead of 16
GiB), leaving plenty of room for full activation residency.

### Throughput numbers at the regimes we could measure

| Run | tokens/s | Steps | Notes |
|---|---:|---:|---|
| v2 baseline (no lowpass, ckpt on) | 11,161 | 103 | reference |
| Per-Linear lowpass (autograd Fn) | 6,881 | 63 | 38 % slowdown from graph breaks |
| Per-Linear + `compiler.disable` | 6,368 | 59 | swap fragmentation for guard overhead |
| Hooks, keep=64 (lossless, no compression) | 7,720 | 71 | proves hook plumbing is sound |
| Hooks, keep=32 (full compression) | 9,743 | 90 | **fast** but loss diverges (model collapses) |
| Hooks, keep=32 + 8000-16000 filter, ckpt off | — | — | OOM at warmup (insufficient memory savings) |

### Verdict

The `--lowpass` code path is correct and the hooks installation matches
the canonical 2026 pattern. At our specific regime (4B full FT, single
80 GiB H100, default ckpt), it cannot reduce peak memory in a way that
either preserves quality or enables a larger batch / seq. The win
demonstrated at LoRA scale doesn't transfer.

Three avenues that *could* unlock it for full FT:
- **Block-aware compression**: install pack/unpack hooks *inside* the
  per-block checkpoint recompute, not just at the outer boundary. Would
  need to patch `torch.utils.checkpoint` or HF's block wrappers.
- **Activation int8 quantization** (instead of frequency dropping) on
  block I/O — quantization noise is lower-bias than low-pass, so might
  preserve quality at the residual stream.
- **FSDP / sequence parallelism** to free the optimizer / activation
  memory enough that no-checkpoint becomes viable, which is where MLP
  intermediate compression actually pays.

v2 Track 2 leaderboard #1 remains the doc-aware adamw_fused baseline
at **+0.6730**.

### Attempt: `@torch.compiler.disable` workaround

To test whether the slowdown was purely from graph fragmentation, the
`LowpassLinear._apply_lowpass` method was wrapped with
`@torch.compiler.disable`. This makes dynamo treat each LowpassLinear
as a single opaque op (no graph break) instead of fragmenting the
compiled region. Result:

| Run | tokens/s | Steps | Peak mem |
|---|---:|---:|---:|
| v2 baseline | 11,161 | 103 | 53.7 GiB |
| v2 + lowpass (no disable) | 6,881 | 63 | 53.7 GiB |
| v2 + lowpass (with disable) | 6,368 | 59 | 53.7 GiB |

`@torch.compiler.disable` did **not** recover throughput — it traded
one source of overhead (graph fragmentation) for another (per-boundary
guard checks + state save/load at the disable points). The decorator
remains in the source because the principle is correct, but it's not
the bottleneck.

### Why the headline gain never materialised

The real reason lowpass doesn't pay off at this configuration:
**peak GPU memory is not set by per-Linear saved activations.** It is
set by `(optimizer_state, model_weights, single-block-recompute
activations)`, in that order, none of which lowpass shrinks:

- Optimizer state (~32 GiB for `adamw_fused` on 4B): always resident.
- Model bf16 weights (~8 GiB): always resident.
- During backward, each transformer block recomputes its forward
  (gradient checkpointing) and momentarily holds intermediate
  activations (~5–10 GiB). Lowpass would reduce these — but they're
  recomputed and discarded per-block, so the *steady-state peak* is
  already low.

The v2 baseline already runs at 53.7 GiB / 80 GiB = 67 %. Adding
lowpass keeps the peak at 53.7 GiB (no savings). With no memory
freed, there's no headroom to enable bigger micro_batch_size /
seq_len that would amortise the projection cost. Lowpass becomes
pure overhead.

### What a real win would need

- **Kernel-fused lowpass linear**: a Triton matmul that simultaneously
  produces `y = x @ W` *and* `x_hat = Hadamard_chunk(x)[:keep]` in a
  single HBM round-trip. The `instant-lowpass` worktree shipped this
  for GraLoRA's mix kernel; an equivalent for plain Linear doesn't yet
  exist.
- **Aggressive compression at the checkpoint boundary**: lowpass needs
  to replace what's stored *between* checkpoint blocks (i.e., the
  block output, not the per-Linear input). At that level the peak
  memory bound is touchable.
- **Or switch from optimizer-state-bound to activation-bound**: use
  `adamw8bit` (frees ~24 GiB of opt state) AND a much larger seq_len
  / mb so activations dominate the peak. Then lowpass on activations
  becomes the lever.

For Track 2's 5-minute budget at 4B full-FT with default optimizer +
checkpointing, lowpass is a net loss. Leaving the `--lowpass` code
path in (it's correct, just unprofitable at this config) for the
future kernel-fusion work; the v2 leaderboard #1 stays the doc-aware
adamw_fused baseline at **+0.673**.

## References

- ConlangCrafter (Alper et al., 2026), arXiv [2508.06094](https://arxiv.org/abs/2508.06094).
- SumTablets (Simmons, 2024), arXiv [2602.22200](https://arxiv.org/abs/2602.22200).
- Linear A Digital Corpus (Salgarella & Castellan, 2015), [aclanthology W15-3715](https://aclanthology.org/W15-3715.pdf).
- Efficient sequence packing without cross-contamination, Krell et al. 2021, arXiv [2107.02027](https://arxiv.org/abs/2107.02027).
- Enhancing training efficiency using packing with flash attention,
  arXiv [2407.09105](https://arxiv.org/abs/2407.09105).
