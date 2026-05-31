#!/usr/bin/env python3
"""Generate a CPT corpus in a ConlangCrafter-produced constructed language.

One-shot pipeline:
1. Pull a language spec from `malper/ConlangCrafter` (HF).
2. Stream-generate fresh prose via Gemini on Vertex AI, using the spec as
   system instructions and rotating topic seeds for diversity.
3. Lexicon-overlap quality gate per chunk; retry up to 2x then discard.
4. Append-only JSONL output during the run so a Ctrl-C is resumable.
5. At the end, convert JSONL to a single parquet and write the spec alongside.

Push to the Hub with `scripts/push_conlang_dataset.py` afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from google import genai
from google.genai import types as genai_types


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "conlang_cpt"
CONLANG_DATASET = "malper/ConlangCrafter"
DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_PROJECT = os.environ.get("VERTEXAI_PROJECT", "tearedcoder")
DEFAULT_LOCATION = os.environ.get("VERTEXAI_LOCATION", "global")

# Diverse topic prompts to avoid repetitive corpus. The model is told to write
# fresh prose in the conlang, no English. ~50 unique seeds; randomly sampled
# per chunk request. Split into TRAIN and HELDOUT so a held-out generalization
# eval corpus can be generated from the SAME grammar but DISJOINT topics
# (--topic-set heldout), making the eval topic-OOD rather than just seed-OOD.
TRAIN_TOPIC_SEEDS = [
    "a description of a village at dawn",
    "a hunter tracking prey through a forest",
    "a parent telling a child a folk tale at night",
    "a dialogue between two travelers meeting on a road",
    "instructions for building a small wooden boat",
    "a song sung during a harvest festival",
    "a description of the seasons and how they change",
    "a person mourning a relative who has died",
    "a riddle told between friends and its solution",
    "a description of the stars and the moon at midnight",
    "a recipe for cooking a stew with fish and roots",
    "a description of a long river flowing to the sea",
    "two children playing a game with stones",
    "a quiet morning preparing food for a family",
    "an argument between siblings that is later reconciled",
    "a wise elder giving advice to a young adult",
    "a description of a market with many sellers and buyers",
    "a description of birds gathering in the sky",
    "a journey across mountains in winter",
    "a description of a sudden storm and its aftermath",
    "a love poem about a distant person",
    "a description of building a stone wall by hand",
    "a worker resting at noon under a tree",
    "a fisherman setting out before sunrise",
    "a chase scene where someone escapes danger",
    "a description of a calm lake in summer",
    "a craftsman making a clay pot",
    "two strangers becoming friends over a shared meal",
    "a description of a small island in the sea",
    "a person learning a new skill from a teacher",
    "a description of a forest after a long rain",
    "a celebration after a long battle ends",
    "a child asking many questions about animals",
    "a long walk through fields at sunset",
    "a person remembering their childhood home",
    "a description of a snake hiding in tall grass",
    "a meeting of elders to decide a difficult question",
    "a description of how fire is made and used",
    "a story about a clever bird outsmarting a predator",
    "a description of a baby learning to walk",
    "two friends parting ways at the edge of the forest",
    "a description of a deep cave and what lives inside",
    "a person crossing a swollen river by raft",
    "a description of an old house with many memories",
    "a dialogue about whether to stay or to leave",
    "a description of an unfamiliar animal seen for the first time",
    "a song about returning home after a long absence",
    "a description of a wedding feast lasting all night",
    "a hunter sharing meat with their family at the end of the day",
    "a description of the wind blowing through tall reeds",
]

# Disjoint from TRAIN_TOPIC_SEEDS — used only for the held-out eval corpus so the
# generalization metric is measured on topics never seen during training.
HELDOUT_TOPIC_SEEDS = [
    "a blacksmith forging a tool over a hot fire",
    "a description of a desert at the hottest hour of the day",
    "a traveler lost in fog trying to find the way back",
    "a grandmother teaching a grandchild to weave cloth",
    "a description of a waterfall heard from far away",
    "two rivals competing in a footrace before a crowd",
    "a healer gathering herbs to treat a sick child",
    "a description of an eclipse and the fear it brings",
    "a shepherd counting a flock at the end of the day",
    "a description of a bridge built across a deep ravine",
    "a potter teaching an apprentice to shape clay",
    "a description of frost forming on grass at dawn",
    "a sailor describing a strange coast seen from the water",
    "a council debating how to share a poor harvest",
    "a description of bees building a hive in a hollow tree",
    "a child afraid of thunder being comforted by a parent",
    "a description of a dye made from crushed berries",
    "two friends repairing a roof after a storm",
    "a description of a herd of animals migrating across a plain",
    "a person teaching another to read the tracks of animals",
    "a description of a spring bubbling up from rock",
    "a feast prepared to welcome a returning traveler",
    "a description of a loom and the cloth woven on it",
    "a quarrel between neighbors over a shared well",
    "a description of an owl hunting in the dark",
    "a person learning to swim in a cold lake",
    "a description of salt gathered from a dried shore",
    "an old hunter passing a bow to a younger one",
    "a description of a comet crossing the night sky",
    "a group digging a channel to bring water to a field",
]

CHUNK_INSTRUCTION = (
    "Write a passage of roughly {target_chars} characters in this constructed "
    "language. Topic: {topic}. Write only in the constructed language. Do not "
    "include any English, glosses, brackets, IPA notation, transliterations, "
    "translations, or commentary. Do not number lines. Do not use Markdown "
    "headings or bullet lists. Write as natural connected prose suitable for "
    "language-model continued pretraining. Use the lexicon and grammar exactly "
    "as described. Vary sentence length. Do not repeat sentences."
)


@dataclass
class ConlangSpec:
    language_id: str
    model: str
    phonology: str
    grammar: str
    lexicon_raw: str
    lexicon_words: list[str]

    def system_prompt(self) -> str:
        return (
            "You write fluent prose in a constructed language. "
            "The language is specified below as PHONOLOGY, GRAMMAR, and LEXICON sections. "
            "Always write only in this language; never produce any English text in your reply, "
            "not even labels, comments, glosses, headings, or translations. "
            "Use only words from the LEXICON below or transparent inflections of them "
            "as described in the GRAMMAR. Follow the GRAMMAR (morphology, syntax, "
            "agreement, word order, case marking, particles) precisely.\n\n"
            f"=== PHONOLOGY ===\n{self.phonology}\n\n"
            f"=== GRAMMAR ===\n{self.grammar}\n\n"
            f"=== LEXICON (CSV) ===\n{self.lexicon_raw}\n"
        )


def parse_lexicon_words(lexicon_csv: str) -> list[str]:
    """Extract surface forms (column 0) from the lexicon CSV string."""
    words: list[str] = []
    try:
        reader = csv.reader(io.StringIO(lexicon_csv))
        for i, row in enumerate(reader):
            if not row:
                continue
            head = row[0].strip()
            if not head:
                continue
            if i == 0 and head.lower() in {"word", "lemma", "form"}:
                continue
            # Strip diacritics? No — keep as-is, match exactly. Also strip
            # stress marks (ˈ) and morpheme boundaries (.) for overlap.
            clean = head.replace("ˈ", "").replace("ˌ", "").replace(".", "").strip("/").strip()
            if clean:
                words.append(clean)
    except Exception:
        pass
    return words


def is_latin_script(lexicon_csv: str) -> bool:
    """Heuristic: the lexicon surface forms are predominantly ASCII letters.

    The synthetic-conlang loss floor is partly a tokenizer artifact — Qwen
    shreds IPA / tone marks into predictable sub-character pieces. A Latin-script
    language tokenizes into fewer, larger pieces, so it has a higher baseline
    loss and is harder to fit. (See docs/foreign-cpt-findings.md.)
    """
    words = parse_lexicon_words(lexicon_csv)
    alpha = [c for w in words for c in w if c.isalpha()]
    if not alpha:
        return False
    ascii_frac = sum(1 for c in alpha if ord(c) < 128) / len(alpha)
    return ascii_frac > 0.9


def load_spec(
    language_id: str | None,
    min_lexicon_words: int = 0,
    latin_script: bool = False,
) -> ConlangSpec:
    ds = load_dataset(CONLANG_DATASET, split="test")
    if language_id:
        row = next((r for r in ds if r["language_id"] == language_id), None)
        if row is None:
            ids = [r["language_id"] for r in ds][:10]
            raise SystemExit(f"language_id {language_id!r} not found. Sample ids: {ids}")
    else:
        # Default: pick a DeepSeek-R1 spec with a long lexicon. With filters, pick
        # the longest-lexicon spec satisfying them (harder = larger lexicon and/or
        # Latin script for a higher tokenizer baseline).
        cands = list(ds)
        if latin_script:
            cands = [r for r in cands if is_latin_script(r["lexicon"])]
        else:
            cands = [r for r in cands if r["model"] == "DeepSeek-R1"]
        if min_lexicon_words > 0:
            cands = [r for r in cands if len(parse_lexicon_words(r["lexicon"])) >= min_lexicon_words]
        if not cands:
            raise SystemExit(
                f"no spec matched filters (latin_script={latin_script}, "
                f"min_lexicon_words={min_lexicon_words}); relax the constraints"
            )
        cands.sort(key=lambda r: len(r["lexicon"]), reverse=True)
        row = cands[0]
    words = parse_lexicon_words(row["lexicon"])
    return ConlangSpec(
        language_id=row["language_id"],
        model=row["model"],
        phonology=row["phonology"],
        grammar=row["grammar"],
        lexicon_raw=row["lexicon"],
        lexicon_words=words,
    )


WORD_RE = re.compile(r"[^\s\.,!\?;:\"'()\[\]{}<>—–\-/\\]+", re.UNICODE)
ENGLISH_WORD_RE = re.compile(r"\b[a-zA-Z]{2,}\b")
COMMON_ENGLISH = {
    "the", "and", "of", "to", "in", "a", "is", "that", "for", "it",
    "with", "as", "on", "are", "this", "was", "by", "be", "from", "or",
    "an", "have", "has", "but", "not", "you", "we", "they", "he", "she",
    "his", "her", "their", "its", "which", "who", "what", "when", "where",
    "would", "could", "should", "will", "can", "may", "translation", "note",
    "example", "english", "language", "constructed", "phonology", "grammar",
    "lexicon", "passage", "sentence", "story", "description", "dialogue",
}


def english_word_ratio(text: str) -> float:
    words = ENGLISH_WORD_RE.findall(text)
    if not words:
        return 0.0
    eng = sum(1 for w in words if w.lower() in COMMON_ENGLISH)
    return eng / max(len(words), 1)


def tokenize_words(text: str) -> list[str]:
    return [w for w in WORD_RE.findall(text) if w]


def lexicon_overlap(text: str, lexicon_roots: list[str]) -> float:
    """Fraction of word tokens that contain a lexicon root as a substring.

    ConlangCrafter languages are typically polysynthetic with rich agreement
    morphology, so a single root like ``kɤlɯn`` may surface as
    ``kɤlɯnkʼɤt``. Substring matching is robust to this; exact matching is
    far too strict.
    """
    words = tokenize_words(text)
    if not words:
        return 0.0
    # Only roots of length >= 2 are useful as substring markers.
    roots = [r for r in lexicon_roots if len(r) >= 2]
    hits = 0
    for w in words:
        for r in roots:
            if r in w:
                hits += 1
                break
    return hits / max(len(words), 1)


def inject_typos(text: str, rate: float, rng: random.Random) -> str:
    """Perturb characters at `rate` to make the corpus harder to memorize.

    Substitutions draw from the chunk's own character set, so typos stay in the
    conlang's script (no foreign characters introduced). Operations: delete,
    duplicate, substitute, transpose. Applied AFTER the quality gate so overlap
    is measured on clean text. Note: typos add irreducible entropy — they raise
    the loss floor and break rote string memorization, but part of the added
    difficulty is noise rather than deeper structure. Keep the rate modest
    (~1-3%). Do NOT apply to the held-out eval corpus.
    """
    if rate <= 0.0 or not text:
        return text
    alphabet = list(set(text))
    out: list[str] = []
    for ch in text:
        if rng.random() < rate:
            op = rng.randint(0, 3)
            if op == 0:  # delete
                continue
            if op == 1:  # duplicate
                out.append(ch)
                out.append(ch)
                continue
            if op == 2:  # substitute with another char from the same script
                out.append(rng.choice(alphabet))
                continue
            # transpose with the previously emitted char
            if out:
                out[-1], ch = ch, out[-1]
            out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


def typo_rng(seed: int, chunk_id: int) -> random.Random:
    """Stable per-chunk RNG for typo injection across resumes."""
    return random.Random(f"{seed}:{chunk_id}")


@dataclass
class QualityGate:
    min_chars: int
    min_lex_overlap: float
    max_english_ratio: float

    def check(self, text: str, lexicon_roots: list[str]) -> tuple[bool, str]:
        if len(text) < self.min_chars:
            return False, f"short:{len(text)}"
        eng = english_word_ratio(text)
        if eng > self.max_english_ratio:
            return False, f"english:{eng:.2f}"
        ov = lexicon_overlap(text, lexicon_roots)
        if ov < self.min_lex_overlap:
            return False, f"lex:{ov:.2f}"
        return True, "ok"


async def generate_one(
    client: genai.Client,
    model: str,
    spec: ConlangSpec,
    topic: str,
    target_chars: int,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict]:
    # IPA / non-Latin tokens are sub-character on the Gemini tokenizer, so
    # target_chars * 2 underestimates badly; cap at 8192. We need slack so the
    # model finishes naturally with FinishReason.STOP — when it hits
    # MAX_TOKENS, the truncated final part may have text=None and resp.text
    # returns "" even after thousands of generated tokens. Always leave
    # headroom rather than tracking target_chars tightly.
    max_out = 8192
    config = genai_types.GenerateContentConfig(
        system_instruction=spec.system_prompt(),
        temperature=0.95,
        top_p=0.95,
        max_output_tokens=max_out,
        candidate_count=1,
        # Gemini 3.5 Flash uses extended thinking by default; it would eat
        # the entire token budget on thoughts (e.g. 764 thinking tokens, 32
        # output) for this kind of generation task. Disable it.
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )
    prompt = CHUNK_INSTRUCTION.format(target_chars=target_chars, topic=topic)
    async with semaphore:
        t0 = time.time()
        resp = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        dt = time.time() - t0
    # resp.text returns "" when the (only) part has text=None — happens on
    # MAX_TOKENS truncation. Fall back to manual part scraping.
    text = (resp.text or "").strip()
    if not text and resp.candidates:
        parts = (resp.candidates[0].content.parts or []) if resp.candidates[0].content else []
        text = "".join(p.text for p in parts if p.text).strip()
    meta = {
        "topic": topic,
        "latency_s": round(dt, 2),
        "finish_reason": str(resp.candidates[0].finish_reason) if resp.candidates else "?",
    }
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        meta["prompt_tokens"] = int(getattr(usage, "prompt_token_count", 0) or 0)
        meta["output_tokens"] = int(getattr(usage, "candidates_token_count", 0) or 0)
    return text, meta


async def run_loop(
    client: genai.Client,
    model: str,
    spec: ConlangSpec,
    target_output_tokens: int,
    target_chars_per_chunk: int,
    concurrency: int,
    gate: QualityGate,
    output_jsonl: Path,
    topic_seeds: list[str],
    seed: int = 1337,
    typo_rate: float = 0.0,
    max_retries: int = 2,
    debug_dir: Path | None = None,
    debug_first_n: int = 3,
) -> dict:
    random.seed(seed)
    lexicon_roots = list(spec.lexicon_words)
    semaphore = asyncio.Semaphore(concurrency)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Resume from previous run: count existing chunks, chars, and an
    # estimated output-token sum (~1.3 chars/token for IPA-heavy conlangs;
    # this is a lower-bound estimate so we slightly over-generate on resume).
    written_chars = 0
    accepted_output_tokens = 0
    chunk_id = 0
    if output_jsonl.exists():
        for line in output_jsonl.open():
            try:
                row = json.loads(line)
                written_chars += len(row.get("text", ""))
                accepted_output_tokens += int(row.get("output_tokens", 0))
                chunk_id += 1
            except Exception:
                pass
        print(f"[resume] {chunk_id} chunks, {written_chars} chars, ~{accepted_output_tokens} tokens from prior run")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fh = output_jsonl.open("a")
    accepted = chunk_id
    rejected = 0
    rejected_reasons: dict[str, int] = {}
    total_prompt_tokens = 0
    total_output_tokens = 0
    t_start = time.time()
    in_flight: set[asyncio.Task] = set()
    next_chunk_id = chunk_id

    def schedule_one():
        nonlocal next_chunk_id
        topic = random.choice(topic_seeds)
        task = asyncio.create_task(
            generate_one(client, model, spec, topic, target_chars_per_chunk, semaphore)
        )
        task.chunk_id = next_chunk_id
        task.attempts = 0
        task.topic = topic
        next_chunk_id += 1
        in_flight.add(task)

    # Prime the pump
    while len(in_flight) < concurrency and accepted_output_tokens < target_output_tokens:
        schedule_one()

    while in_flight:
        done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            in_flight.discard(task)
            try:
                text, meta = task.result()
            except Exception as e:
                rejected += 1
                rejected_reasons[type(e).__name__] = rejected_reasons.get(type(e).__name__, 0) + 1
                print(f"[err] chunk {task.chunk_id}: {type(e).__name__}: {e}", flush=True)
                if accepted_output_tokens < target_output_tokens:
                    schedule_one()
                continue

            total_prompt_tokens += meta.get("prompt_tokens", 0)
            total_output_tokens += meta.get("output_tokens", 0)
            ok, reason = gate.check(text, lexicon_roots)
            # Dump first few raw outputs for eyeballing.
            if debug_dir is not None and (accepted + rejected) < debug_first_n:
                (debug_dir / f"raw_{accepted + rejected:03d}.txt").write_text(
                    f"[topic={task.topic}] [gate={reason}] [len={len(text)}]\n\n{text}\n",
                    encoding="utf-8",
                )
            if not ok:
                rejected_reasons[reason.split(":")[0]] = rejected_reasons.get(reason.split(":")[0], 0) + 1
                print(f"[reject] chunk={task.chunk_id} attempt={task.attempts} reason={reason}", flush=True)
                if task.attempts < max_retries:
                    # Retry with a different topic seed.
                    task.attempts += 1
                    new_topic = random.choice(topic_seeds)
                    retry = asyncio.create_task(
                        generate_one(client, model, spec, new_topic, target_chars_per_chunk, semaphore)
                    )
                    retry.chunk_id = task.chunk_id
                    retry.attempts = task.attempts
                    retry.topic = new_topic
                    in_flight.add(retry)
                    continue
                rejected += 1
                if accepted_output_tokens < target_output_tokens:
                    schedule_one()
                continue

            # Accept. Inject typos AFTER the gate passed (gate measured clean
            # text). Seed per-chunk so the corruption is reproducible across
            # resumes regardless of completion order.
            if typo_rate > 0.0:
                text = inject_typos(text, typo_rate, typo_rng(seed, task.chunk_id))
            row = {
                "text": text,
                "topic": task.topic,
                "chunk_id": task.chunk_id,
                "latency_s": meta["latency_s"],
                "output_tokens": meta.get("output_tokens", 0),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            accepted += 1
            written_chars += len(text)
            accepted_output_tokens += int(meta.get("output_tokens", 0) or 0)

            # Print on every accept for the first 10, then every 5 thereafter.
            log_now = (
                accepted <= 10
                or accepted % 5 == 0
                or accepted_output_tokens >= target_output_tokens
            )
            if log_now:
                elapsed = time.time() - t_start
                rate_tok = accepted_output_tokens / max(elapsed, 1e-6)
                eta = max(0, (target_output_tokens - accepted_output_tokens) / max(rate_tok, 1e-6))
                print(
                    f"[gen] accepted={accepted} rejected={rejected} "
                    f"out_tok={accepted_output_tokens}/{target_output_tokens} "
                    f"({100*accepted_output_tokens/target_output_tokens:.1f}%) "
                    f"chars={written_chars} rate={rate_tok:.0f} tok/s "
                    f"eta={eta/60:.1f}min tok_in={total_prompt_tokens}",
                    flush=True,
                )

            if accepted_output_tokens < target_output_tokens:
                schedule_one()

    fh.close()
    return {
        "accepted": accepted,
        "rejected": rejected,
        "rejected_reasons": rejected_reasons,
        "total_chars": written_chars,
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "elapsed_s": round(time.time() - t_start, 1),
    }


def write_parquet_from_jsonl(jsonl_path: Path, parquet_path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No rows in {jsonl_path}")
    table = pa.Table.from_pylist(
        [
            {
                "text": r["text"],
                "topic": r.get("topic", ""),
                "chunk_id": int(r.get("chunk_id", i)),
            }
            for i, r in enumerate(rows)
        ]
    )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path, compression="zstd")
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language-id", default=None, help="ConlangCrafter language_id (default: longest DeepSeek-R1 spec)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Vertex Gemini model id")
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--location", default=DEFAULT_LOCATION)
    p.add_argument("--target-tokens", type=int, default=10_000_000, help="Approx output token budget (≈4 chars/token)")
    p.add_argument("--chunk-chars", type=int, default=4000, help="Target characters per chunk")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--min-lex-overlap", type=float, default=0.50)
    p.add_argument("--max-english-ratio", type=float, default=0.05)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=1337, help="RNG seed for topic sampling and typo injection")
    p.add_argument(
        "--topic-set",
        choices=("train", "heldout"),
        default="train",
        help="train = TRAIN_TOPIC_SEEDS; heldout = disjoint HELDOUT_TOPIC_SEEDS for the generalization eval corpus",
    )
    p.add_argument(
        "--variant",
        default="",
        help="Suffix appended to the output dir (e.g. 'heldout', 'typo2') so variants of the same language_id don't collide",
    )
    p.add_argument(
        "--min-lexicon-words",
        type=int,
        default=0,
        help="When auto-selecting a spec, require at least this many lexicon words (harder = larger lexicon)",
    )
    p.add_argument(
        "--latin-script",
        action="store_true",
        help="Auto-select a Latin-script spec (higher tokenizer baseline, harder to fit) instead of the default IPA spec",
    )
    p.add_argument(
        "--typo-rate",
        type=float,
        default=0.02,
        help="Per-character corruption rate applied AFTER the quality gate (train only). Default 0.02 is the canonical harder-corpus setting; pass 0 to disable. Forced to 0 for --topic-set heldout (the eval corpus stays clean).",
    )
    p.add_argument("--smoke", action="store_true", help="Tiny run: ~50k chars, conc=4")
    args = p.parse_args()

    if args.smoke:
        args.target_tokens = 50_000
        args.concurrency = 4

    if args.topic_set == "heldout" and args.typo_rate > 0.0:
        # The held-out eval corpus must stay clean, otherwise the generalization
        # metric would be measuring tolerance to noise rather than the language.
        print(
            f"[info] --topic-set heldout: forcing --typo-rate 0 (was {args.typo_rate}); "
            "the held-out eval corpus is kept clean.",
            file=sys.stderr,
        )
        args.typo_rate = 0.0

    topic_seeds = TRAIN_TOPIC_SEEDS if args.topic_set == "train" else HELDOUT_TOPIC_SEEDS

    print(f"loading conlang spec from {CONLANG_DATASET}…")
    spec = load_spec(args.language_id, min_lexicon_words=args.min_lexicon_words, latin_script=args.latin_script)
    print(f"  language_id={spec.language_id}  generator={spec.model}  lexicon_words={len(spec.lexicon_words)}")

    dir_name = spec.language_id + (f"_{args.variant}" if args.variant else "")
    output_dir = args.output_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "chunks.jsonl"
    parquet_path = output_dir / "train.parquet"
    spec_path = output_dir / "spec.md"
    summary_path = output_dir / "synthesis_summary.json"

    spec_path.write_text(
        f"# ConlangCrafter spec {spec.language_id}\n"
        f"Generator: {spec.model}\n\n"
        f"## Phonology\n\n{spec.phonology}\n\n"
        f"## Grammar\n\n{spec.grammar}\n\n"
        f"## Lexicon (CSV)\n\n```csv\n{spec.lexicon_raw}\n```\n",
        encoding="utf-8",
    )

    print(f"Vertex AI: project={args.project} location={args.location} model={args.model}")
    client = genai.Client(vertexai=True, project=args.project, location=args.location)

    gate = QualityGate(
        min_chars=args.min_chars,
        min_lex_overlap=args.min_lex_overlap,
        max_english_ratio=args.max_english_ratio,
    )

    print(f"target: {args.target_tokens:,} output tokens (stop on token count)")
    print(f"concurrency: {args.concurrency}  chunk_chars: {args.chunk_chars}")
    print(f"quality gate: min_chars={gate.min_chars} min_lex_overlap={gate.min_lex_overlap} max_english_ratio={gate.max_english_ratio}")
    print(f"topic_set={args.topic_set} ({len(topic_seeds)} seeds)  seed={args.seed}  typo_rate={args.typo_rate}  variant={args.variant or '-'}")

    summary = asyncio.run(
        run_loop(
            client=client,
            model=args.model,
            spec=spec,
            target_output_tokens=args.target_tokens,
            target_chars_per_chunk=args.chunk_chars,
            concurrency=args.concurrency,
            gate=gate,
            output_jsonl=jsonl_path,
            topic_seeds=topic_seeds,
            seed=args.seed,
            typo_rate=args.typo_rate,
            max_retries=args.max_retries,
            debug_dir=output_dir / "debug",
            debug_first_n=5,
        )
    )

    n_rows = write_parquet_from_jsonl(jsonl_path, parquet_path)
    summary_full = {
        **summary,
        "language_id": spec.language_id,
        "generator_model": spec.model,
        "vertex_model": args.model,
        "rows_parquet": n_rows,
        "parquet_path": str(parquet_path),
        "spec_path": str(spec_path),
        "jsonl_path": str(jsonl_path),
        "target_tokens": args.target_tokens,
        "chunk_chars": args.chunk_chars,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "topic_set": args.topic_set,
        "variant": args.variant,
        "typo_rate": args.typo_rate,
        "min_lexicon_words": args.min_lexicon_words,
        "latin_script": args.latin_script,
    }
    summary_path.write_text(json.dumps(summary_full, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary_full, indent=2, ensure_ascii=False))
    print(f"\nWrote parquet: {parquet_path}")
    print(f"Push with: uv run python scripts/push_conlang_dataset.py {output_dir}")


if __name__ == "__main__":
    main()
