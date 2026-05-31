"""Smoke tests for the foreign-distribution CPT data path.

Covers:
1. The configurable CPT text-field flag exists with the expected default.
2. A locally-generated synthesis parquet (if present) has the right row shape
   for the CPT loader to consume.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import main


ROOT = Path(__file__).resolve().parents[1]


def _load_synth_module():
    """Load scripts/synthesize_conlang_cpt.py as a module.

    Registered in sys.modules before exec so @dataclass introspection (which
    reads cls.__module__ from sys.modules) works.
    """
    spec = importlib.util.spec_from_file_location(
        "synthesize_conlang_cpt", ROOT / "scripts" / "synthesize_conlang_cpt.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class CptTextFieldDefaultTest(unittest.TestCase):
    def test_default_is_text(self) -> None:
        self.assertEqual(main.DEFAULT_CPT_TEXT_FIELD, "text")


class CptDatasetDefaultTest(unittest.TestCase):
    def test_default_is_hard_typo2_dataset(self) -> None:
        self.assertEqual(main.DEFAULT_DATASET_ID, "TearedModels/conlangcrafter-cpt-bd412d52-hard-typo2")
        self.assertEqual(main.DEFAULT_DATASET_REVISION, "c8aae01b831778315a2522cf20f5111f4ab7b903")
        self.assertEqual(main.EVAL_VERSION, "v3")

    def test_clean_conlang_is_legacy_only(self) -> None:
        self.assertEqual(main.LEGACY_CONLANG_DATASET_ID, "TearedModels/conlangcrafter-cpt-bd412d52")
        self.assertEqual(main.LEGACY_CONLANG_DATASET_REVISION, "5cfd047a92023011326e8383d45d97db22add909")
        self.assertNotEqual(main.DEFAULT_DATASET_ID, main.LEGACY_CONLANG_DATASET_ID)


class LegacyEntrypointTest(unittest.TestCase):
    def test_legacy_targets_live_under_legacy_directory(self) -> None:
        root_run = (ROOT / "run.sh").read_text(encoding="utf-8")
        legacy_run = (ROOT / "legacy" / "run.sh").read_text(encoding="utf-8")

        self.assertNotIn("legacy-conlang-track", root_run)
        self.assertIn("clean-conlang-track2", legacy_run)
        self.assertIn(main.LEGACY_CONLANG_DATASET_ID, legacy_run)


class HeldoutEvalDefaultsTest(unittest.TestCase):
    """The held-out eval feature must be OFF by default so existing runs and
    cached eval tensors are unaffected."""

    def test_constants_default_disabled(self) -> None:
        self.assertEqual(main.DEFAULT_HELDOUT_EVAL_DATASET_ID, "")
        self.assertEqual(main.DEFAULT_HELDOUT_EVAL_DATASET_CONFIG, "")
        self.assertEqual(main.DEFAULT_HELDOUT_EVAL_DATASET_REVISION, "")
        self.assertEqual(main.DEFAULT_HELDOUT_EVAL_SPLIT, "train")


class CptEvalKeyPayloadTest(unittest.TestCase):
    """The eval-cache key must be byte-identical to the legacy one when no
    held-out corpus is active (so old caches/records stay valid), and must
    change when a held-out corpus is set."""

    BASE = dict(
        model_id="m",
        model_revision="r",
        dataset_id="d",
        dataset_config="",
        dataset_revision="rev",
        seq_len=4096,
        eval_blocks=64,
        seed=1337,
        cpt_text_field="text",
        pack_align=1,
    )

    def _key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def test_disabled_matches_legacy_payload(self) -> None:
        # The exact legacy dict that produced the v2 records' cache keys.
        legacy = {
            "model": "m",
            "model_revision": "r",
            "dataset": "d",
            "dataset_config": "",
            "dataset_revision": "rev",
            "seq_len": 4096,
            "eval_blocks": 64,
            "sequence_packing": main.DEFAULT_SEQUENCE_PACKING,
            "packing_strategy": main.DEFAULT_PACKING_STRATEGY,
            "seed": 1337,
            "kind": "all_token_cpt_packed_v3_docaware",
            "text_field": "text",
            "pack_align": 1,
        }
        produced = main._cpt_eval_key_payload(**self.BASE)
        self.assertEqual(produced, legacy)
        self.assertEqual(self._key(produced), self._key(legacy))

    def test_heldout_changes_key(self) -> None:
        disabled = main._cpt_eval_key_payload(**self.BASE)
        enabled = main._cpt_eval_key_payload(
            **self.BASE,
            heldout_eval_active=True,
            heldout_eval_dataset_id="someuser/heldout",
            heldout_eval_dataset_revision="abc",
        )
        self.assertNotEqual(self._key(disabled), self._key(enabled))
        self.assertEqual(enabled["kind"], "all_token_cpt_packed_v4_heldout")
        self.assertEqual(enabled["heldout_eval_dataset"], "someuser/heldout")

    def test_inactive_flag_ignores_heldout_id(self) -> None:
        # Passing a held-out id but active=False must NOT change the key.
        disabled = main._cpt_eval_key_payload(**self.BASE)
        still_disabled = main._cpt_eval_key_payload(
            **self.BASE,
            heldout_eval_active=False,
            heldout_eval_dataset_id="someuser/heldout",
        )
        self.assertEqual(self._key(disabled), self._key(still_disabled))


class GeneratorHardeningTest(unittest.TestCase):
    """The synthesis script's harder-task knobs exist and behave."""

    def test_topic_sets_disjoint(self) -> None:
        mod = _load_synth_module()
        train = set(mod.TRAIN_TOPIC_SEEDS)
        heldout = set(mod.HELDOUT_TOPIC_SEEDS)
        self.assertTrue(train)
        self.assertTrue(heldout)
        self.assertEqual(train & heldout, set(), "train and heldout topics must be disjoint")

    def test_inject_typos_changes_text_deterministically(self) -> None:
        import random as _random

        mod = _load_synth_module()
        text = "kʼu hun waˈla " * 50
        # rate 0 is a no-op
        self.assertEqual(mod.inject_typos(text, 0.0, _random.Random(0)), text)
        a = mod.inject_typos(text, 0.1, _random.Random(123))
        b = mod.inject_typos(text, 0.1, _random.Random(123))
        self.assertEqual(a, b, "same seed -> same corruption")
        self.assertNotEqual(a, text, "typos should change the text")

    def test_typo_rng_is_chunk_deterministic(self) -> None:
        mod = _load_synth_module()
        self.assertEqual(mod.typo_rng(1337, 42).random(), mod.typo_rng(1337, 42).random())
        self.assertNotEqual(mod.typo_rng(1337, 42).random(), mod.typo_rng(1337, 43).random())


class LocalConlangParquetTest(unittest.TestCase):
    """If a local synthesis parquet exists (from running
    `scripts/synthesize_conlang_cpt.py`), check its row schema."""

    def _find_parquet(self) -> Path | None:
        candidates = sorted((ROOT / "data" / "conlang_cpt").glob("*/train.parquet"))
        return candidates[0] if candidates else None

    def test_parquet_shape(self) -> None:
        parquet_path = self._find_parquet()
        if parquet_path is None:
            self.skipTest("no local conlang corpus yet; run scripts/synthesize_conlang_cpt.py")

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_path)
        self.assertGreater(pf.metadata.num_rows, 0)

        columns = set(pf.schema.names)
        self.assertIn("text", columns)

        table = pf.read_row_group(0)
        first_text = table["text"][0].as_py()
        self.assertIsInstance(first_text, str)
        self.assertGreater(len(first_text), 50)


class SynthesisSummaryTest(unittest.TestCase):
    """If a local synthesis summary exists, sanity-check it."""

    def test_summary_fields(self) -> None:
        candidates = sorted((ROOT / "data" / "conlang_cpt").glob("*/synthesis_summary.json"))
        if not candidates:
            self.skipTest("no synthesis_summary.json yet")
        summary = json.loads(candidates[0].read_text(encoding="utf-8"))
        for key in ("language_id", "total_chars", "rows_parquet"):
            self.assertIn(key, summary)
        self.assertGreater(summary["total_chars"], 0)
        self.assertGreater(summary["rows_parquet"], 0)


if __name__ == "__main__":
    unittest.main()
