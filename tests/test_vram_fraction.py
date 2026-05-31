"""Unit tests for the VRAM allocator-fraction guardrail.

`compute_vram_fraction` is a pure function (no torch / CUDA), so these run
anywhere `main` imports. Run locally with
``uv run python -m unittest tests.test_vram_fraction -v``.
"""

from __future__ import annotations

import unittest

from main import H100_80GB_TOTAL_MEMORY_GIB, compute_vram_fraction


class ComputeVramFractionTest(unittest.TestCase):
    target = 0.92
    reference = H100_80GB_TOTAL_MEMORY_GIB  # ~79.1788 GiB

    def test_80gb_card_uses_target_fraction(self) -> None:
        fraction = compute_vram_fraction(self.reference, self.target, self.reference)
        self.assertAlmostEqual(fraction, self.target, places=6)

    def test_93gb_card_keeps_same_byte_budget(self) -> None:
        detected = 93.0
        fraction = compute_vram_fraction(detected, self.target, self.reference)
        # On the larger NVL SKU the fraction is reduced...
        self.assertLess(fraction, self.target)
        # ...so the effective byte budget matches 92% of the 80 GB reference card.
        self.assertAlmostEqual(
            fraction * detected, self.target * self.reference, places=6
        )

    def test_zero_target_disables_cap(self) -> None:
        self.assertEqual(
            compute_vram_fraction(self.reference, 0.0, self.reference), 0.0
        )
        self.assertEqual(
            compute_vram_fraction(self.reference, -1.0, self.reference), 0.0
        )

    def test_smaller_card_never_raises_cap(self) -> None:
        fraction = compute_vram_fraction(40.0, self.target, self.reference)
        self.assertEqual(fraction, self.target)


if __name__ == "__main__":
    unittest.main()
