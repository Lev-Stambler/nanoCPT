"""Unit tests for INSTANT-faithful low-pass `nn.Linear` replacement.

Run locally (`uv run python -m unittest tests.test_lowpass_linear -v`) if
torch is installed. Otherwise the tests skip; the Modal smoke run is the
real verification.
"""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn as nn

    from lowpass import (
        LowpassConfig,
        LowpassLinear,
        _dequantize_symmetric_int8,
        _fixed_projector,
        _project_fixed_token_basis,
        make_module_filter,
        mlp_module_filter,
        replace_linear_with_lowpass,
    )
except ImportError as exc:  # pragma: no cover
    _IMPORT_ERROR: Exception | None = exc
else:
    _IMPORT_ERROR = None


def _skip_if_no_torch(test_fn):
    def wrapper(self, *args, **kwargs):
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"torch unavailable in this env: {_IMPORT_ERROR}")
        return test_fn(self, *args, **kwargs)

    wrapper.__name__ = test_fn.__name__
    return wrapper


def _make_linear_and_lowpass(in_features: int, out_features: int, config: LowpassConfig):
    torch.manual_seed(1337)
    baseline = nn.Linear(in_features, out_features, bias=True)
    lp = LowpassLinear.from_linear(baseline, config)
    return baseline, lp


class ForwardParityTest(unittest.TestCase):
    """Forward is exact in INSTANT: y = F.linear(x, w, b) regardless of projector."""

    @_skip_if_no_torch
    def test_forward_matches_linear(self):
        config = LowpassConfig(projector_kind="dct", max_rank=8, min_rank=4)
        baseline, lp = _make_linear_and_lowpass(256, 384, config)
        x = torch.randn(2, 128, 256, requires_grad=False)
        with torch.no_grad():
            y_ref = baseline(x)
            y_lp = lp(x)
        self.assertTrue(torch.allclose(y_ref, y_lp, atol=1e-6, rtol=1e-6))


class ActivationStorageTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_rejects_unknown_activation_storage(self):
        with self.assertRaises(ValueError):
            LowpassConfig(activation_storage="fp8")

    @_skip_if_no_torch
    def test_int8_storage_matches_float_storage(self):
        torch.manual_seed(2026)
        config_float = LowpassConfig(
            projector_kind="dct",
            max_rank=16,
            min_rank=4,
            exact_input_grad=True,
            compress_gradients=False,
            activation_storage="float",
        )
        config_int8 = LowpassConfig(
            projector_kind="dct",
            max_rank=16,
            min_rank=4,
            exact_input_grad=True,
            compress_gradients=False,
            activation_storage="int8",
        )
        linear_float = nn.Linear(64, 96, bias=True)
        linear_int8 = nn.Linear(64, 96, bias=True)
        linear_int8.load_state_dict(linear_float.state_dict())
        lp_float = LowpassLinear.from_linear(linear_float, config_float)
        lp_int8 = LowpassLinear.from_linear(linear_int8, config_int8)

        x = torch.randn(2, 64, 64)
        x_float = x.detach().clone().requires_grad_(True)
        x_int8 = x.detach().clone().requires_grad_(True)
        y_float = lp_float(x_float)
        y_int8 = lp_int8(x_int8)
        self.assertTrue(torch.allclose(y_float, y_int8, atol=1e-6, rtol=1e-6))

        grad_out = torch.randn_like(y_float)
        y_float.backward(grad_out)
        y_int8.backward(grad_out)
        self.assertTrue(torch.allclose(x_float.grad, x_int8.grad, atol=1e-6, rtol=1e-6))

        rms_ref = lp_float.weight.grad.pow(2).mean().sqrt()
        rms_err = (lp_int8.weight.grad - lp_float.weight.grad).pow(2).mean().sqrt()
        rel_err = (rms_err / rms_ref).item()
        self.assertLess(rel_err, 0.08, msg=f"INT8 storage grad_w RMS-rel error {rel_err:.3f} > 0.08")


class ExactInputGradTest(unittest.TestCase):
    """With exact_input_grad=True grad_x = grad_output @ weight exactly."""

    @_skip_if_no_torch
    def test_input_grad_exact_when_flag_set(self):
        config = LowpassConfig(
            projector_kind="dct",
            max_rank=8,
            min_rank=4,
            exact_input_grad=True,
        )
        baseline, lp = _make_linear_and_lowpass(256, 384, config)

        x = torch.randn(2, 128, 256)
        x_ref = x.clone().detach().requires_grad_(True)
        x_lp = x.clone().detach().requires_grad_(True)
        y_ref = baseline(x_ref)
        y_lp = lp(x_lp)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)
        y_lp.backward(grad_out)
        self.assertIsNotNone(x_ref.grad)
        self.assertIsNotNone(x_lp.grad)
        self.assertTrue(torch.allclose(x_ref.grad, x_lp.grad, atol=1e-5, rtol=1e-5))


class ParamGradToleranceTest(unittest.TestCase):
    """grad_w is approximate via projection; check RMS-rel error is bounded."""

    @_skip_if_no_torch
    def test_param_grad_within_tolerance(self):
        config = LowpassConfig(projector_kind="dct", max_rank=8, min_rank=4)
        baseline, lp = _make_linear_and_lowpass(256, 384, config)

        x = torch.randn(2, 128, 256, requires_grad=True)
        y_ref = baseline(x)
        grad_out = torch.randn_like(y_ref)
        baseline.zero_grad()
        y_ref.backward(grad_out)
        ref_grad_w = baseline.weight.grad.clone()

        x2 = x.detach().clone().requires_grad_(True)
        y_lp = lp(x2)
        lp.weight.grad = None
        y_lp.backward(grad_out)
        lp_grad_w = lp.weight.grad.clone()

        rms_ref = ref_grad_w.pow(2).mean().sqrt()
        rms_err = (lp_grad_w - ref_grad_w).pow(2).mean().sqrt()
        rel_err = (rms_err / rms_ref).item()
        # rank 8 of seq 128 keeps 6.25% of basis; bound at 50% RMS-rel.
        self.assertLess(rel_err, 0.50, msg=f"param grad RMS-rel error {rel_err:.3f} > 0.50")

    @_skip_if_no_torch
    def test_chunked_dct_full_local_rank_matches_linear_grad(self):
        config = LowpassConfig(
            projector_kind="dct",
            max_rank=16,
            min_rank=4,
            chunk_size=16,
            exact_input_grad=True,
            compress_gradients=False,
        )
        baseline, lp = _make_linear_and_lowpass(64, 96, config)

        x = torch.randn(2, 64, 64, requires_grad=True)
        y_ref = baseline(x)
        grad_out = torch.randn_like(y_ref)
        baseline.zero_grad()
        y_ref.backward(grad_out)
        ref_grad_w = baseline.weight.grad.clone()

        x2 = x.detach().clone().requires_grad_(True)
        y_lp = lp(x2)
        lp.weight.grad = None
        y_lp.backward(grad_out)
        self.assertTrue(torch.allclose(ref_grad_w, lp.weight.grad, atol=2e-4, rtol=2e-4))

    @_skip_if_no_torch
    def test_chunked_dct_weight_grad_uses_block_diagonal_basis(self):
        torch.manual_seed(17)
        config = LowpassConfig(
            projector_kind="dct",
            max_rank=4,
            min_rank=4,
            chunk_size=8,
            exact_input_grad=True,
            compress_gradients=False,
        )
        _baseline, lp = _make_linear_and_lowpass(8, 12, config)

        x = torch.randn(2, 16, 8, requires_grad=True)
        grad_out = torch.randn(2, 16, 12)
        lp(x).backward(grad_out)

        basis = _fixed_projector("dct", 8, 4, x.device, x.dtype)
        x_hat = torch.einsum("rl,nklc->nkrc", basis, x.detach().reshape(2, 2, 8, 8))
        go_hat = torch.einsum("rl,nklo->nkro", basis, grad_out.reshape(2, 2, 8, 12))
        expected = torch.einsum(
            "nkro,nkri->oi",
            go_hat,
            x_hat,
        )
        self.assertTrue(torch.allclose(expected, lp.weight.grad, atol=1e-5, rtol=1e-5))


class ChunkedProjectionTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_chunked_dct_projects_each_chunk_independently(self):
        x = torch.randn(2, 64, 32)
        whole = _project_fixed_token_basis("dct", x, rank=8, chunk_size=0)
        chunked = _project_fixed_token_basis("dct", x, rank=8, chunk_size=16)
        self.assertEqual(tuple(whole.shape), (2, 8, 32))
        self.assertEqual(tuple(chunked.shape), (2, 32, 32))


class ChunkedTritonProjectionTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_chunked_dct_triton_matches_reference(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        try:
            from lowpass_triton import chunked_project, chunked_project_quantize_int8
        except ImportError as exc:
            raise unittest.SkipTest(f"Triton unavailable: {exc}") from exc

        torch.manual_seed(2027)
        x = torch.randn(2, 128, 33, device="cuda", dtype=torch.float32)
        basis = _fixed_projector("dct", 16, 8, x.device, x.dtype)
        expected = torch.einsum("rl,nklc->nkrc", basis, x.reshape(2, 8, 16, 33)).reshape(
            2, 64, 33
        )

        projected = chunked_project(x, basis, chunk_size=16, block_channels=16)
        self.assertTrue(torch.allclose(projected, expected, atol=1e-5, rtol=1e-5))

        q, scales = chunked_project_quantize_int8(x, basis, chunk_size=16, block_channels=16)
        dequantized = _dequantize_symmetric_int8(q, scales, torch.float32)
        expected_q = torch.empty_like(q)
        expected_scales = torch.empty_like(scales)
        for block_index, start in enumerate(range(0, expected.shape[-1], 16)):
            block = expected[..., start : start + 16]
            scale = block.abs().amax(dim=-1).div(127.0).clamp_min(torch.finfo(torch.float32).tiny)
            expected_scales[..., block_index] = scale
            expected_q[..., start : start + 16] = (
                block.div(scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
            )
        self.assertTrue(torch.allclose(scales, expected_scales, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.all((q.to(torch.int16) - expected_q.to(torch.int16)).abs() <= 1))
        rel = (dequantized - expected).pow(2).mean().sqrt() / expected.pow(2).mean().sqrt()
        self.assertLess(float(rel), 0.02)


class FallbackOn2DInputTest(unittest.TestCase):
    """2D input has no seq axis → must fall back to exact F.linear semantics."""

    @_skip_if_no_torch
    def test_2d_input_grad_matches_linear_exactly(self):
        config = LowpassConfig(projector_kind="dct", max_rank=8, min_rank=4)
        baseline, lp = _make_linear_and_lowpass(256, 384, config)
        x_ref = torch.randn(16, 256, requires_grad=True)
        x_lp = x_ref.detach().clone().requires_grad_(True)
        y_ref = baseline(x_ref)
        y_lp = lp(x_lp)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)
        y_lp.backward(grad_out)
        self.assertTrue(torch.allclose(x_ref.grad, x_lp.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(
            torch.allclose(baseline.weight.grad, lp.weight.grad, atol=1e-5, rtol=1e-5)
        )


class ReplaceWalkerTest(unittest.TestCase):
    @_skip_if_no_torch
    def test_replace_walks_model_mlp_filter(self):
        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(128, 256, bias=False)
                self.up_proj = nn.Linear(128, 256, bias=False)
                self.down_proj = nn.Linear(256, 128, bias=False)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        class TinyAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(128, 128, bias=False)
                self.k_proj = nn.Linear(128, 128, bias=False)
                self.v_proj = nn.Linear(128, 128, bias=False)

            def forward(self, x):
                return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = TinyMLP()
                self.attn = TinyAttn()

            def forward(self, x):
                return self.mlp(x) + self.attn(x)

        config = LowpassConfig(projector_kind="dct", max_rank=8, min_rank=4)
        model = TinyBlock()
        replaced = replace_linear_with_lowpass(model, config, mlp_module_filter)
        self.assertEqual(len(replaced), 3)
        for name in replaced:
            self.assertTrue("mlp" in name)
        self.assertIsInstance(model.attn.q_proj, nn.Linear)
        self.assertNotIsInstance(model.attn.q_proj, LowpassLinear)
        self.assertIsInstance(model.mlp.gate_proj, LowpassLinear)

    @_skip_if_no_torch
    def test_make_module_filter_choices(self):
        self.assertIs(make_module_filter("mlp"), mlp_module_filter)
        self.assertIsNone(make_module_filter("all"))
        none_filter = make_module_filter("none")
        self.assertFalse(none_filter("mlp.gate_proj", nn.Linear(8, 8)))
        with self.assertRaises(ValueError):
            make_module_filter("nonsense")


if __name__ == "__main__":
    unittest.main()
