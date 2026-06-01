"""Per-Linear activation+gradient compression, port of INSTANT (Doan et al, ICLR 2026).

Mirrors `/home/lev/hermes-home/Research/AdaptiveRoundingSimp-qad-clean/packages/optimizers/src/optimizers/instant.py`.

Each wrapped `nn.Linear` saves a sequence-axis-projected
`x_hat = P @ x` (shape `[..., rank_a, hidden]`) for backward
instead of the full `[..., seq, hidden]` activation. By default the
projector `P` (and a separate gradient-side projector `Q`) are built
from a per-layer SVD of activation/gradient samples collected during a
short calibration phase. With `projector_kind="dct"` the SVD path is
swapped for a fixed DCT basis — calibration then only reports captured
energy rather than building a data-dependent basis.

Backward:
- `grad_w = einsum("nro,nri->oi", Q @ grad_output, P @ x)` — both sides
  projected.
- `grad_x = Q^T @ ((Q @ grad_output) @ weight)` when
  `exact_input_grad=False` (the paper's mode). With
  `exact_input_grad=True` we recover the exact `grad_x = grad_output @ weight`
  at the cost of a separate full-rank matmul.

Calibration: install forward + grad hooks on every `LowpassLinear`,
run `steps` batches with `_calibrating=True` (which disables the
compressed path during calibration), collect per-layer
activation/gradient samples, SVD each pair, and write the rank +
projectors back into the modules. See `calibrate_lowpass`.

The paper does NOT compose with gradient checkpointing — activation
compression is presented as an alternative to ckpt. We follow that
convention: don't stack the two when measuring.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


_FIXED_PROJECTOR_CACHE: dict[tuple[str, int, int, str, int, str], Tensor] = {}


@dataclass(frozen=True)
class LowpassConfig:
    """All knobs follow `InstantConfig` in the reference impl."""

    max_seq_len: int = 256
    activation_energy: float = 0.95
    gradient_energy: float = 0.95
    min_rank: int = 4
    max_rank: int = 64
    max_rank_policy: str = "fixed"
    max_rank_scale: float = 1.0
    max_rank_absolute_cap: int | None = None
    projector_kind: str = "svd"
    oversample: int = 8
    power_iterations: int = 2
    calibration_max_columns: int = 2048
    store_calibration_on_cpu: bool = True
    exact_input_grad: bool = False
    compress_gradients: bool = True
    activation_storage: str = "float"
    chunk_size: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_rank_policy not in {"fixed", "log2", "log2_sq"}:
            raise ValueError(f"unknown max_rank_policy {self.max_rank_policy!r}")
        if self.projector_kind not in {"svd", "dct", "hadamard", "haar", "random"}:
            raise ValueError(f"unknown projector_kind {self.projector_kind!r}")
        if self.activation_storage not in {"float", "int8"}:
            raise ValueError(f"unknown activation_storage {self.activation_storage!r}")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be non-negative")
        if self.chunk_size > 0 and self.projector_kind != "dct":
            raise ValueError("chunk_size currently requires projector_kind='dct'")
        if self.min_rank < 1:
            raise ValueError("min_rank must be >= 1")
        if self.max_rank < 1:
            raise ValueError("max_rank must be >= 1")
        if self.max_rank_scale <= 0.0:
            raise ValueError("max_rank_scale must be > 0")
        if self.max_rank_absolute_cap is not None and self.max_rank_absolute_cap < self.min_rank:
            raise ValueError("max_rank_absolute_cap must be >= min_rank")


@dataclass(frozen=True)
class LowpassStats:
    module_name: str
    activation_rank: int
    gradient_rank: int
    activation_effective_max_rank: int
    gradient_effective_max_rank: int
    activation_columns: int
    gradient_columns: int
    activation_selected_energy: float
    activation_cap_energy: float
    activation_sketch_energy: float
    activation_rank90_total: int
    activation_rank95_total: int
    activation_rank98_total: int
    activation_rank99_total: int
    gradient_selected_energy: float
    gradient_cap_energy: float
    gradient_sketch_energy: float
    gradient_rank90_total: int
    gradient_rank95_total: int
    gradient_rank98_total: int
    gradient_rank99_total: int
    fallback_calls: int


@dataclass(frozen=True)
class _ProjectorResult:
    projector: Tensor
    rank: int
    effective_max_rank: int
    selected_energy: float
    cap_energy: float
    sketch_energy: float
    rank90_total: int
    rank95_total: int
    rank98_total: int
    rank99_total: int


def _sequence_sample_matrix(x: Tensor, seq_len: int) -> Tensor:
    if x.ndim < 3:
        raise ValueError(f"expected [..., seq, hidden], got shape {tuple(x.shape)}")
    work = x.detach()
    if work.shape[-2] > seq_len:
        work = work[..., :seq_len, :]
    elif work.shape[-2] < seq_len:
        pad = seq_len - work.shape[-2]
        work = F.pad(work, (0, 0, 0, pad))
    work = work.float().movedim(-2, 0).contiguous()
    return work.reshape(seq_len, -1)


def _append_sample_columns(
    chunks: list[Tensor],
    current_columns: int,
    sample: Tensor,
    max_columns: int,
    *,
    store_on_cpu: bool,
) -> int:
    remaining = max_columns - current_columns
    if remaining <= 0:
        return current_columns
    take = min(remaining, sample.shape[1])
    if take < sample.shape[1]:
        indices = torch.randperm(sample.shape[1], device=sample.device)[:take]
        sample = sample.index_select(1, indices)
    if store_on_cpu:
        sample = sample.cpu()
    chunks.append(sample.contiguous())
    return current_columns + take


def _rank_from_singular_values(
    values: Tensor, energy_threshold: float, min_rank: int, max_rank: int
) -> int:
    if values.numel() == 0:
        return 0
    energy = values.float().square()
    total = energy.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return min(max_rank, max(min_rank, 1), values.numel())
    cumulative = torch.cumsum(energy, dim=0) / total
    rank = int(
        torch.searchsorted(cumulative, torch.tensor(energy_threshold, device=cumulative.device)).item()
    ) + 1
    rank = max(rank, min_rank)
    rank = min(rank, max_rank, values.numel())
    return rank


def _energy_at_rank(cumulative_energy: Tensor, rank: int) -> float:
    if cumulative_energy.numel() == 0 or rank <= 0:
        return 0.0
    index = min(rank, cumulative_energy.numel()) - 1
    value = float(cumulative_energy[index].detach().cpu())
    return min(max(value, 0.0), 1.0)


def _rank_at_total_energy(cumulative_energy: Tensor, threshold: float) -> int:
    if cumulative_energy.numel() == 0:
        return 0
    reached = cumulative_energy >= threshold
    if not bool(reached.any().detach().cpu()):
        return int(cumulative_energy.numel()) + 1
    return int(torch.nonzero(reached, as_tuple=False)[0].item()) + 1


def _torch_is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None and hasattr(compiler, "is_compiling"):
        return bool(compiler.is_compiling())
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "is_compiling"):
        return bool(dynamo.is_compiling())
    return False


def _triton_disabled() -> bool:
    return os.environ.get("LOWPASS_DISABLE_TRITON", "0") == "1" or os.environ.get(
        "INSTANT_DISABLE_TRITON", "0"
    ) == "1"


def _quantize_symmetric_int8(x: Tensor) -> tuple[Tensor, Tensor]:
    scale = x.float().abs().amax(dim=-1, keepdim=True).div(127.0)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    q = x.float().div(scale).round().clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _dequantize_symmetric_int8(q: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    scale_work = scale.to(dtype)
    if scale_work.shape[-1] != 1:
        block_size = math.ceil(q.shape[-1] / scale_work.shape[-1])
        scale_work = scale_work.repeat_interleave(block_size, dim=-1)[..., : q.shape[-1]]
    return q.to(dtype).mul(scale_work)


def _project_quantize_symmetric_int8(p: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    if x.is_cuda and p.is_cuda and x.ndim == 3 and p.ndim == 2 and not _triton_disabled():
        try:
            from lowpass_triton import project_quantize_int8

            return project_quantize_int8(x, p)
        except Exception:
            pass
    x_hat = torch.einsum("rl,...lc->...rc", p, x)
    return _quantize_symmetric_int8(x_hat)


def _effective_max_rank_for_dimension(projector_dim: int, limit: int, config: LowpassConfig) -> int:
    if limit == 0:
        return 0
    if config.max_rank_policy == "fixed":
        raw_rank = config.max_rank
    else:
        log2_dim = math.log2(max(int(projector_dim), 2))
        if config.max_rank_policy == "log2":
            raw_rank = math.ceil(config.max_rank_scale * log2_dim)
        elif config.max_rank_policy == "log2_sq":
            raw_rank = math.ceil(config.max_rank_scale * log2_dim * log2_dim)
        else:
            raise ValueError(f"unknown max_rank_policy {config.max_rank_policy!r}")
    rank = max(config.min_rank, raw_rank)
    if config.max_rank_absolute_cap is not None:
        rank = min(rank, config.max_rank_absolute_cap)
    return min(rank, limit)


def _effective_max_rank(samples: Tensor, config: LowpassConfig) -> int:
    return _effective_max_rank_for_dimension(int(samples.shape[0]), min(samples.shape), config)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _bit_reverse(value: int, width: int) -> int:
    result = 0
    for _ in range(width):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _hadamard_index_for_sequency(sequency: int, width: int) -> int:
    gray = sequency ^ (sequency >> 1)
    return _bit_reverse(gray, width)


def _fixed_projector_cache_key(
    kind: str, seq_len: int, rank: int, device: torch.device, dtype: torch.dtype
) -> tuple[str, int, int, str, int, str]:
    device_index = -1 if device.index is None else int(device.index)
    return kind, int(seq_len), int(rank), device.type, device_index, str(dtype)


def _dct_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    positions = torch.arange(seq_len, device=device, dtype=torch.float32).add_(0.5)
    freqs = torch.arange(rank, device=device, dtype=torch.float32).unsqueeze(1)
    projector = torch.cos((math.pi / float(seq_len)) * freqs * positions.unsqueeze(0))
    if rank > 0:
        projector[0].mul_(math.sqrt(1.0 / float(seq_len)))
    if rank > 1:
        projector[1:].mul_(math.sqrt(2.0 / float(seq_len)))
    return projector.to(dtype=dtype).contiguous()


def _hadamard_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not _is_power_of_two(seq_len):
        raise ValueError(f"Hadamard token projection requires power-of-two seq_len, got {seq_len}")
    width = int(math.log2(seq_len))
    row_indices = torch.tensor(
        [_hadamard_index_for_sequency(index, width) for index in range(rank)],
        device=device,
        dtype=torch.long,
    )
    columns = torch.arange(seq_len, device=device, dtype=torch.long)
    parity = torch.zeros((rank, seq_len), device=device, dtype=torch.bool)
    for bit in range(width):
        row_bit = torch.bitwise_and(torch.bitwise_right_shift(row_indices[:, None], bit), 1).bool()
        col_bit = torch.bitwise_and(torch.bitwise_right_shift(columns[None, :], bit), 1).bool()
        parity.logical_xor_(row_bit & col_bit)
    projector = torch.where(
        parity,
        torch.tensor(-1.0, device=device, dtype=torch.float32),
        torch.tensor(1.0, device=device, dtype=torch.float32),
    )
    projector.mul_(1.0 / math.sqrt(float(seq_len)))
    return projector.to(dtype=dtype).contiguous()


def _haar_projector(seq_len: int, rank: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if not _is_power_of_two(seq_len):
        raise ValueError(f"Haar token projection requires power-of-two seq_len, got {seq_len}")
    projector = torch.zeros((rank, seq_len), device=device, dtype=torch.float32)
    projector[0].fill_(1.0 / math.sqrt(float(seq_len)))
    row = 1
    block = seq_len
    while row < rank and block > 1:
        half = block // 2
        value = 1.0 / math.sqrt(float(block))
        for start in range(0, seq_len, block):
            if row >= rank:
                break
            projector[row, start : start + half].fill_(value)
            projector[row, start + half : start + block].fill_(-value)
            row += 1
        block //= 2
    return projector.to(dtype=dtype).contiguous()


def _random_orthogonal_projector(
    seq_len: int, rank: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    basis_rank = min(seq_len, max(rank, 64))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17_729 + int(seq_len) * 1_003)
    matrix = torch.randn((seq_len, basis_rank), generator=generator, dtype=torch.float32)
    q, _r = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank].T.to(device=device, dtype=dtype).contiguous()


def _fixed_projector(
    kind: str, seq_len: int, rank: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    rank = min(max(int(rank), 0), int(seq_len))
    if rank <= 0:
        return torch.empty((0, seq_len), device=device, dtype=dtype)
    key = _fixed_projector_cache_key(kind, seq_len, rank, device, dtype)
    cached = _FIXED_PROJECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if kind == "dct":
        projector = _dct_projector(seq_len, rank, device, dtype)
    elif kind == "hadamard":
        projector = _hadamard_projector(seq_len, rank, device, dtype)
    elif kind == "haar":
        projector = _haar_projector(seq_len, rank, device, dtype)
    elif kind == "random":
        projector = _random_orthogonal_projector(seq_len, rank, device, dtype)
    else:
        raise ValueError(f"unknown fixed projector kind {kind!r}")
    _FIXED_PROJECTOR_CACHE[key] = projector
    return projector


def _chunked_basis_coefficients(
    kind: str,
    chunk_size: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if kind != "dct":
        raise ValueError("chunked fixed token basis currently supports only dct")
    return _fixed_projector(kind, chunk_size, min(max(int(rank), 0), chunk_size), device, dtype)


def _project_chunked_fixed_token_basis(kind: str, x: Tensor, rank: int, chunk_size: int) -> Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if x.shape[-2] % chunk_size != 0:
        raise ValueError(f"sequence length {x.shape[-2]} is not divisible by chunk_size {chunk_size}")
    local_rank = min(max(int(rank), 0), int(chunk_size))
    if local_rank <= 0:
        return torch.empty(*x.shape[:-2], 0, x.shape[-1], device=x.device, dtype=x.dtype)
    coefficients = _chunked_basis_coefficients(kind, chunk_size, local_rank, x.device, x.dtype)
    if x.is_cuda and x.ndim == 3 and not _triton_disabled():
        try:
            from lowpass_triton import chunked_project

            return chunked_project(x, coefficients, chunk_size=chunk_size)
        except Exception:
            pass
    chunk_count = x.shape[-2] // chunk_size
    x_chunks = x.reshape(*x.shape[:-2], chunk_count, chunk_size, x.shape[-1])
    projected = torch.einsum("rl,...klc->...krc", coefficients, x_chunks)
    return projected.reshape(*x.shape[:-2], chunk_count * local_rank, x.shape[-1])


def _unproject_chunked_fixed_token_basis(
    kind: str,
    x_hat: Tensor,
    *,
    seq_len: int,
    rank: int,
    chunk_size: int,
) -> Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if seq_len % chunk_size != 0:
        raise ValueError(f"sequence length {seq_len} is not divisible by chunk_size {chunk_size}")
    chunk_count = seq_len // chunk_size
    local_rank = min(max(int(rank), 0), int(chunk_size))
    if local_rank <= 0:
        return torch.empty(*x_hat.shape[:-2], seq_len, x_hat.shape[-1], device=x_hat.device, dtype=x_hat.dtype)
    if x_hat.shape[-2] != chunk_count * local_rank:
        raise ValueError(
            f"chunked projection rank mismatch: got {x_hat.shape[-2]}, expected {chunk_count * local_rank}"
        )
    coefficients = _chunked_basis_coefficients(kind, chunk_size, local_rank, x_hat.device, x_hat.dtype)
    x_hat_chunks = x_hat.reshape(*x_hat.shape[:-2], chunk_count, local_rank, x_hat.shape[-1])
    restored = torch.einsum("rl,...krc->...klc", coefficients, x_hat_chunks)
    return restored.reshape(*x_hat.shape[:-2], seq_len, x_hat.shape[-1])


def _project_fixed_token_basis(kind: str, x: Tensor, rank: int, chunk_size: int = 0) -> Tensor:
    if chunk_size > 0:
        return _project_chunked_fixed_token_basis(kind, x, rank, chunk_size)
    if x.is_cuda and x.ndim == 3 and kind in {"hadamard", "haar"} and not _triton_disabled():
        try:
            from lowpass_triton import piecewise_project

            coefficients_and_segment_len = _piecewise_projector_coefficients(
                kind, x.shape[-2], rank, x.device, x.dtype
            )
            if coefficients_and_segment_len is not None:
                coefficients, segment_len = coefficients_and_segment_len
                return piecewise_project(x, coefficients, segment_len=segment_len)
        except Exception:
            pass
    projector = _fixed_projector(kind, x.shape[-2], rank, x.device, x.dtype)
    return torch.einsum("rl,...lc->...rc", projector, x)


def _unproject_fixed_token_basis(
    kind: str,
    x_hat: Tensor,
    *,
    seq_len: int,
    rank: int,
    chunk_size: int = 0,
) -> Tensor:
    if chunk_size > 0:
        return _unproject_chunked_fixed_token_basis(
            kind, x_hat, seq_len=seq_len, rank=rank, chunk_size=chunk_size
        )
    projector = _fixed_projector(kind, seq_len, rank, x_hat.device, x_hat.dtype)
    return torch.einsum("rl,...ri->...li", projector, x_hat)


def _piecewise_segment_count(kind: str, seq_len: int, rank: int) -> int | None:
    rank = min(max(int(rank), 0), int(seq_len))
    if kind not in {"hadamard", "haar"} or rank <= 0:
        return None
    if not _is_power_of_two(seq_len):
        return None
    segment_count = min(_next_power_of_two(rank), seq_len)
    if seq_len % segment_count != 0:
        return None
    return segment_count


def _piecewise_projector_coefficients(
    kind: str,
    seq_len: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, int] | None:
    segment_count = _piecewise_segment_count(kind, seq_len, rank)
    if segment_count is None:
        return None
    segment_len = seq_len // segment_count
    projector = _fixed_projector(kind, seq_len, rank, device, dtype)
    coefficients = projector[:, ::segment_len].contiguous()
    return coefficients, segment_len


def _project_fixed_token_basis_quantized(
    kind: str, x: Tensor, rank: int, chunk_size: int = 0
) -> tuple[Tensor, Tensor]:
    if chunk_size > 0:
        if x.shape[-2] % chunk_size != 0:
            raise ValueError(f"sequence length {x.shape[-2]} is not divisible by chunk_size {chunk_size}")
        local_rank = min(max(int(rank), 0), int(chunk_size))
        coefficients = _chunked_basis_coefficients(kind, chunk_size, local_rank, x.device, x.dtype)
        if x.is_cuda and x.ndim == 3 and not _triton_disabled():
            try:
                from lowpass_triton import chunked_project_quantize_int8

                return chunked_project_quantize_int8(x, coefficients, chunk_size=chunk_size)
            except Exception:
                pass
        x_hat = _project_chunked_fixed_token_basis(kind, x, local_rank, chunk_size)
        return _quantize_symmetric_int8(x_hat)
    if x.is_cuda and x.ndim == 3 and kind in {"hadamard", "haar"} and not _triton_disabled():
        try:
            from lowpass_triton import piecewise_project_quantize_int8

            coefficients_and_segment_len = _piecewise_projector_coefficients(
                kind, x.shape[-2], rank, x.device, x.dtype
            )
            if coefficients_and_segment_len is not None:
                coefficients, segment_len = coefficients_and_segment_len
                return piecewise_project_quantize_int8(x, coefficients, segment_len=segment_len)
        except Exception:
            pass
    projector = _fixed_projector(kind, x.shape[-2], rank, x.device, x.dtype)
    return _project_quantize_symmetric_int8(projector, x)


def _fixed_transform_projector(
    samples: Tensor,
    *,
    energy_threshold: float,
    config: LowpassConfig,
    device: torch.device,
) -> _ProjectorResult:
    if samples.ndim != 2:
        raise ValueError(f"samples must be rank-2, got shape {tuple(samples.shape)}")
    limit = min(samples.shape)
    if limit == 0:
        raise ValueError("cannot build a lowpass projector from empty samples")
    effective_max_rank = _effective_max_rank(samples, config)
    work = samples.to(device=device, dtype=torch.float32, non_blocking=True)
    total_energy = work.square().sum()
    projector = _fixed_projector(
        config.projector_kind, work.shape[0], effective_max_rank, device, torch.float32
    )
    coefficients = projector.matmul(work)
    component_energy = coefficients.square().sum(dim=1)
    # Fixed token bases are explicit rank sweeps: dct_r8 means keep the first
    # eight basis rows in every layer. Calibration only reports captured energy.
    rank = effective_max_rank
    if bool(torch.isfinite(total_energy).detach().cpu()) and float(total_energy.detach().cpu()) > 0.0:
        cumulative_total_energy = torch.cumsum(component_energy, dim=0) / total_energy
    else:
        cumulative_total_energy = torch.zeros_like(component_energy)
    return _ProjectorResult(
        projector=torch.empty((0, work.shape[0]), device=device, dtype=torch.float32),
        rank=rank,
        effective_max_rank=effective_max_rank,
        selected_energy=_energy_at_rank(cumulative_total_energy, rank),
        cap_energy=_energy_at_rank(cumulative_total_energy, effective_max_rank),
        sketch_energy=_energy_at_rank(cumulative_total_energy, effective_max_rank),
        rank90_total=_rank_at_total_energy(cumulative_total_energy, 0.90),
        rank95_total=_rank_at_total_energy(cumulative_total_energy, 0.95),
        rank98_total=_rank_at_total_energy(cumulative_total_energy, 0.98),
        rank99_total=_rank_at_total_energy(cumulative_total_energy, 0.99),
    )


def _randomized_projector(
    samples: Tensor,
    *,
    energy_threshold: float,
    config: LowpassConfig,
    device: torch.device,
) -> _ProjectorResult:
    if samples.ndim != 2:
        raise ValueError(f"samples must be rank-2, got shape {tuple(samples.shape)}")
    limit = min(samples.shape)
    if limit == 0:
        raise ValueError("cannot build a lowpass projector from empty samples")
    effective_max_rank = _effective_max_rank(samples, config)
    q = min(limit, effective_max_rank + max(0, config.oversample))
    work = samples.to(device=device, dtype=torch.float32, non_blocking=True)
    total_energy = work.square().sum()
    # torch.svd_lowrank is PyTorch's randomized low-rank SVD; calibration should
    # only estimate the top subspace.
    u, s, _v = torch.svd_lowrank(work, q=q, niter=config.power_iterations)
    rank = _rank_from_singular_values(s, energy_threshold, config.min_rank, effective_max_rank)
    singular_energy = s.float().square()
    if bool(torch.isfinite(total_energy).detach().cpu()) and float(total_energy.detach().cpu()) > 0.0:
        cumulative_total_energy = torch.cumsum(singular_energy, dim=0) / total_energy
    else:
        cumulative_total_energy = torch.zeros_like(singular_energy)
    projector = u[:, :rank].T.contiguous()
    return _ProjectorResult(
        projector=projector,
        rank=rank,
        effective_max_rank=effective_max_rank,
        selected_energy=_energy_at_rank(cumulative_total_energy, rank),
        cap_energy=_energy_at_rank(cumulative_total_energy, effective_max_rank),
        sketch_energy=_energy_at_rank(cumulative_total_energy, q),
        rank90_total=_rank_at_total_energy(cumulative_total_energy, 0.90),
        rank95_total=_rank_at_total_energy(cumulative_total_energy, 0.95),
        rank98_total=_rank_at_total_energy(cumulative_total_energy, 0.98),
        rank99_total=_rank_at_total_energy(cumulative_total_energy, 0.99),
    )


class _LowpassLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        activation_projector: Tensor | None,
        gradient_projector: Tensor | None,
        activation_rank: int,
        gradient_rank: int,
        use_lowpass: bool,
        exact_input_grad: bool,
        compress_gradients: bool,
        activation_storage: str,
        projector_kind: str,
        chunk_size: int,
    ) -> Tensor:
        needs_gradient_projector = compress_gradients
        y = F.linear(x, weight, bias)
        use_fixed_projector = projector_kind != "svd"
        chunk_size = int(chunk_size)
        if (
            not use_lowpass
            or x.ndim < 3
            or activation_rank <= 0
            or (needs_gradient_projector and gradient_rank <= 0)
            or (chunk_size > 0 and (not use_fixed_projector or projector_kind != "dct"))
            or (chunk_size > 0 and int(x.shape[-2]) % chunk_size != 0)
            or (
                not use_fixed_projector
                and (
                    activation_projector is None
                    or (needs_gradient_projector and gradient_projector is None)
                    or x.shape[-2] != activation_projector.shape[1]
                    or (needs_gradient_projector and x.shape[-2] != gradient_projector.shape[1])
                )
            )
            or (projector_kind in {"hadamard", "haar"} and not _is_power_of_two(int(x.shape[-2])))
        ):
            ctx.mode = "exact"
            ctx.has_bias = bias is not None
            ctx.save_for_backward(x, weight)
            return y

        work_dtype = x.dtype
        ctx.mode = "lowpass"
        ctx.input_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.has_bias = bias is not None
        ctx.exact_input_grad = exact_input_grad or not compress_gradients
        ctx.compress_gradients = compress_gradients
        ctx.activation_storage = activation_storage
        ctx.projector_kind = projector_kind
        ctx.chunk_size = chunk_size
        ctx.seq_len = int(x.shape[-2])
        ctx.activation_rank = int(min(activation_rank, x.shape[-2]))
        ctx.gradient_rank = int(min(gradient_rank, x.shape[-2])) if gradient_rank > 0 else 0
        if chunk_size > 0:
            ctx.activation_rank = int(min(ctx.activation_rank, chunk_size))
            ctx.gradient_rank = int(min(ctx.gradient_rank, chunk_size)) if ctx.gradient_rank > 0 else 0
        p: Tensor | None = None
        q: Tensor | None = None
        if not use_fixed_projector:
            p = activation_projector.to(device=x.device, dtype=work_dtype)
            q = gradient_projector.to(device=x.device, dtype=work_dtype) if gradient_projector is not None else None
        if activation_storage == "int8":
            if use_fixed_projector:
                x_hat_q, x_hat_scale = _project_fixed_token_basis_quantized(
                    projector_kind,
                    x.to(work_dtype),
                    ctx.activation_rank,
                    chunk_size,
                )
            else:
                x_hat_q, x_hat_scale = _project_quantize_symmetric_int8(p, x.to(work_dtype))
            if compress_gradients and not use_fixed_projector:
                ctx.save_for_backward(x_hat_q, x_hat_scale, weight, p, q)
            else:
                saved_tensors = (
                    (x_hat_q, x_hat_scale, weight)
                    if use_fixed_projector
                    else (x_hat_q, x_hat_scale, weight, p)
                )
                ctx.save_for_backward(*saved_tensors)
        else:
            if use_fixed_projector:
                x_hat = _project_fixed_token_basis(
                    projector_kind, x.to(work_dtype), ctx.activation_rank, chunk_size
                )
            else:
                x_hat = torch.einsum("rl,...lc->...rc", p, x.to(work_dtype))
            if compress_gradients and not use_fixed_projector:
                ctx.save_for_backward(x_hat, weight, p, q)
            else:
                saved_tensors = (x_hat, weight) if use_fixed_projector else (x_hat, weight, p)
                ctx.save_for_backward(*saved_tensors)
        return y

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:  # type: ignore[override]
        if ctx.mode == "exact":
            x, weight = ctx.saved_tensors
            grad_x = grad_weight = grad_bias = None
            if ctx.needs_input_grad[0]:
                grad_x = grad_output.matmul(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.reshape(-1, grad_output.shape[-1]).T.matmul(
                    x.reshape(-1, x.shape[-1])
                )
            if ctx.has_bias and ctx.needs_input_grad[2]:
                reduce_dims = tuple(range(grad_output.ndim - 1))
                grad_bias = grad_output.sum(dim=reduce_dims)
            return grad_x, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None

        use_fixed_projector = ctx.projector_kind != "svd"
        if ctx.activation_storage == "int8":
            if use_fixed_projector:
                x_hat_q, x_hat_scale, weight = ctx.saved_tensors
            elif ctx.compress_gradients:
                x_hat_q, x_hat_scale, weight, p, q = ctx.saved_tensors
            else:
                x_hat_q, x_hat_scale, weight, p = ctx.saved_tensors
            work_dtype = grad_output.dtype if use_fixed_projector else p.dtype
            x_hat = _dequantize_symmetric_int8(x_hat_q, x_hat_scale, work_dtype)
        else:
            if use_fixed_projector:
                x_hat, weight = ctx.saved_tensors
            elif ctx.compress_gradients:
                x_hat, weight, p, q = ctx.saved_tensors
            else:
                x_hat, weight, p = ctx.saved_tensors
            work_dtype = x_hat.dtype if use_fixed_projector else p.dtype
        go = grad_output.to(work_dtype)
        grad_x = grad_weight = grad_bias = None

        if ctx.needs_input_grad[1]:
            if use_fixed_projector:
                go_for_w = _project_fixed_token_basis(
                    ctx.projector_kind, go, ctx.activation_rank, ctx.chunk_size
                )
            else:
                go_for_w = torch.einsum("rl,...lo->...ro", p, go)
            grad_weight = torch.einsum(
                "nro,nri->oi",
                go_for_w.reshape(-1, go_for_w.shape[-2], go_for_w.shape[-1]),
                x_hat.reshape(-1, x_hat.shape[-2], x_hat.shape[-1]),
            ).to(ctx.weight_dtype)

        if ctx.needs_input_grad[0]:
            if ctx.exact_input_grad:
                grad_x = go.matmul(weight.to(work_dtype)).to(ctx.input_dtype)
            else:
                if not ctx.compress_gradients:
                    raise RuntimeError(
                        "compressed input gradients require compress_gradients=True"
                    )
                if use_fixed_projector:
                    go_hat = _project_fixed_token_basis(
                        ctx.projector_kind, go, ctx.gradient_rank, ctx.chunk_size
                    )
                else:
                    go_hat = torch.einsum("rl,...lo->...ro", q, go)
                gx_hat = go_hat.matmul(weight.to(work_dtype))
                if use_fixed_projector:
                    grad_x = _unproject_fixed_token_basis(
                        ctx.projector_kind,
                        gx_hat,
                        seq_len=ctx.seq_len,
                        rank=ctx.gradient_rank,
                        chunk_size=ctx.chunk_size,
                    ).to(ctx.input_dtype)
                else:
                    grad_x = torch.einsum("rl,...ri->...li", q, gx_hat).to(ctx.input_dtype)

        if ctx.has_bias and ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=reduce_dims)

        return grad_x, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None


class LowpassLinear(torch.nn.Module):
    """Drop-in `nn.Linear` replacement using INSTANT-style compression."""

    def __init__(self, linear: torch.nn.Linear, config: LowpassConfig | None = None) -> None:
        super().__init__()
        self.config = config or LowpassConfig()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        device = linear.weight.device
        self.register_buffer("activation_projector", torch.empty(0, device=device), persistent=True)
        self.register_buffer("gradient_projector", torch.empty(0, device=device), persistent=True)
        self.register_buffer("fallback_calls", torch.zeros((), dtype=torch.long, device=device), persistent=False)
        self.register_buffer("lowpass_calls", torch.zeros((), dtype=torch.long, device=device), persistent=False)
        self._calibrating = False
        self.activation_transform_rank = 0
        self.gradient_transform_rank = 0

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, config: LowpassConfig | None = None) -> "LowpassLinear":
        return cls(linear, config)

    @property
    def has_projectors(self) -> bool:
        if self.config.projector_kind != "svd":
            return True
        has_activation = self.activation_projector.ndim == 2
        has_gradient = self.gradient_projector.ndim == 2
        return has_activation and (has_gradient or not self.config.compress_gradients)

    def set_projectors(
        self, activation_projector: Tensor, gradient_projector: Tensor | None = None
    ) -> None:
        projector_dtype = self.weight.dtype if self.weight.is_floating_point() else torch.float32
        self.activation_projector = activation_projector.detach().to(
            device=self.weight.device,
            dtype=projector_dtype,
        ).contiguous()
        if gradient_projector is None:
            self.gradient_projector = torch.empty(0, device=self.weight.device)
        else:
            self.gradient_projector = gradient_projector.detach().to(
                device=self.weight.device,
                dtype=projector_dtype,
            ).contiguous()
        self.activation_transform_rank = int(activation_projector.shape[0])
        self.gradient_transform_rank = 0 if gradient_projector is None else int(gradient_projector.shape[0])

    def set_transform_ranks(self, activation_rank: int, gradient_rank: int = 0) -> None:
        self.activation_transform_rank = max(int(activation_rank), 0)
        self.gradient_transform_rank = max(int(gradient_rank), 0)

    def _rank_for_sequence_length(self, seq_len: int, *, gradient: bool = False) -> int:
        configured = self.gradient_transform_rank if gradient else self.activation_transform_rank
        rank_limit = self.config.chunk_size if self.config.chunk_size > 0 else seq_len
        if configured > 0:
            return min(configured, rank_limit)
        return _effective_max_rank_for_dimension(rank_limit, rank_limit, self.config)

    def forward(self, x: Tensor) -> Tensor:
        use_fixed_projector = self.config.projector_kind != "svd"
        chunk_size = int(self.config.chunk_size)
        activation_rank = self._rank_for_sequence_length(x.shape[-2]) if x.ndim >= 3 else 0
        gradient_rank = (
            self._rank_for_sequence_length(x.shape[-2], gradient=True)
            if self.config.compress_gradients and x.ndim >= 3
            else 0
        )
        use_lowpass = (
            self.config.enabled
            and not self._calibrating
            and self.has_projectors
            and x.ndim >= 3
            and activation_rank > 0
            and (chunk_size <= 0 or int(x.shape[-2]) % chunk_size == 0)
            and (
                use_fixed_projector
                or (
                    x.shape[-2] == self.activation_projector.shape[1]
                    and (
                        not self.config.compress_gradients
                        or x.shape[-2] == self.gradient_projector.shape[1]
                    )
                )
            )
            and (self.config.projector_kind not in {"hadamard", "haar"} or _is_power_of_two(int(x.shape[-2])))
        )
        if use_lowpass:
            if not _torch_is_compiling():
                self.lowpass_calls += 1
            p: Tensor | None = None if use_fixed_projector else self.activation_projector
            q: Tensor | None = None if use_fixed_projector else self.gradient_projector
        else:
            if not _torch_is_compiling():
                self.fallback_calls += 1
            p = q = None
        return _LowpassLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            p,
            q,
            activation_rank,
            gradient_rank,
            use_lowpass,
            self.config.exact_input_grad,
            self.config.compress_gradients,
            self.config.activation_storage,
            self.config.projector_kind,
            chunk_size,
        )


_MLP_NAME_PARTS = ("mlp", "gate_proj", "up_proj", "down_proj", "feed_forward", "ffn")


def mlp_module_filter(name: str, _module: torch.nn.Linear) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in _MLP_NAME_PARTS)


def make_module_filter(target: str) -> Callable[[str, torch.nn.Linear], bool] | None:
    normalized = str(target).lower().replace("-", "_")
    if normalized == "mlp":
        return mlp_module_filter
    if normalized in {"all", "every", "any"}:
        return None
    if normalized in {"all_no_lmhead", "all_except_lmhead", "all_minus_lmhead"}:
        return lambda name, _module: (
            "lm_head" not in name.lower() and "embed_tokens" not in name.lower()
        )
    if normalized in {"none", "off"}:
        return lambda _name, _module: False
    raise ValueError(f"unknown lowpass target filter {target!r}")


def replace_linear_with_lowpass(
    model: torch.nn.Module,
    config: LowpassConfig,
    module_filter: Callable[[str, torch.nn.Linear], bool] | None = None,
) -> list[str]:
    replaced: list[str] = []

    def visit(parent: torch.nn.Module, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LowpassLinear):
                continue
            if isinstance(child, torch.nn.Linear):
                if module_filter is None or module_filter(full_name, child):
                    setattr(parent, child_name, LowpassLinear.from_linear(child, config))
                    replaced.append(full_name)
                continue
            visit(child, full_name)

    visit(model, "")
    return replaced


class _CalibrationCollector:
    def __init__(self, module_name: str, module: LowpassLinear, config: LowpassConfig) -> None:
        self.module_name = module_name
        self.module = module
        self.config = config
        self.activation_chunks: list[Tensor] = []
        self.gradient_chunks: list[Tensor] = []
        self.activation_columns = 0
        self.gradient_columns = 0

    def add_activation(self, x: Tensor) -> None:
        sample = _sequence_sample_matrix(x, self.config.max_seq_len)
        self.activation_columns = _append_sample_columns(
            self.activation_chunks,
            self.activation_columns,
            sample,
            self.config.calibration_max_columns,
            store_on_cpu=self.config.store_calibration_on_cpu,
        )

    def add_gradient(self, grad_output: Tensor) -> None:
        sample = _sequence_sample_matrix(grad_output, self.config.max_seq_len)
        self.gradient_columns = _append_sample_columns(
            self.gradient_chunks,
            self.gradient_columns,
            sample,
            self.config.calibration_max_columns,
            store_on_cpu=self.config.store_calibration_on_cpu,
        )

    def finalize(self) -> LowpassStats:
        if not self.activation_chunks:
            raise RuntimeError(f"no activation calibration samples captured for {self.module_name}")
        if self.config.compress_gradients and not self.gradient_chunks:
            raise RuntimeError(f"no gradient calibration samples captured for {self.module_name}")
        activation_samples = torch.cat(self.activation_chunks, dim=1)
        device = self.module.weight.device
        projector_builder = (
            _fixed_transform_projector if self.config.projector_kind != "svd" else _randomized_projector
        )
        activation_result = projector_builder(
            activation_samples,
            energy_threshold=self.config.activation_energy,
            config=self.config,
            device=device,
        )
        q: Tensor | None = None
        gradient_rank = 0
        gradient_effective_max_rank = 0
        gradient_selected_energy = 0.0
        gradient_cap_energy = 0.0
        gradient_sketch_energy = 0.0
        gradient_rank90_total = 0
        gradient_rank95_total = 0
        gradient_rank98_total = 0
        gradient_rank99_total = 0
        if self.config.compress_gradients:
            gradient_samples = torch.cat(self.gradient_chunks, dim=1)
            gradient_result = projector_builder(
                gradient_samples,
                energy_threshold=self.config.gradient_energy,
                config=self.config,
                device=device,
            )
            q = gradient_result.projector
            gradient_rank = gradient_result.rank
            gradient_effective_max_rank = gradient_result.effective_max_rank
            gradient_selected_energy = gradient_result.selected_energy
            gradient_cap_energy = gradient_result.cap_energy
            gradient_sketch_energy = gradient_result.sketch_energy
            gradient_rank90_total = gradient_result.rank90_total
            gradient_rank95_total = gradient_result.rank95_total
            gradient_rank98_total = gradient_result.rank98_total
            gradient_rank99_total = gradient_result.rank99_total
        if self.config.projector_kind == "svd":
            self.module.set_projectors(activation_result.projector, q)
        else:
            self.module.set_transform_ranks(activation_result.rank, gradient_rank)
        return LowpassStats(
            module_name=self.module_name,
            activation_rank=activation_result.rank,
            gradient_rank=gradient_rank,
            activation_effective_max_rank=activation_result.effective_max_rank,
            gradient_effective_max_rank=gradient_effective_max_rank,
            activation_columns=self.activation_columns,
            gradient_columns=self.gradient_columns,
            activation_selected_energy=activation_result.selected_energy,
            activation_cap_energy=activation_result.cap_energy,
            activation_sketch_energy=activation_result.sketch_energy,
            activation_rank90_total=activation_result.rank90_total,
            activation_rank95_total=activation_result.rank95_total,
            activation_rank98_total=activation_result.rank98_total,
            activation_rank99_total=activation_result.rank99_total,
            gradient_selected_energy=gradient_selected_energy,
            gradient_cap_energy=gradient_cap_energy,
            gradient_sketch_energy=gradient_sketch_energy,
            gradient_rank90_total=gradient_rank90_total,
            gradient_rank95_total=gradient_rank95_total,
            gradient_rank98_total=gradient_rank98_total,
            gradient_rank99_total=gradient_rank99_total,
            fallback_calls=int(self.module.fallback_calls.detach().cpu()),
        )


def _try_finalize_collector(collector: _CalibrationCollector) -> LowpassStats | None:
    if not collector.activation_chunks:
        return None
    if collector.config.compress_gradients and not collector.gradient_chunks:
        return None
    return collector.finalize()


def _lowpass_modules(model: torch.nn.Module) -> dict[str, LowpassLinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LowpassLinear)}


def calibrate_lowpass(
    model: torch.nn.Module,
    batches: Iterable[Any],
    loss_fn: Callable[[torch.nn.Module, Any], Tensor],
    *,
    steps: int,
    config: LowpassConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[LowpassStats]:
    lowpass_config = config or LowpassConfig()
    modules = _lowpass_modules(model)
    if not modules:
        return []

    collectors = {
        name: _CalibrationCollector(name, module, lowpass_config)
        for name, module in modules.items()
    }
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    old_flags = {module: module._calibrating for module in modules.values()}
    for module in modules.values():
        module._calibrating = True

    def make_hook(name: str) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs or not isinstance(inputs[0], Tensor) or not isinstance(output, Tensor):
                return
            collectors[name].add_activation(inputs[0])
            if lowpass_config.compress_gradients and output.requires_grad:
                output.register_hook(lambda grad, module_name=name: collectors[module_name].add_gradient(grad))

        return hook

    try:
        for name, module in modules.items():
            hooks.append(module.register_forward_hook(make_hook(name)))

        for index, batch in enumerate(batches):
            if index >= steps:
                break
            model.zero_grad(set_to_none=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, batch)
            if not isinstance(loss, Tensor) or loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar torch.Tensor")
            loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
        for module, flag in old_flags.items():
            module._calibrating = flag
        model.zero_grad(set_to_none=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

    stats: list[LowpassStats] = []
    for collector in collectors.values():
        finalized = _try_finalize_collector(collector)
        if finalized is not None:
            stats.append(finalized)
    return stats
