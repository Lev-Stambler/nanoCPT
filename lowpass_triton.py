from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - import guard exercised by caller.
    raise RuntimeError("Triton is required for lowpass_triton") from exc


@triton.jit
def _project_quantize_int8_kernel(
    x_ptr,
    p_ptr,
    q_ptr,
    scale_ptr,
    n_items: tl.constexpr,
    seq_len: tl.constexpr,
    channels: tl.constexpr,
    rank: tl.constexpr,
    num_channel_blocks: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = tl.arange(0, BLOCK_L)

    accum = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for start_l in range(0, seq_len, BLOCK_L):
        cur_l = start_l + offs_l
        p_vals = tl.load(
            p_ptr + offs_r[:, None] * seq_len + cur_l[None, :],
            mask=(offs_r[:, None] < rank) & (cur_l[None, :] < seq_len),
            other=0.0,
        )
        x_vals = tl.load(
            x_ptr + pid_n * seq_len * channels + cur_l[:, None] * channels + offs_c[None, :],
            mask=(cur_l[:, None] < seq_len) & (offs_c[None, :] < channels),
            other=0.0,
        )
        accum += tl.dot(p_vals, x_vals, input_precision="ieee", out_dtype=tl.float32)

    valid = (offs_r[:, None] < rank) & (offs_c[None, :] < channels)
    abs_accum = tl.where(valid, tl.abs(accum), 0.0)
    scale = tl.maximum(tl.max(abs_accum, axis=1) / 127.0, 1.1754943508222875e-38)
    scaled = tl.clamp(accum / scale[:, None], -127.0, 127.0)
    q_i32 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [scaled],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q = q_i32.to(tl.int8)

    q_offsets = pid_n * rank * channels + offs_r[:, None] * channels + offs_c[None, :]
    tl.store(q_ptr + q_offsets, q, mask=valid)
    scale_offsets = pid_n * rank * num_channel_blocks + offs_r * num_channel_blocks + pid_c
    tl.store(scale_ptr + scale_offsets, scale, mask=offs_r < rank)


@triton.jit
def _piecewise_project_kernel(
    x_ptr,
    coeff_ptr,
    out_ptr,
    n_items: tl.constexpr,
    seq_len: tl.constexpr,
    channels: tl.constexpr,
    rank: tl.constexpr,
    segment_count: tl.constexpr,
    segment_len: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = tl.arange(0, BLOCK_L)

    accum = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for segment in range(0, segment_count):
        segment_accum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        segment_start = segment * segment_len
        for start_l in range(0, segment_len, BLOCK_L):
            cur_l = segment_start + start_l + offs_l
            x_vals = tl.load(
                x_ptr + pid_n * seq_len * channels + cur_l[:, None] * channels + offs_c[None, :],
                mask=(cur_l[:, None] < segment_start + segment_len) & (offs_c[None, :] < channels),
                other=0.0,
            )
            segment_accum += tl.sum(x_vals.to(tl.float32), axis=0)
        coeff = tl.load(
            coeff_ptr + offs_r * segment_count + segment,
            mask=offs_r < rank,
            other=0.0,
        )
        accum += coeff[:, None].to(tl.float32) * segment_accum[None, :]

    valid = (offs_r[:, None] < rank) & (offs_c[None, :] < channels)
    out_offsets = pid_n * rank * channels + offs_r[:, None] * channels + offs_c[None, :]
    tl.store(out_ptr + out_offsets, accum, mask=valid)


@triton.jit
def _piecewise_project_quantize_int8_kernel(
    x_ptr,
    coeff_ptr,
    q_ptr,
    scale_ptr,
    n_items: tl.constexpr,
    seq_len: tl.constexpr,
    channels: tl.constexpr,
    rank: tl.constexpr,
    segment_count: tl.constexpr,
    segment_len: tl.constexpr,
    num_channel_blocks: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = tl.arange(0, BLOCK_L)

    accum = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for segment in range(0, segment_count):
        segment_accum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        segment_start = segment * segment_len
        for start_l in range(0, segment_len, BLOCK_L):
            cur_l = segment_start + start_l + offs_l
            x_vals = tl.load(
                x_ptr + pid_n * seq_len * channels + cur_l[:, None] * channels + offs_c[None, :],
                mask=(cur_l[:, None] < segment_start + segment_len) & (offs_c[None, :] < channels),
                other=0.0,
            )
            segment_accum += tl.sum(x_vals.to(tl.float32), axis=0)
        coeff = tl.load(
            coeff_ptr + offs_r * segment_count + segment,
            mask=offs_r < rank,
            other=0.0,
        )
        accum += coeff[:, None].to(tl.float32) * segment_accum[None, :]

    valid = (offs_r[:, None] < rank) & (offs_c[None, :] < channels)
    abs_accum = tl.where(valid, tl.abs(accum), 0.0)
    scale = tl.maximum(tl.max(abs_accum, axis=1) / 127.0, 1.1754943508222875e-38)
    scaled = tl.clamp(accum / scale[:, None], -127.0, 127.0)
    q_i32 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [scaled],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q = q_i32.to(tl.int8)

    q_offsets = pid_n * rank * channels + offs_r[:, None] * channels + offs_c[None, :]
    tl.store(q_ptr + q_offsets, q, mask=valid)
    scale_offsets = pid_n * rank * num_channel_blocks + offs_r * num_channel_blocks + pid_c
    tl.store(scale_ptr + scale_offsets, scale, mask=offs_r < rank)


@triton.jit
def _chunked_project_kernel(
    x_ptr,
    coeff_ptr,
    out_ptr,
    n_items: tl.constexpr,
    seq_len: tl.constexpr,
    channels: tl.constexpr,
    rank: tl.constexpr,
    chunk_size: tl.constexpr,
    chunk_count: tl.constexpr,
    total_rank: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_out_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = tl.arange(0, BLOCK_L)
    chunk_idx = (pid_r * BLOCK_R) // rank
    local_r = offs_out_r - chunk_idx * rank

    accum = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for start_l in range(0, chunk_size, BLOCK_L):
        cur_l = start_l + offs_l
        coeff = tl.load(
            coeff_ptr + local_r[:, None] * chunk_size + cur_l[None, :],
            mask=(offs_out_r[:, None] < total_rank) & (cur_l[None, :] < chunk_size),
            other=0.0,
        )
        absolute_l = chunk_idx * chunk_size + cur_l
        x_vals = tl.load(
            x_ptr + pid_n * seq_len * channels + absolute_l[:, None] * channels + offs_c[None, :],
            mask=(absolute_l[:, None] < seq_len) & (offs_c[None, :] < channels),
            other=0.0,
        )
        accum += tl.dot(coeff, x_vals, input_precision="ieee", out_dtype=tl.float32)

    valid = (offs_out_r[:, None] < total_rank) & (offs_c[None, :] < channels)
    out_offsets = pid_n * total_rank * channels + offs_out_r[:, None] * channels + offs_c[None, :]
    tl.store(out_ptr + out_offsets, accum, mask=valid)


@triton.jit
def _chunked_project_quantize_int8_kernel(
    x_ptr,
    coeff_ptr,
    q_ptr,
    scale_ptr,
    n_items: tl.constexpr,
    seq_len: tl.constexpr,
    channels: tl.constexpr,
    rank: tl.constexpr,
    chunk_size: tl.constexpr,
    chunk_count: tl.constexpr,
    total_rank: tl.constexpr,
    num_channel_blocks: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    offs_out_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_l = tl.arange(0, BLOCK_L)
    chunk_idx = (pid_r * BLOCK_R) // rank
    local_r = offs_out_r - chunk_idx * rank

    accum = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for start_l in range(0, chunk_size, BLOCK_L):
        cur_l = start_l + offs_l
        coeff = tl.load(
            coeff_ptr + local_r[:, None] * chunk_size + cur_l[None, :],
            mask=(offs_out_r[:, None] < total_rank) & (cur_l[None, :] < chunk_size),
            other=0.0,
        )
        absolute_l = chunk_idx * chunk_size + cur_l
        x_vals = tl.load(
            x_ptr + pid_n * seq_len * channels + absolute_l[:, None] * channels + offs_c[None, :],
            mask=(absolute_l[:, None] < seq_len) & (offs_c[None, :] < channels),
            other=0.0,
        )
        accum += tl.dot(coeff, x_vals, input_precision="ieee", out_dtype=tl.float32)

    valid = (offs_out_r[:, None] < total_rank) & (offs_c[None, :] < channels)
    abs_accum = tl.where(valid, tl.abs(accum), 0.0)
    scale = tl.maximum(tl.max(abs_accum, axis=1) / 127.0, 1.1754943508222875e-38)
    scaled = tl.clamp(accum / scale[:, None], -127.0, 127.0)
    q_i32 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [scaled],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    q = q_i32.to(tl.int8)

    q_offsets = pid_n * total_rank * channels + offs_out_r[:, None] * channels + offs_c[None, :]
    tl.store(q_ptr + q_offsets, q, mask=valid)
    scale_offsets = pid_n * total_rank * num_channel_blocks + offs_out_r * num_channel_blocks + pid_c
    tl.store(scale_ptr + scale_offsets, scale, mask=offs_out_r < total_rank)


def is_available() -> bool:
    return torch.cuda.is_available()


def project_quantize_int8(
    x: torch.Tensor,
    projector: torch.Tensor,
    *,
    block_channels: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``projector @ x`` and store blockwise symmetric INT8."""
    if not x.is_cuda or not projector.is_cuda:
        raise ValueError("project_quantize_int8 requires CUDA tensors")
    if x.ndim != 3 or projector.ndim != 2:
        raise ValueError(
            f"expected x [N,L,C] and projector [R,L], got {tuple(x.shape)} and {tuple(projector.shape)}"
        )
    if x.shape[1] != projector.shape[1]:
        raise ValueError(f"sequence mismatch: x has L={x.shape[1]}, projector has L={projector.shape[1]}")
    if block_channels <= 0:
        raise ValueError("block_channels must be positive")
    block_channels = max(16, int(block_channels))

    x_work = x.contiguous()
    p_work = projector.contiguous()
    n_items, seq_len, channels = x_work.shape
    rank = p_work.shape[0]
    num_channel_blocks = triton.cdiv(channels, block_channels)
    q = torch.empty((n_items, rank, channels), device=x.device, dtype=torch.int8)
    scales = torch.empty((n_items, rank, num_channel_blocks), device=x.device, dtype=torch.float32)
    block_r = 16
    grid = (n_items, triton.cdiv(rank, block_r), num_channel_blocks)
    _project_quantize_int8_kernel[grid](
        x_work,
        p_work,
        q,
        scales,
        n_items,
        seq_len,
        channels,
        rank,
        num_channel_blocks,
        BLOCK_R=block_r,
        BLOCK_C=block_channels,
        BLOCK_L=64,
        num_warps=4,
        num_stages=4,
    )
    return q, scales


def piecewise_project(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    segment_len: int,
    block_channels: int = 64,
) -> torch.Tensor:
    """Project by piecewise-constant token basis coefficients."""
    if not x.is_cuda or not coefficients.is_cuda:
        raise ValueError("piecewise_project requires CUDA tensors")
    if x.ndim != 3 or coefficients.ndim != 2:
        raise ValueError(
            f"expected x [N,L,C] and coefficients [R,K], got {tuple(x.shape)} and {tuple(coefficients.shape)}"
        )
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")
    x_work = x.contiguous()
    coeff_work = coefficients.contiguous()
    n_items, seq_len, channels = x_work.shape
    rank, segment_count = coeff_work.shape
    if segment_count * int(segment_len) != seq_len:
        raise ValueError(f"segment coefficients imply L={segment_count * int(segment_len)}, but x has L={seq_len}")
    block_channels = max(16, int(block_channels))
    out = torch.empty((n_items, rank, channels), device=x.device, dtype=x.dtype)
    block_r = 16
    grid = (n_items, triton.cdiv(rank, block_r), triton.cdiv(channels, block_channels))
    _piecewise_project_kernel[grid](
        x_work,
        coeff_work,
        out,
        n_items,
        seq_len,
        channels,
        rank,
        segment_count,
        int(segment_len),
        BLOCK_R=block_r,
        BLOCK_C=block_channels,
        BLOCK_L=64,
        num_warps=4,
        num_stages=4,
    )
    return out


def piecewise_project_quantize_int8(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    segment_len: int,
    block_channels: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a piecewise-constant token projection and store INT8."""
    if not x.is_cuda or not coefficients.is_cuda:
        raise ValueError("piecewise_project_quantize_int8 requires CUDA tensors")
    if x.ndim != 3 or coefficients.ndim != 2:
        raise ValueError(
            f"expected x [N,L,C] and coefficients [R,K], got {tuple(x.shape)} and {tuple(coefficients.shape)}"
        )
    if segment_len <= 0:
        raise ValueError("segment_len must be positive")
    x_work = x.contiguous()
    coeff_work = coefficients.contiguous()
    n_items, seq_len, channels = x_work.shape
    rank, segment_count = coeff_work.shape
    if segment_count * int(segment_len) != seq_len:
        raise ValueError(f"segment coefficients imply L={segment_count * int(segment_len)}, but x has L={seq_len}")
    block_channels = max(16, int(block_channels))
    num_channel_blocks = triton.cdiv(channels, block_channels)
    q = torch.empty((n_items, rank, channels), device=x.device, dtype=torch.int8)
    scales = torch.empty((n_items, rank, num_channel_blocks), device=x.device, dtype=torch.float32)
    block_r = 16
    grid = (n_items, triton.cdiv(rank, block_r), num_channel_blocks)
    _piecewise_project_quantize_int8_kernel[grid](
        x_work,
        coeff_work,
        q,
        scales,
        n_items,
        seq_len,
        channels,
        rank,
        segment_count,
        int(segment_len),
        num_channel_blocks,
        BLOCK_R=block_r,
        BLOCK_C=block_channels,
        BLOCK_L=64,
        num_warps=4,
        num_stages=4,
    )
    return q, scales


def chunked_project(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    chunk_size: int,
    block_channels: int = 64,
) -> torch.Tensor:
    """Project each fixed-size token chunk by the same local basis."""
    if not x.is_cuda or not coefficients.is_cuda:
        raise ValueError("chunked_project requires CUDA tensors")
    if x.ndim != 3 or coefficients.ndim != 2:
        raise ValueError(
            f"expected x [N,L,C] and coefficients [R,K], got {tuple(x.shape)} and {tuple(coefficients.shape)}"
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    x_work = x.contiguous()
    coeff_work = coefficients.contiguous()
    n_items, seq_len, channels = x_work.shape
    rank, coeff_chunk_size = coeff_work.shape
    if coeff_chunk_size != chunk_size:
        raise ValueError(f"coefficients use K={coeff_chunk_size}, expected chunk_size={chunk_size}")
    if seq_len % chunk_size != 0:
        raise ValueError(f"sequence length {seq_len} is not divisible by chunk_size {chunk_size}")
    chunk_count = seq_len // chunk_size
    total_rank = chunk_count * rank
    block_channels = max(16, int(block_channels))
    out = torch.empty((n_items, total_rank, channels), device=x.device, dtype=x.dtype)
    block_r = 16
    while block_r > 1 and rank % block_r != 0:
        block_r //= 2
    grid = (n_items, triton.cdiv(total_rank, block_r), triton.cdiv(channels, block_channels))
    _chunked_project_kernel[grid](
        x_work,
        coeff_work,
        out,
        n_items,
        seq_len,
        channels,
        rank,
        chunk_size,
        chunk_count,
        total_rank,
        BLOCK_R=block_r,
        BLOCK_C=block_channels,
        BLOCK_L=64,
        num_warps=4,
        num_stages=4,
    )
    return out


def chunked_project_quantize_int8(
    x: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    chunk_size: int,
    block_channels: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local chunk projections and store blockwise symmetric INT8."""
    if not x.is_cuda or not coefficients.is_cuda:
        raise ValueError("chunked_project_quantize_int8 requires CUDA tensors")
    if x.ndim != 3 or coefficients.ndim != 2:
        raise ValueError(
            f"expected x [N,L,C] and coefficients [R,K], got {tuple(x.shape)} and {tuple(coefficients.shape)}"
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    x_work = x.contiguous()
    coeff_work = coefficients.contiguous()
    n_items, seq_len, channels = x_work.shape
    rank, coeff_chunk_size = coeff_work.shape
    if coeff_chunk_size != chunk_size:
        raise ValueError(f"coefficients use K={coeff_chunk_size}, expected chunk_size={chunk_size}")
    if seq_len % chunk_size != 0:
        raise ValueError(f"sequence length {seq_len} is not divisible by chunk_size {chunk_size}")
    chunk_count = seq_len // chunk_size
    total_rank = chunk_count * rank
    block_channels = max(16, int(block_channels))
    num_channel_blocks = triton.cdiv(channels, block_channels)
    q = torch.empty((n_items, total_rank, channels), device=x.device, dtype=torch.int8)
    scales = torch.empty((n_items, total_rank, num_channel_blocks), device=x.device, dtype=torch.float32)
    block_r = 16
    while block_r > 1 and rank % block_r != 0:
        block_r //= 2
    grid = (n_items, triton.cdiv(total_rank, block_r), num_channel_blocks)
    _chunked_project_quantize_int8_kernel[grid](
        x_work,
        coeff_work,
        q,
        scales,
        n_items,
        seq_len,
        channels,
        rank,
        chunk_size,
        chunk_count,
        total_rank,
        num_channel_blocks,
        BLOCK_R=block_r,
        BLOCK_C=block_channels,
        BLOCK_L=64,
        num_warps=4,
        num_stages=4,
    )
    return q, scales
