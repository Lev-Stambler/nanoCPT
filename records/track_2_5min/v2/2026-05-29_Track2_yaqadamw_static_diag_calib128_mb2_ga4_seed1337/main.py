"""modded-continued-training: a Modal H100 speedrun for fine-tuning.

Track 1 trains Qwen3.5-4B-Base on UltraChat general SFT data by default and
scores the run by the drop in heldout assistant-only loss. The legacy FineMath
continued-pretraining path remains available with ``--data-mode cpt``.

Track 1: 30-minute budget (default)
Track 2: 5-minute sprint
Track 3: 2-hour endurance
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import modal


APP_NAME = "modded-continued-training"
CACHE_MOUNT = Path("/cache")
HF_CACHE = CACHE_MOUNT / "huggingface"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B-Base"
DEFAULT_MODEL_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
DEFAULT_DATASET_ID = "TearedModels/conlangcrafter-cpt-bd412d52"
DEFAULT_DATASET_CONFIG = ""
DEFAULT_DATASET_REVISION = "5cfd047a92023011326e8383d45d97db22add909"
# Legacy CPT dataset — kept so the records under records/track_1_30min/2026-05-*
# can be reproduced. Pass --dataset-id explicitly to use it.
LEGACY_CPT_DATASET_ID = "HuggingFaceTB/finemath"
LEGACY_CPT_DATASET_CONFIG = "finemath-4plus"
LEGACY_CPT_DATASET_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"
DEFAULT_SFT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SFT_DATASET_CONFIG = ""
DEFAULT_SFT_DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DEFAULT_SFT_TRAIN_SPLIT = "train_sft"
DEFAULT_SFT_EVAL_SPLIT = "test_sft"
DEFAULT_EFFECTIVE_TOKENS_PER_STEP = 32_768
DEFAULT_FULL_MICRO_BATCH_SIZE = 1
DEFAULT_EVAL_MICRO_BATCH_SIZE = 2
DEFAULT_MUON_QUANT_BLOCK_SIZE = 2048
DEFAULT_NORMUON_BETA2 = 0.95
DEFAULT_NORMUON_EPS = 1.0e-8
DEFAULT_SEQUENCE_PACKING = True
DEFAULT_PACKING_STRATEGY = "stream_concat_no_padding"
DEFAULT_CPT_TEXT_FIELD = "text"
DEFAULT_LOWPASS_KEEP = 8
DEFAULT_LOWPASS_TARGET_FILTER = "mlp"
LOWPASS_TARGET_FILTER_CHOICES = {"mlp", "all", "none"}
# Eval-correctness version. Bump when a change alters absolute eval-loss
# numbers (e.g. document-aware attention masking, eval-set tokenization
# changes). Records under records/<track>/v<N>/ are only comparable within
# the same version.
EVAL_VERSION = "v2"

DATA_MODE_CHOICES = {"sft", "cpt"}
OPTIMIZER_CHOICES = {
    "auto",
    "adamw8bit",
    "adamw_fused",
    "muon",
    "muon8",
    "normuon",
    "yaqadamw",
}
LR_SCHEDULE_CHOICES = {"constant", "linear", "cosine", "wsd"}
MUON_LR_ADJUSTMENT_CHOICES = {"original", "match_rms_adamw"}

PEAK_GPU_STAT_KEYS = {
    "cuda_max_memory_allocated_gib": "peak_cuda_memory_allocated_gib",
    "cuda_max_memory_reserved_gib": "peak_cuda_memory_reserved_gib",
    "cuda_max_memory_allocated_fraction": "peak_cuda_memory_allocated_fraction",
    "cuda_max_memory_reserved_fraction": "peak_cuda_memory_reserved_fraction",
    "gpu_utilization_percent": "peak_gpu_utilization_percent",
    "gpu_memory_utilization_percent": "peak_gpu_memory_utilization_percent",
    "gpu_memory_used_gib": "peak_gpu_memory_used_gib",
    "gpu_memory_used_fraction": "peak_gpu_memory_used_fraction",
    "gpu_power_watts": "peak_gpu_power_watts",
}

TRACKS: dict[str, dict[str, Any]] = {
    "1": {"name": "30min", "default_minutes": 30.0, "record_dir": "records/track_1_30min"},
    "2": {"name": "5min", "default_minutes": 5.0, "record_dir": "records/track_2_5min"},
    "3": {"name": "2hr", "default_minutes": 120.0, "record_dir": "records/track_3_2hr"},
}

SFT_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "prompter": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}


def _iter_sft_turns(row: dict[str, Any]) -> list[tuple[str, Any]] | None:
    conversations = row.get("conversations")
    if isinstance(conversations, list):
        if not all(isinstance(turn, dict) for turn in conversations):
            return None
        return [(turn.get("from"), turn.get("value")) for turn in conversations]

    messages = row.get("messages")
    if isinstance(messages, list):
        if not all(isinstance(turn, dict) for turn in messages):
            return None
        return [(turn.get("role"), turn.get("content")) for turn in messages]

    return None


def _render_sft_row(
    row: dict[str, Any],
    tokenize_piece,
    seq_len: int,
) -> tuple[list[int], list[int]] | None:
    turns = _iter_sft_turns(row)
    if not turns:
        return None

    input_ids: list[int] = []
    labels: list[int] = []
    supervised_tokens = 0
    for source_role, value in turns:
        if not isinstance(source_role, str):
            return None
        role = SFT_ROLE_MAP.get(source_role.lower())
        if role is None:
            return None

        if not isinstance(value, str):
            value = "" if value is None else json.dumps(value, sort_keys=True)

        prefix_ids = tokenize_piece(f"<|im_start|>{role}\n")
        content_ids = tokenize_piece(value)
        suffix_ids = tokenize_piece("<|im_end|>\n")
        input_ids.extend(prefix_ids)
        input_ids.extend(content_ids)
        input_ids.extend(suffix_ids)

        if role == "assistant":
            labels.extend([-100] * len(prefix_ids))
            labels.extend(content_ids)
            labels.extend(suffix_ids)
            supervised_tokens += len(content_ids) + len(suffix_ids)
        else:
            labels.extend([-100] * (len(prefix_ids) + len(content_ids) + len(suffix_ids)))

    if not input_ids or supervised_tokens == 0:
        return None
    if len(input_ids) != len(labels):
        raise RuntimeError("SFT renderer produced misaligned input_ids and labels")
    if len(input_ids) > seq_len:
        input_ids = input_ids[-seq_len:]
        labels = labels[-seq_len:]
        if not any(label != -100 for label in labels):
            return None
    return input_ids, labels


def _resolve_dataset_defaults(
    data_mode: str,
    dataset_id: str,
    dataset_config: str,
    dataset_revision: str,
) -> tuple[str, str, str]:
    if not dataset_id:
        if data_mode == "sft":
            return DEFAULT_SFT_DATASET_ID, DEFAULT_SFT_DATASET_CONFIG, DEFAULT_SFT_DATASET_REVISION
        return DEFAULT_DATASET_ID, DEFAULT_DATASET_CONFIG, DEFAULT_DATASET_REVISION

    # Older CLI invocations may pass the legacy FineMath CPT defaults explicitly
    # even when running in SFT mode (back when those were the only defaults).
    # Redirect those to the SFT defaults so existing scripts don't break.
    legacy_cpt_tuple = (LEGACY_CPT_DATASET_ID, LEGACY_CPT_DATASET_CONFIG, LEGACY_CPT_DATASET_REVISION)
    if data_mode == "sft" and (dataset_id, dataset_config, dataset_revision) == legacy_cpt_tuple:
        return DEFAULT_SFT_DATASET_ID, DEFAULT_SFT_DATASET_CONFIG, DEFAULT_SFT_DATASET_REVISION
    return dataset_id, dataset_config, dataset_revision


def _supervised_token_count(labels) -> int:
    if isinstance(labels, list):
        if labels and isinstance(labels[0], list):
            return sum(1 for row in labels for label in row[1:] if label != -100)
        return sum(1 for label in labels[1:] if label != -100)
    shifted = labels[..., 1:]
    return int((shifted != -100).sum())


app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("modded-continued-training-cache", create_if_missing=True)
wandb_secret_values = {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")}
if os.environ.get("WANDB_BASE_URL"):
    wandb_secret_values["WANDB_BASE_URL"] = os.environ["WANDB_BASE_URL"]
wandb_env_secret = modal.Secret.from_dict(wandb_secret_values)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .env(
        {
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
            "FLASH_ATTENTION_FORCE_CXX11_ABI": "TRUE",
            "MAX_JOBS": "8",
            "NVCC_THREADS": "2",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "true",
            "TORCH_CUDA_ARCH_LIST": "9.0",
        }
    )
    .apt_install("build-essential", "git", "ninja-build")
    .pip_install("torch==2.8.0", "packaging", "wheel")
    .pip_install(
        "accelerate>=1.0.0",
        "bitsandbytes>=0.46.0",
        "causal-conv1d==1.6.2.post1",
        "datasets>=3.0.0",
        "flash-attn==2.8.3",
        "flash-linear-attention==0.5.0",
        "hf_transfer>=0.1.9",
        "huggingface_hub>=0.30.0",
        "numpy>=2.0.0",
        "nvidia-ml-py>=12.560.30",
        "safetensors>=0.5.0",
        "tilelang",
        "tqdm>=4.66.0",
        "wandb>=0.18.0",
        "git+https://github.com/huggingface/transformers.git",
        extra_options="--no-build-isolation",
    )
    .add_local_python_source("lowpass")
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "record"


def _format_record_text(summary: dict[str, Any]) -> str:
    return (
        f"Track {summary['track']} ({summary['track_name']})\n"
        f"Description: {summary['record_description']}\n"
        f"Contributors: {summary['record_contributors']}\n"
        f"Date: {summary['record_date']}\n"
        f"Data mode: {summary.get('data_mode', 'cpt')}\n"
        f"Loss mask: {summary.get('loss_mask', 'all_tokens')}\n"
        f"Sequence packing: {summary.get('sequence_packing', DEFAULT_SEQUENCE_PACKING)} "
        f"({summary.get('packing_strategy', DEFAULT_PACKING_STRATEGY)})\n"
        f"Minutes: {summary['minutes']}\n"
        f"Eval loss drop: {summary['eval_loss_drop']:.6f}\n"
        f"Baseline eval loss: {summary['baseline_eval_loss']:.6f}\n"
        f"Final eval loss: {summary['final_eval_loss']:.6f}\n"
        f"Steps: {summary['steps']}\n"
        f"Tokens: {summary['tokens']:,}\n"
        f"Supervised tokens: {summary.get('supervised_tokens', summary['tokens']):,}\n"
        f"Eval supervised tokens: {summary.get('eval_supervised_tokens', 0):,}\n"
        f"Compile + warmup (untimed): {summary.get('elapsed_compile_warmup_seconds', 0.0):.1f}s\n"
        f"Elapsed budget: {summary['elapsed_budget_seconds']:.1f}s\n"
        f"Budget tokens/sec: {summary['tokens_per_second']:.0f}\n"
        f"Train-loop elapsed: {summary['elapsed_train_loop_seconds']:.1f}s\n"
        f"Train-loop tokens/sec: {summary['train_loop_tokens_per_second']:.0f}\n"
    )


def _write_local_record(summary: dict[str, Any], artifacts: dict[str, str]) -> Path:
    track = str(summary["track"])
    if track not in TRACKS:
        raise ValueError(f"record track must be one of: {', '.join(TRACKS.keys())}")

    record_name = f"{summary['record_date']}_{_slugify(str(summary['record_description']))}"
    version = str(summary.get("eval_version", EVAL_VERSION))
    record_dir = Path(TRACKS[track]["record_dir"]) / version / record_name
    if record_dir.exists():
        record_dir = record_dir.with_name(f"{record_dir.name}_{summary['run_id']}")
    record_dir.mkdir(parents=True, exist_ok=False)

    for filename in ("main.py", "config.json", "summary.json", "record.txt", "metrics.jsonl"):
        content = artifacts.get(filename)
        if not content and filename == "main.py":
            content = Path(__file__).read_text()
        if content is not None:
            (record_dir / filename).write_text(content)
    return record_dir


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={str(CACHE_MOUNT): cache_volume},
    secrets=[wandb_env_secret],
)
def run_track1(
    minutes: float = 30.0,
    seq_len: int = 4096,
    micro_batch_size: int = 0,
    grad_accum: int = 0,
    lr: float = 0.0,
    weight_decay: float = -1.0,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
    eval_micro_batch_size: int = 0,
    seed: int = 1337,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_id: str = "",
    dataset_config: str = "",
    dataset_revision: str = "",
    cpt_text_field: str = DEFAULT_CPT_TEXT_FIELD,
    data_mode: str = "cpt",
    optimizer_name: Literal[
        "auto",
        "adamw8bit",
        "adamw_fused",
        "muon",
        "muon8",
        "normuon",
        "yaqadamw",
    ] = "auto",
    gradient_checkpointing: Literal["auto", "true", "false"] = "auto",
    muon_lr_adjustment: Literal["original", "match_rms_adamw"] = "match_rms_adamw",
    muon_quant_block_size: int = DEFAULT_MUON_QUANT_BLOCK_SIZE,
    normuon_beta2: float = DEFAULT_NORMUON_BETA2,
    normuon_eps: float = DEFAULT_NORMUON_EPS,
    muon_lr: float = 0.0,
    adamw_tail_lr: float = 0.0,
    yaqa_mode: Literal["static", "online_ema"] = "static",
    yaqa_calib_sequences: int = 256,
    yaqa_min_scale: float = 0.1,
    yaqa_max_scale: float = 10.0,
    yaqa_eps: float = 1.0e-8,
    lowpass: bool = False,
    lowpass_projector_kind: Literal["dct", "hadamard", "haar"] = "dct",
    lowpass_keep: int = DEFAULT_LOWPASS_KEEP,
    lowpass_target_filter: Literal["mlp", "all", "none"] = DEFAULT_LOWPASS_TARGET_FILTER,
    lowpass_min_hidden_dim: int = 8000,
    lowpass_max_hidden_dim: int = 16000,
    lr_schedule: Literal["constant", "linear", "cosine", "wsd"] = "constant",
    lr_decay_fraction: float = 0.1,
    min_lr_ratio: float = 0.0,
    attn_implementation: Literal["flex_attention", "flash_attention_2", "sdpa", "eager"] = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    save_final: bool = False,
    log_every: int = 5,
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_name: str = "",
    wandb_group: str = "",
    wandb_tags: str = "",
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    track: str = "1",
    record_description: str = "",
    record_contributors: str = "",
) -> dict[str, Any]:
    import datetime as dt
    import math
    import os
    import random
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass
    if attn_implementation == "flex_attention":
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel

            Qwen3_5PreTrainedModel._supports_flex_attn = True
            ModelClass._supports_flex_attn = True
            print("enabled qwen3.5 flex_attention support flag", flush=True)
        except Exception as exc:
            print(f"warning: could not enable qwen3.5 flex_attention support flag: {exc}", flush=True)

    if minutes <= 0:
        raise ValueError("--minutes must be positive")
    if seq_len < 128:
        raise ValueError("--seq-len must be at least 128")
    if micro_batch_size < 0 or grad_accum < 0 or eval_micro_batch_size < 0:
        raise ValueError(
            "--micro-batch-size, --grad-accum, and --eval-micro-batch-size "
            "must be non-negative; use 0 for auto"
        )
    if eval_blocks < 1:
        raise ValueError("--eval-blocks must be positive")
    if warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if lr < 0:
        raise ValueError("--lr must be non-negative; use 0 for the mode default")
    if weight_decay < 0.0 and weight_decay != -1.0:
        raise ValueError("--weight-decay must be >= 0; use -1 for the mode default")
    if track not in TRACKS:
        raise ValueError(f"--track must be one of: {', '.join(TRACKS.keys())}")
    data_mode = str(data_mode or "cpt").lower().replace("-", "_")
    if data_mode not in DATA_MODE_CHOICES:
        raise ValueError(f"--data-mode must be one of: {', '.join(sorted(DATA_MODE_CHOICES))}")
    gradient_checkpointing = str(gradient_checkpointing).lower()
    if gradient_checkpointing not in {"auto", "true", "false"}:
        raise ValueError("--gradient-checkpointing must be one of: auto, true, false")
    optimizer_name = str(optimizer_name).lower()
    if optimizer_name not in OPTIMIZER_CHOICES:
        raise ValueError(f"--optimizer-name must be one of: {', '.join(sorted(OPTIMIZER_CHOICES))}")
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("--wandb-mode must be one of: online, offline, disabled")
    if muon_lr_adjustment not in MUON_LR_ADJUSTMENT_CHOICES:
        raise ValueError(
            "--muon-lr-adjustment must be one of: "
            f"{', '.join(sorted(MUON_LR_ADJUSTMENT_CHOICES))}"
        )
    if muon_quant_block_size < 1:
        raise ValueError("--muon-quant-block-size must be positive")
    if normuon_beta2 < 0.0 or normuon_beta2 >= 1.0:
        raise ValueError("--normuon-beta2 must be in [0, 1)")
    if normuon_eps <= 0.0:
        raise ValueError("--normuon-eps must be positive")
    if muon_lr < 0.0:
        raise ValueError("--muon-lr must be non-negative; use 0 to fall back to --lr")
    if adamw_tail_lr < 0.0:
        raise ValueError("--adamw-tail-lr must be non-negative; use 0 to fall back to --lr")
    if yaqa_mode not in {"static", "online_ema"}:
        raise ValueError("--yaqa-mode must be one of: static, online_ema")
    if yaqa_calib_sequences < 1:
        raise ValueError("--yaqa-calib-sequences must be positive")
    if yaqa_min_scale <= 0.0:
        raise ValueError("--yaqa-min-scale must be positive")
    if yaqa_max_scale < yaqa_min_scale:
        raise ValueError("--yaqa-max-scale must be >= --yaqa-min-scale")
    if yaqa_eps <= 0.0:
        raise ValueError("--yaqa-eps must be positive")
    lowpass_target_filter = str(lowpass_target_filter).lower()
    if lowpass_target_filter not in LOWPASS_TARGET_FILTER_CHOICES:
        raise ValueError(
            f"--lowpass-target-filter must be one of: {', '.join(sorted(LOWPASS_TARGET_FILTER_CHOICES))}"
        )
    if lowpass_keep < 1:
        raise ValueError("--lowpass-keep must be positive")
    lr_schedule = str(lr_schedule).lower()
    if lr_schedule not in LR_SCHEDULE_CHOICES:
        raise ValueError(f"--lr-schedule must be one of: {', '.join(sorted(LR_SCHEDULE_CHOICES))}")
    if lr_decay_fraction < 0.0 or lr_decay_fraction >= 1.0:
        raise ValueError("--lr-decay-fraction must be in [0, 1)")
    if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if attn_implementation not in {"flex_attention", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(
            "--attn-implementation must be one of: flex_attention, flash_attention_2, sdpa, eager"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Modal H100 training path")
    if data_mode == "sft" and track != "1":
        raise ValueError("--data-mode sft is currently implemented only for Track 1")

    track_info = TRACKS[track]
    requested_dataset_id = dataset_id
    requested_dataset_config = dataset_config
    requested_dataset_revision = dataset_revision
    dataset_id, dataset_config, dataset_revision = _resolve_dataset_defaults(
        data_mode,
        dataset_id,
        dataset_config,
        dataset_revision,
    )
    requested_micro_batch_size = micro_batch_size
    requested_grad_accum = grad_accum
    requested_eval_micro_batch_size = eval_micro_batch_size
    if micro_batch_size == 0:
        micro_batch_size = DEFAULT_FULL_MICRO_BATCH_SIZE
    if eval_micro_batch_size == 0:
        eval_micro_batch_size = min(micro_batch_size, DEFAULT_EVAL_MICRO_BATCH_SIZE)
    if grad_accum == 0:
        tokens_per_micro_batch = seq_len * micro_batch_size
        grad_accum = max(1, DEFAULT_EFFECTIVE_TOKENS_PER_STEP // tokens_per_micro_batch)
    requested_lr = lr
    requested_weight_decay = weight_decay
    requested_optimizer_name = optimizer_name
    resolved_optimizer_name = "adamw_fused" if optimizer_name == "auto" else optimizer_name
    wandb_tags_list = [tag.strip() for tag in wandb_tags.split(",") if tag.strip()]
    wandb_enabled = bool(wandb_project) and wandb_mode != "disabled"
    if lr <= 0.0:
        lr = 2.0e-5
    requested_muon_lr = muon_lr
    requested_adamw_tail_lr = adamw_tail_lr
    resolved_muon_lr = muon_lr if muon_lr > 0.0 else lr
    resolved_adamw_tail_lr = adamw_tail_lr if adamw_tail_lr > 0.0 else lr
    weight_decay = weight_decay if weight_decay >= 0.0 else 0.1
    checkpointing_enabled = gradient_checkpointing != "false"
    gradient_checkpointing_fallback_used = False
    # Whole-seq lowpass projects across the entire sequence (does not chunk),
    # so document-boundary padding is unnecessary.
    pack_align = 1

    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch._dynamo as dynamo

        dynamo.config.recompile_limit = 64
    except Exception as exc:
        print(f"warning: could not set torch dynamo config: {exc}", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_id = f"{dt.datetime.now(dt.UTC).strftime('%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:8]}"
    run_dir = CACHE_MOUNT / "runs" / run_id
    eval_dir = CACHE_MOUNT / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    bytes_per_gib = 1024**3
    gpu_index = torch.cuda.current_device()
    gpu_props = torch.cuda.get_device_properties(gpu_index)
    gpu_name = torch.cuda.get_device_name(gpu_index)
    nvml = None
    nvml_handle = None
    nvml_error = ""
    peak_gpu_stats: dict[str, float] = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml = pynvml
        nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception as exc:
        nvml_error = str(exc)

    def collect_gpu_stats() -> dict[str, Any]:
        stats: dict[str, Any] = {}
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
            allocated = torch.cuda.memory_allocated(gpu_index)
            reserved = torch.cuda.memory_reserved(gpu_index)
            peak_allocated = torch.cuda.max_memory_allocated(gpu_index)
            peak_reserved = torch.cuda.max_memory_reserved(gpu_index)
            stats.update(
                {
                    "cuda_memory_allocated_gib": allocated / bytes_per_gib,
                    "cuda_memory_reserved_gib": reserved / bytes_per_gib,
                    "cuda_max_memory_allocated_gib": peak_allocated / bytes_per_gib,
                    "cuda_max_memory_reserved_gib": peak_reserved / bytes_per_gib,
                    "cuda_memory_free_gib": free_bytes / bytes_per_gib,
                    "cuda_memory_total_gib": total_bytes / bytes_per_gib,
                    "cuda_memory_reserved_fraction": reserved / max(total_bytes, 1),
                    "cuda_max_memory_reserved_fraction": peak_reserved / max(total_bytes, 1),
                    "cuda_max_memory_allocated_fraction": peak_allocated / max(total_bytes, 1),
                }
            )
        except Exception as exc:
            stats["cuda_memory_stats_error"] = str(exc)

        if nvml is not None and nvml_handle is not None:
            try:
                memory_info = nvml.nvmlDeviceGetMemoryInfo(nvml_handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(nvml_handle)
                power_watts = nvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                stats.update(
                    {
                        "gpu_utilization_percent": utilization.gpu,
                        "gpu_memory_utilization_percent": utilization.memory,
                        "gpu_memory_used_gib": memory_info.used / bytes_per_gib,
                        "gpu_memory_total_gib": memory_info.total / bytes_per_gib,
                        "gpu_memory_used_fraction": memory_info.used / max(memory_info.total, 1),
                        "gpu_power_watts": power_watts,
                    }
                )
            except Exception as exc:
                stats["gpu_nvml_stats_error"] = str(exc)
        elif nvml_error:
            stats["gpu_nvml_init_error"] = nvml_error
        return stats

    def update_peak_gpu_stats(stats: dict[str, Any]) -> None:
        for source_key, target_key in PEAK_GPU_STAT_KEYS.items():
            value = stats.get(source_key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                peak_gpu_stats[target_key] = max(peak_gpu_stats.get(target_key, 0.0), numeric_value)

    config = {
        "run_id": run_id,
        "track": track,
        "track_name": track_info["name"],
        "record_description": record_description,
        "record_contributors": record_contributors,
        "minutes": minutes,
        "seq_len": seq_len,
        "micro_batch_size": micro_batch_size,
        "requested_micro_batch_size": requested_micro_batch_size,
        "grad_accum": grad_accum,
        "requested_grad_accum": requested_grad_accum,
        "effective_tokens_per_step": seq_len * micro_batch_size * grad_accum,
        "target_effective_tokens_per_step": DEFAULT_EFFECTIVE_TOKENS_PER_STEP,
        "gpu_name": gpu_name,
        "gpu_total_memory_gib": gpu_props.total_memory / bytes_per_gib,
        "lr": lr,
        "requested_lr": requested_lr,
        "weight_decay": weight_decay,
        "requested_weight_decay": requested_weight_decay,
        "warmup_steps": warmup_steps,
        "eval_blocks": eval_blocks,
        "eval_micro_batch_size": eval_micro_batch_size,
        "requested_eval_micro_batch_size": requested_eval_micro_batch_size,
        "seed": seed,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "requested_dataset_id": requested_dataset_id,
        "requested_dataset_config": requested_dataset_config,
        "requested_dataset_revision": requested_dataset_revision,
        "data_mode": data_mode,
        "loss_mask": "assistant_only" if data_mode == "sft" else "all_tokens",
        "sequence_packing": DEFAULT_SEQUENCE_PACKING,
        "packing_strategy": DEFAULT_PACKING_STRATEGY,
        "packed_block_size": seq_len,
        "padding_tokens_per_block": 0,
        "tuning_mode": "full",
        "optimizer_name": resolved_optimizer_name,
        "requested_optimizer_name": requested_optimizer_name,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_enabled": checkpointing_enabled,
        "gradient_checkpointing_fallback_used": gradient_checkpointing_fallback_used,
        "muon_lr_adjustment": muon_lr_adjustment,
        "muon_quant_block_size": muon_quant_block_size,
        "normuon_beta2": normuon_beta2,
        "normuon_eps": normuon_eps,
        "muon_lr": resolved_muon_lr,
        "adamw_tail_lr": resolved_adamw_tail_lr,
        "requested_muon_lr": requested_muon_lr,
        "requested_adamw_tail_lr": requested_adamw_tail_lr,
        "yaqa_mode": yaqa_mode,
        "yaqa_calib_sequences": yaqa_calib_sequences,
        "yaqa_min_scale": yaqa_min_scale,
        "yaqa_max_scale": yaqa_max_scale,
        "yaqa_eps": yaqa_eps,
        "lowpass": bool(lowpass),
        "lowpass_projector_kind": lowpass_projector_kind,
        "lowpass_keep": lowpass_keep,
        "lowpass_min_hidden_dim": lowpass_min_hidden_dim,
        "lowpass_max_hidden_dim": lowpass_max_hidden_dim,
        "eval_version": EVAL_VERSION,
        "pack_align": pack_align,
        "lr_schedule": lr_schedule,
        "lr_decay_fraction": lr_decay_fraction,
        "min_lr_ratio": min_lr_ratio,
        "attn_implementation": attn_implementation,
        "compile_model": compile_model,
        "compile_mode": compile_mode,
        "compile_warmup": compile_warmup,
        "save_final": save_final,
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_name": wandb_name,
        "wandb_group": wandb_group,
        "wandb_tags": wandb_tags_list,
        "wandb_mode": wandb_mode,
    }

    def write_config() -> None:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    write_config()

    wandb_run = None

    def wandb_log(record: dict[str, Any]) -> None:
        if wandb_run is None:
            return
        payload: dict[str, int | float | bool] = {}
        for key, value in record.items():
            if key == "time":
                continue
            if isinstance(value, bool):
                payload[key] = value
            elif isinstance(value, int):
                payload[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                payload[key] = value
        event = record.get("event")
        if event in {"baseline_eval", "final_eval"} and isinstance(record.get("eval_loss"), (int, float)):
            payload[f"{event}/eval_loss"] = float(record["eval_loss"])
        if payload:
            wandb_run.log(payload)

    def log_metric(record: dict[str, Any], include_gpu: bool = False) -> None:
        if include_gpu:
            gpu_stats = collect_gpu_stats()
            update_peak_gpu_stats(gpu_stats)
            record = {**record, **gpu_stats}
        record = {"time": time.time(), **record}
        with metrics_path.open("a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        print(json.dumps(record, default=_json_default), flush=True)
        wandb_log(record)

    def log_gpu(label: str) -> None:
        log_metric({"event": "gpu", "label": label}, include_gpu=True)

    if wandb_enabled:
        if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "W&B online mode requires WANDB_API_KEY in the local environment before "
                "running Modal, or use --wandb-mode offline"
            )
        import wandb

        os.environ.setdefault("WANDB_DIR", str(run_dir))
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity or None,
            name=wandb_name or record_description or run_id,
            group=wandb_group or f"track-{track}",
            tags=wandb_tags_list,
            config=config,
            mode=wandb_mode,
            dir=str(run_dir),
        )

    def dataset_stream(shuffle: bool = False, purpose: Literal["train", "eval"] = "train"):
        if data_mode == "sft" and dataset_id == DEFAULT_SFT_DATASET_ID:
            split = DEFAULT_SFT_EVAL_SPLIT if purpose == "eval" else DEFAULT_SFT_TRAIN_SPLIT
        else:
            split = "train"
        ds = load_dataset(
            dataset_id,
            dataset_config or None,
            split=split,
            streaming=True,
            revision=dataset_revision or None,
            cache_dir=str(HF_CACHE / "datasets"),
        )
        if shuffle:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        return ds

    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision or None,
        cache_dir=str(HF_CACHE / "hub"),
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{model_id} tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_piece(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    def tokenize_text(text: str) -> list[int]:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if ids:
            ids.append(tokenizer.eos_token_id)
        return ids

    def render_sft_row(row: dict[str, Any]) -> tuple[list[int], list[int]] | None:
        return _render_sft_row(row, tokenize_piece, seq_len)

    # When lowpass is on, pad each document to a multiple of lowpass_chunk_size
    # so that every Hadamard chunk is purely within one document. Pad labels are
    # -100 (ignored in loss); pad position_ids continue the document's monotonic
    # count, so doc-aware attention keeps them in the same attention block.
    pad_token_id = int(tokenizer.pad_token_id)

    def align_doc(row_ids: list[int], row_labels: list[int]) -> tuple[list[int], list[int]]:
        if pack_align <= 1 or not row_ids:
            return row_ids, row_labels
        remainder = len(row_ids) % pack_align
        if remainder == 0:
            return row_ids, row_labels
        pad_count = pack_align - remainder
        return (
            row_ids + [pad_token_id] * pad_count,
            row_labels + [-100] * pad_count,
        )

    def build_eval_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Path, int]:
        if data_mode == "sft":
            return build_sft_eval_cache()
        return build_cpt_eval_cache()

    def load_eval_payload(eval_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        payload = torch.load(eval_path, map_location="cpu")
        input_ids = payload["input_ids"]
        labels = payload.get("labels")
        if labels is None:
            labels = input_ids.clone()
        position_ids = payload.get("position_ids")
        if position_ids is None:
            # Legacy cache without doc-aware position_ids; fall back to monotonic.
            position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long).expand_as(input_ids).contiguous()
        supervised_tokens = int(payload.get("supervised_tokens", _supervised_token_count(labels)))
        return input_ids, labels, position_ids, int(payload["skip_docs"]), supervised_tokens

    def build_cpt_eval_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Path, int]:
        key_payload = {
            "model": model_id,
            "model_revision": model_revision,
            "dataset": dataset_id,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "seq_len": seq_len,
            "eval_blocks": eval_blocks,
            "sequence_packing": DEFAULT_SEQUENCE_PACKING,
            "packing_strategy": DEFAULT_PACKING_STRATEGY,
            "seed": seed,
            "kind": "all_token_cpt_packed_v3_docaware",
            "text_field": cpt_text_field,
            "pack_align": pack_align,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()[:20]
        eval_path = eval_dir / f"{key}.pt"
        if eval_path.exists():
            input_ids, labels, position_ids, skip_docs, supervised_tokens = load_eval_payload(eval_path)
            return input_ids, labels, position_ids, skip_docs, eval_path, supervised_tokens

        need_tokens = eval_blocks * seq_len
        token_buffer: list[int] = []
        label_buffer: list[int] = []
        position_buffer: list[int] = []
        skip_docs = 0
        for row in dataset_stream(shuffle=False):
            text = row.get(cpt_text_field)
            if not isinstance(text, str) or not text.strip():
                skip_docs += 1
                continue
            row_ids = tokenize_text(text)
            padded_ids, padded_labels = align_doc(list(row_ids), list(row_ids))
            token_buffer.extend(padded_ids)
            label_buffer.extend(padded_labels)
            position_buffer.extend(range(len(padded_ids)))
            skip_docs += 1
            if len(token_buffer) >= need_tokens:
                break

        if len(token_buffer) < need_tokens:
            raise RuntimeError(
                f"could only build {len(token_buffer)} eval tokens, needed {need_tokens}"
            )

        input_ids = torch.tensor(token_buffer[:need_tokens], dtype=torch.long).view(eval_blocks, seq_len)
        position_ids = torch.tensor(position_buffer[:need_tokens], dtype=torch.long).view(eval_blocks, seq_len)
        labels = torch.tensor(label_buffer[:need_tokens], dtype=torch.long).view(eval_blocks, seq_len)
        supervised_tokens = int((labels != -100).sum())
        torch.save(
            {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "skip_docs": skip_docs,
                "supervised_tokens": supervised_tokens,
                "key_payload": key_payload,
            },
            eval_path,
        )
        return input_ids, labels, position_ids, skip_docs, eval_path, supervised_tokens

    def build_sft_eval_cache() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Path, int]:
        key_payload = {
            "model": model_id,
            "model_revision": model_revision,
            "dataset": dataset_id,
            "dataset_config": dataset_config,
            "dataset_revision": dataset_revision,
            "seq_len": seq_len,
            "eval_blocks": eval_blocks,
            "eval_split": DEFAULT_SFT_EVAL_SPLIT if dataset_id == DEFAULT_SFT_DATASET_ID else "train",
            "sequence_packing": DEFAULT_SEQUENCE_PACKING,
            "packing_strategy": DEFAULT_PACKING_STRATEGY,
            "kind": "chatml_assistant_only_sft_packed_v3_docaware",
            "pack_align": pack_align,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()[:20]
        eval_path = eval_dir / f"{key}.pt"
        if eval_path.exists():
            input_ids, labels, position_ids, skip_docs, supervised_tokens = load_eval_payload(eval_path)
            return input_ids, labels, position_ids, skip_docs, eval_path, supervised_tokens

        token_buffer: list[int] = []
        label_buffer: list[int] = []
        position_buffer: list[int] = []
        input_blocks: list[torch.Tensor] = []
        label_blocks: list[torch.Tensor] = []
        position_blocks: list[torch.Tensor] = []
        skip_docs = 0

        def drain_blocks() -> None:
            while len(token_buffer) >= seq_len and len(input_blocks) < eval_blocks:
                block_ids = token_buffer[:seq_len]
                block_labels = label_buffer[:seq_len]
                block_positions = position_buffer[:seq_len]
                del token_buffer[:seq_len]
                del label_buffer[:seq_len]
                del position_buffer[:seq_len]
                if _supervised_token_count(block_labels) == 0:
                    continue
                input_blocks.append(torch.tensor(block_ids, dtype=torch.long))
                label_blocks.append(torch.tensor(block_labels, dtype=torch.long))
                position_blocks.append(torch.tensor(block_positions, dtype=torch.long))

        for row in dataset_stream(shuffle=False, purpose="eval"):
            skip_docs += 1
            rendered = render_sft_row(row)
            if rendered is None:
                continue
            row_ids, row_labels = rendered
            padded_ids, padded_labels = align_doc(list(row_ids), list(row_labels))
            token_buffer.extend(padded_ids)
            label_buffer.extend(padded_labels)
            position_buffer.extend(range(len(padded_ids)))
            drain_blocks()
            if len(input_blocks) >= eval_blocks:
                break

        if len(input_blocks) < eval_blocks:
            raise RuntimeError(
                f"could only build {len(input_blocks)} SFT eval blocks, needed {eval_blocks}"
            )

        input_ids = torch.stack(input_blocks)
        labels = torch.stack(label_blocks)
        position_ids = torch.stack(position_blocks)
        supervised_tokens = _supervised_token_count(labels)
        torch.save(
            {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
                "skip_docs": 0 if dataset_id == DEFAULT_SFT_DATASET_ID else skip_docs,
                "eval_docs": skip_docs,
                "supervised_tokens": supervised_tokens,
                "key_payload": key_payload,
            },
            eval_path,
        )
        train_skip = 0 if dataset_id == DEFAULT_SFT_DATASET_ID else skip_docs
        return input_ids, labels, position_ids, train_skip, eval_path, supervised_tokens

    eval_input_ids, eval_labels, eval_position_ids, train_skip_docs, eval_path, eval_supervised_tokens = build_eval_cache()
    config["eval_supervised_tokens"] = eval_supervised_tokens
    write_config()
    print(
        f"fixed eval cache: {eval_path} skip_docs={train_skip_docs} "
        f"supervised_tokens={eval_supervised_tokens}",
        flush=True,
    )

    def disable_model_cache(current_model: torch.nn.Module) -> None:
        if hasattr(current_model, "config"):
            current_model.config.use_cache = False
            if hasattr(current_model.config, "text_config"):
                current_model.config.text_config.use_cache = False
        if hasattr(current_model, "model") and hasattr(current_model.model, "language_model"):
            current_model.model.language_model.config.use_cache = False

    def set_visual_eval(current_model: torch.nn.Module) -> None:
        root_model = getattr(current_model, "_orig_mod", current_model)
        for name, module in root_model.named_modules():
            if name == "visual" or name.endswith(".visual"):
                module.eval()

    def freeze_visual(current_model: torch.nn.Module) -> None:
        for name, module in current_model.named_modules():
            if name == "visual" or name.endswith(".visual"):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
                module.eval()

    def set_gradient_checkpointing(current_model: torch.nn.Module, enabled: bool) -> None:
        if enabled:
            if hasattr(current_model, "gradient_checkpointing_enable"):
                # use_reentrant=False is required for `saved_tensors_hooks`
                # installed by the lowpass context to see saves that happen
                # inside the per-block checkpoint recompute. Reentrant mode
                # runs the recompute in a separate autograd context that
                # bypasses our outer hooks, so the MLP intermediate etc.
                # never reach pack/unpack.
                try:
                    current_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    current_model.gradient_checkpointing_enable()
            return
        if hasattr(current_model, "gradient_checkpointing_disable"):
            current_model.gradient_checkpointing_disable()

    def train_batches(batch_size: int | None = None):
        active_batch_size = micro_batch_size if batch_size is None else batch_size
        ds = dataset_stream(shuffle=False).skip(train_skip_docs).shuffle(seed=seed, buffer_size=10_000)
        token_buffer: list[int] = []
        label_buffer: list[int] = []
        position_buffer: list[int] = []
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def make_batch() -> dict[str, torch.Tensor]:
            input_batch = torch.stack([item[0] for item in batch])
            label_batch = torch.stack([item[1] for item in batch])
            position_batch = torch.stack([item[2] for item in batch])
            batch.clear()
            return {
                "input_ids": input_batch,
                "labels": label_batch,
                "position_ids": position_batch,
            }

        while True:
            for row in ds:
                if data_mode == "sft":
                    rendered = render_sft_row(row)
                    if rendered is None:
                        continue
                    row_ids, row_labels = rendered
                else:
                    text = row.get(cpt_text_field)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    row_ids = tokenize_text(text)
                    row_labels = row_ids
                padded_ids, padded_labels = align_doc(list(row_ids), list(row_labels))
                token_buffer.extend(padded_ids)
                label_buffer.extend(padded_labels)
                position_buffer.extend(range(len(padded_ids)))

                while len(token_buffer) >= seq_len:
                    block_ids = token_buffer[:seq_len]
                    block_labels = label_buffer[:seq_len]
                    block_positions = position_buffer[:seq_len]
                    del token_buffer[:seq_len]
                    del label_buffer[:seq_len]
                    del position_buffer[:seq_len]
                    if data_mode == "sft" and _supervised_token_count(block_labels) == 0:
                        continue
                    batch.append(
                        (
                            torch.tensor(block_ids, dtype=torch.long),
                            torch.tensor(block_labels, dtype=torch.long),
                            torch.tensor(block_positions, dtype=torch.long),
                        )
                    )
                    if len(batch) == active_batch_size:
                        yield make_batch()
            ds = dataset_stream(shuffle=False).skip(train_skip_docs).shuffle(seed=seed, buffer_size=10_000)

    device = torch.device("cuda")

    print("loading model", flush=True)
    model = ModelClass.from_pretrained(
        model_id,
        revision=model_revision or None,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        cache_dir=str(HF_CACHE / "hub"),
        low_cpu_mem_usage=True,
    )
    disable_model_cache(model)
    freeze_visual(model)

    set_gradient_checkpointing(model, checkpointing_enabled)

    model.to(device)
    uncompiled_model = model
    torch.cuda.synchronize()
    log_gpu("after_model_to_cuda")

    if lowpass:
        from lowpass import LowpassConfig, make_module_filter, replace_linear_with_lowpass

        lowpass_config = LowpassConfig(
            projector_kind=lowpass_projector_kind,
            keep=lowpass_keep,
            min_hidden_dim=lowpass_min_hidden_dim,
            max_hidden_dim=lowpass_max_hidden_dim,
        )
        filter_fn = make_module_filter(lowpass_target_filter)
        replaced = replace_linear_with_lowpass(model, lowpass_config, filter_fn)
        print(
            f"lowpass: replaced {len(replaced)} nn.Linear modules "
            f"(target={lowpass_target_filter}, keep={lowpass_keep}, "
            f"hidden_dim∈[{lowpass_min_hidden_dim}, "
            f"{lowpass_max_hidden_dim if lowpass_max_hidden_dim > 0 else '∞'}], "
            f"projector={lowpass_projector_kind})",
            flush=True,
        )

    named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in named_trainable_params]
    if not trainable_params:
        raise RuntimeError("no trainable parameters were found")
    trainable_visual_params = [n for n, p in named_trainable_params if "visual" in n]
    if trainable_visual_params:
        raise RuntimeError(f"visual parameters must remain frozen: {trainable_visual_params[:5]}")
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {trainable_count:,} / {total_count:,}", flush=True)

    class Muon(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr: float,
            momentum: float = 0.95,
            weight_decay: float = 0.0,
            ns_steps: int = 5,
            nesterov: bool = True,
            lr_adjustment: str = "original",
        ):
            if lr_adjustment not in MUON_LR_ADJUSTMENT_CHOICES:
                raise ValueError(f"unsupported Muon LR adjustment: {lr_adjustment}")
            super().__init__(
                params,
                dict(
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    ns_steps=ns_steps,
                    nesterov=nesterov,
                    lr_adjustment=lr_adjustment,
                ),
            )

        @staticmethod
        def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int) -> torch.Tensor:
            assert grad.ndim == 2
            a, b, c = 3.4445, -4.7750, 2.0315
            x = grad.bfloat16()
            transposed = x.size(0) > x.size(1)
            if transposed:
                x = x.T
            x = x / (x.norm() + 1.0e-7)
            for _ in range(steps):
                xx_t = x @ x.T
                x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
            if transposed:
                x = x.T
            return x.to(grad.dtype)

        @staticmethod
        def adjust_lr(lr: float, lr_adjustment: str, shape: torch.Size) -> float:
            rows, cols = shape[:2]
            if lr_adjustment == "match_rms_adamw":
                return lr * 0.2 * math.sqrt(max(rows, cols))
            return lr * math.sqrt(max(1.0, rows / cols))

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]
                lr_adjustment = group["lr_adjustment"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim != 2:
                        raise RuntimeError("Muon only supports 2D matrix parameters")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1.0 - momentum)
                    update = grad.lerp(buf, momentum) if nesterov else buf
                    update = self.zeropower_via_newtonschulz5(update, ns_steps)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    adjusted_lr = self.adjust_lr(lr, lr_adjustment, p.shape)
                    p.add_(update, alpha=-adjusted_lr)
            return loss

    class QuantizedMuon(Muon):
        @staticmethod
        def quantize_blockwise(tensor: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
            flat = tensor.detach().to(torch.float32).reshape(-1)
            numel = flat.numel()
            num_blocks = (numel + block_size - 1) // block_size
            padded = torch.zeros(num_blocks * block_size, device=flat.device, dtype=torch.float32)
            padded[:numel] = flat
            blocks = padded.view(num_blocks, block_size)
            scales = blocks.abs().amax(dim=1).clamp_min(1.0e-12) / 127.0
            quantized = torch.round(blocks / scales[:, None]).clamp_(-127, 127).to(torch.int8)
            return quantized.reshape(-1)[:numel].contiguous().view(tensor.shape), scales

        @staticmethod
        def dequantize_blockwise(
            quantized: torch.Tensor,
            scales: torch.Tensor,
            block_size: int,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            flat_q = quantized.reshape(-1).to(torch.float32)
            scale_per_element = scales.repeat_interleave(block_size)[: flat_q.numel()]
            return (flat_q * scale_per_element).view(quantized.shape).to(dtype)

        def __init__(
            self,
            params,
            lr: float,
            momentum: float = 0.95,
            weight_decay: float = 0.0,
            ns_steps: int = 5,
            nesterov: bool = True,
            lr_adjustment: str = "original",
            block_size: int = DEFAULT_MUON_QUANT_BLOCK_SIZE,
        ):
            super().__init__(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                ns_steps=ns_steps,
                nesterov=nesterov,
                lr_adjustment=lr_adjustment,
            )
            for group in self.param_groups:
                group["block_size"] = block_size

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]
                lr_adjustment = group["lr_adjustment"]
                block_size = group["block_size"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim != 2:
                        raise RuntimeError("Muon only supports 2D matrix parameters")
                    state = self.state[p]
                    if "momentum_q" in state:
                        buf = self.dequantize_blockwise(
                            state["momentum_q"],
                            state["momentum_scale"],
                            block_size,
                            grad.dtype,
                        )
                    else:
                        buf = torch.zeros_like(grad)
                    buf.lerp_(grad, 1.0 - momentum)
                    update = grad.lerp(buf, momentum) if nesterov else buf
                    update = self.zeropower_via_newtonschulz5(update, ns_steps)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    adjusted_lr = self.adjust_lr(lr, lr_adjustment, p.shape)
                    p.add_(update, alpha=-adjusted_lr)
                    state["momentum_q"], state["momentum_scale"] = self.quantize_blockwise(
                        buf,
                        block_size,
                    )
                    state.pop("momentum_buffer", None)
            return loss

    class NorMuon(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr: float,
            beta1: float = 0.95,
            beta2: float = DEFAULT_NORMUON_BETA2,
            eps: float = DEFAULT_NORMUON_EPS,
            weight_decay: float = 0.0,
            ns_steps: int = 5,
        ):
            super().__init__(
                params,
                dict(
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    ns_steps=ns_steps,
                ),
            )

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group["lr"]
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim != 2:
                        raise RuntimeError("NorMuon only supports 2D matrix parameters")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]
                    buf.lerp_(grad, 1.0 - beta1)
                    update = Muon.zeropower_via_newtonschulz5(buf, ns_steps)
                    row_stat = update.to(torch.float32).square().mean(dim=1)
                    if "row_second_moment" not in state:
                        state["row_second_moment"] = torch.zeros(
                            update.shape[0],
                            device=update.device,
                            dtype=torch.float32,
                        )
                    second = state["row_second_moment"]
                    second.lerp_(row_stat, 1.0 - beta2)
                    denom = torch.sqrt(second + eps).to(update.dtype).unsqueeze(1)
                    normalized_update = update / denom
                    update_scale = (
                        0.2
                        * lr
                        * math.sqrt(p.numel())
                        / normalized_update.norm().clamp_min(1.0e-12)
                    )
                    normalized_update.mul_(update_scale)
                    if weight_decay:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(normalized_update, alpha=-1.0)
            return loss

    class YAQADiagAdamW(torch.optim.Optimizer):
        """AdamW with static diagonal Kronecker preconditioning from YAQA Sketch B."""

        def __init__(
            self,
            named_params,
            scales: dict[str, torch.Tensor],
            lr: float = 2.0e-5,
            betas: tuple[float, float] = (0.9, 0.95),
            eps: float = 1.0e-8,
            weight_decay: float = 0.1,
        ):
            params = [p for _n, p in named_params if p.requires_grad]
            super().__init__(
                params,
                dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
            )
            self.scales: dict[int, torch.Tensor] = {}
            for n, p in named_params:
                if p.requires_grad and n in scales:
                    self.scales[id(p)] = scales[n]

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                lr = group["lr"]
                wd = group["weight_decay"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("YAQADiagAdamW does not support sparse gradients")

                    scale = self.scales.get(id(p))
                    if scale is not None:
                        if scale.device != grad.device:
                            scale = scale.to(grad.device)
                        g = grad * scale
                    else:
                        g = grad

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    step = state["step"]

                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                    if wd != 0:
                        p.mul_(1.0 - lr * wd)

                    p.addcdiv_(exp_avg, denom, value=-step_size)
            return loss

    yaqa_scales: dict[str, torch.Tensor] = {}

    def make_optimizer():
        if resolved_optimizer_name == "adamw8bit":
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(
                trainable_params,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=weight_decay,
            )
        if resolved_optimizer_name in {"muon", "muon8", "normuon"}:
            matrix_params: list[torch.nn.Parameter] = []
            adamw_params: list[torch.nn.Parameter] = []
            for name, parameter in named_trainable_params:
                clean_name = name.removeprefix("_orig_mod.")
                is_embed_or_head = any(part in clean_name for part in ("embed", "lm_head"))
                if parameter.ndim == 2 and not is_embed_or_head:
                    matrix_params.append(parameter)
                else:
                    adamw_params.append(parameter)

            optimizers: list[torch.optim.Optimizer] = []
            if matrix_params:
                if resolved_optimizer_name == "muon8":
                    optimizers.append(
                        QuantizedMuon(
                            matrix_params,
                            lr=resolved_muon_lr,
                            momentum=0.95,
                            weight_decay=weight_decay,
                            lr_adjustment=muon_lr_adjustment,
                            block_size=muon_quant_block_size,
                        )
                    )
                elif resolved_optimizer_name == "normuon":
                    optimizers.append(
                        NorMuon(
                            matrix_params,
                            lr=resolved_muon_lr,
                            beta1=0.95,
                            beta2=normuon_beta2,
                            eps=normuon_eps,
                            weight_decay=weight_decay,
                        )
                    )
                else:
                    optimizers.append(
                        Muon(
                            matrix_params,
                            lr=resolved_muon_lr,
                            momentum=0.95,
                            weight_decay=weight_decay,
                            lr_adjustment=muon_lr_adjustment,
                        )
                    )
            adamw_name = "none"
            if adamw_params:
                try:
                    import bitsandbytes as bnb

                    adamw = bnb.optim.AdamW8bit(
                        adamw_params,
                        lr=resolved_adamw_tail_lr,
                        betas=(0.9, 0.95),
                        eps=1.0e-8,
                        weight_decay=weight_decay,
                    )
                    adamw_name = "adamw8bit"
                except Exception as exc:
                    print(
                        f"warning: bitsandbytes AdamW8bit unavailable for {resolved_optimizer_name} tail: {exc}",
                        flush=True,
                    )
                    adamw = torch.optim.AdamW(
                        adamw_params,
                        lr=resolved_adamw_tail_lr,
                        betas=(0.9, 0.95),
                        eps=1.0e-8,
                        weight_decay=weight_decay,
                    )
                    adamw_name = "adamw"
                optimizers.append(adamw)
            if resolved_optimizer_name == "normuon":
                details = f"beta1=0.95 beta2={normuon_beta2} eps={normuon_eps}"
            else:
                details = f"lr_adjustment={muon_lr_adjustment}"
            if resolved_optimizer_name == "muon8":
                details += f" quant=linear8 block_size={muon_quant_block_size}"
            print(
                f"optimizer {resolved_optimizer_name}: {len(matrix_params)} matrix tensors @ lr={resolved_muon_lr:g}, "
                f"{adamw_name}: {len(adamw_params)} non-matrix tensors @ lr={resolved_adamw_tail_lr:g}, {details}",
                flush=True,
            )
            return optimizers
        if resolved_optimizer_name == "yaqadamw":
            return YAQADiagAdamW(
                named_trainable_params,
                scales=yaqa_scales,
                lr=lr,
                betas=(0.9, 0.95),
                eps=1.0e-8,
                weight_decay=weight_decay,
            )
        kwargs: dict[str, Any] = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "eps": 1.0e-8,
            "weight_decay": weight_decay,
        }
        try:
            return torch.optim.AdamW(trainable_params, fused=True, **kwargs)
        except TypeError:
            return torch.optim.AdamW(trainable_params, **kwargs)

    optimizer = make_optimizer()

    def iter_optimizer_groups():
        if isinstance(optimizer, list):
            for opt in optimizer:
                for group in opt.param_groups:
                    yield group
            return
        for group in optimizer.param_groups:
            yield group

    def capture_optimizer_base_lrs() -> None:
        for group in iter_optimizer_groups():
            group["base_lr"] = float(group["lr"])

    def set_optimizer_lr_multiplier(multiplier: float) -> None:
        for group in iter_optimizer_groups():
            group["lr"] = float(group.get("base_lr", lr)) * multiplier

    capture_optimizer_base_lrs()
    torch.cuda.synchronize()
    log_gpu("after_optimizer_init")

    def optimizer_zero_grad() -> None:
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.zero_grad(set_to_none=True)
            return
        optimizer.zero_grad(set_to_none=True)

    def optimizer_step() -> None:
        if isinstance(optimizer, list):
            for opt in optimizer:
                opt.step()
            return
        optimizer.step()

    @torch.no_grad()
    def evaluate(label: str) -> float:
        model.eval()
        weighted_loss_sum = 0.0
        total_supervised = 0
        for start in range(0, eval_blocks, eval_micro_batch_size):
            batch = eval_input_ids[start : start + eval_micro_batch_size].to(device, non_blocking=True)
            labels = eval_labels[start : start + eval_micro_batch_size].to(device, non_blocking=True)
            position_ids = eval_position_ids[start : start + eval_micro_batch_size].to(device, non_blocking=True)
            supervised = _supervised_token_count(labels.detach().cpu())
            if supervised == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch,
                    attention_mask=None,
                    position_ids=position_ids,
                    labels=labels,
                    use_cache=False,
                )
            weighted_loss_sum += float(output.loss.detach().cpu()) * supervised
            total_supervised += supervised
        if total_supervised == 0:
            raise RuntimeError("evaluation batch had no supervised tokens")
        loss = weighted_loss_sum / total_supervised
        log_metric(
            {"event": label, "eval_loss": loss, "eval_supervised_tokens": total_supervised},
            include_gpu=True,
        )
        model.train()
        set_visual_eval(model)
        return loss

    batch_iter = train_batches()

    def is_cuda_oom(exc: BaseException) -> bool:
        return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def refresh_trainable_params() -> None:
        nonlocal named_trainable_params, trainable_params, trainable_count, total_count
        named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        trainable_params = [p for _, p in named_trainable_params]
        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())

    def run_train_warmup() -> None:
        print("running train/compile warmup", flush=True)
        model.train()
        optimizer_zero_grad()
        warmup_losses: list[torch.Tensor] = []
        warmup_supervised_counts: list[int] = []
        for _ in range(grad_accum):
            warmup_batch = next(batch_iter)
            input_ids = warmup_batch["input_ids"].to(device, non_blocking=True)
            labels = warmup_batch["labels"].to(device, non_blocking=True)
            position_ids = warmup_batch["position_ids"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                warmup_loss = model(
                    input_ids=input_ids,
                    attention_mask=None,
                    position_ids=position_ids,
                    labels=labels,
                    use_cache=False,
                ).loss
            warmup_losses.append(warmup_loss)
            warmup_supervised_counts.append(_supervised_token_count(labels.detach().cpu()))
        warmup_supervised_total = sum(warmup_supervised_counts)
        if warmup_supervised_total <= 0:
            raise RuntimeError("warmup batch had no supervised tokens")
        for warmup_loss, supervised in zip(warmup_losses, warmup_supervised_counts):
            (warmup_loss * (supervised / warmup_supervised_total)).backward()
        optimizer_zero_grad()
        torch.cuda.synchronize()

    baseline_loss = evaluate("baseline_eval")
    compile_warmup_start = time.monotonic()
    log_gpu("compile_warmup_start")

    if compile_model:
        print(f"compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(model, dynamic=False, mode=compile_mode)

    if compile_warmup:
        try:
            run_train_warmup()
        except RuntimeError as exc:
            if not (
                gradient_checkpointing == "auto"
                and not checkpointing_enabled
                and is_cuda_oom(exc)
            ):
                raise
            print(
                "warmup OOM without gradient checkpointing; "
                "retrying warmup with checkpointing enabled",
                flush=True,
            )
            gradient_checkpointing_fallback_used = True
            checkpointing_enabled = True
            config["gradient_checkpointing_enabled"] = checkpointing_enabled
            config["gradient_checkpointing_fallback_used"] = gradient_checkpointing_fallback_used
            try:
                optimizer_zero_grad()
            except Exception:
                pass
            model = uncompiled_model
            set_gradient_checkpointing(model, True)
            torch.cuda.empty_cache()
            if compile_model:
                print(f"recompiling model with torch.compile(mode={compile_mode!r})", flush=True)
                model = torch.compile(model, dynamic=False, mode=compile_mode)
            refresh_trainable_params()
            optimizer = make_optimizer()
            capture_optimizer_base_lrs()
            torch.cuda.synchronize()
            log_gpu("after_optimizer_reinit")
            batch_iter = train_batches()
            write_config()
            run_train_warmup()

    # ------------------------------------------------------------------
    # YAQA diagonal calibration (untimed — before budget_start)
    # ------------------------------------------------------------------
    if resolved_optimizer_name == "yaqadamw" and yaqa_mode == "static":
        print(f"running YAQA diagonal calibration ({yaqa_calib_sequences} sequences)", flush=True)
        model.train()
        _yaqa_d_O: dict[str, torch.Tensor] = {}
        _yaqa_d_I: dict[str, torch.Tensor] = {}
        _yaqa_counts: dict[str, int] = {}
        for _ in range(yaqa_calib_sequences):
            optimizer_zero_grad()
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            position_ids = batch["position_ids"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                calib_loss = model(
                    input_ids=input_ids,
                    attention_mask=None,
                    position_ids=position_ids,
                    labels=labels,
                    use_cache=False,
                ).loss
            calib_loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and param.ndim == 2:
                    g = param.grad.float()
                    if name not in _yaqa_d_O:
                        _yaqa_d_O[name] = torch.zeros(g.shape[0], device=g.device, dtype=torch.float32)
                        _yaqa_d_I[name] = torch.zeros(g.shape[1], device=g.device, dtype=torch.float32)
                        _yaqa_counts[name] = 0
                    _yaqa_d_O[name].add_(g.pow(2).sum(dim=1))
                    _yaqa_d_I[name].add_(g.pow(2).sum(dim=0))
                    _yaqa_counts[name] += 1
            optimizer_zero_grad()
        for name in _yaqa_d_O:
            d_O = _yaqa_d_O[name] / _yaqa_counts[name]
            d_I = _yaqa_d_I[name] / _yaqa_counts[name]
            scale = 1.0 / torch.sqrt(d_O.unsqueeze(1) * d_I.unsqueeze(0) + yaqa_eps)
            scale.clamp_(min=yaqa_min_scale, max=yaqa_max_scale)
            yaqa_scales[name] = scale.to(torch.bfloat16)
        print(f"YAQA calibration complete: {len(yaqa_scales)} layers", flush=True)
        torch.cuda.synchronize()
        log_gpu("after_yaqa_calibration")
        # Rebuild optimizer now that scales are collected.
        optimizer = make_optimizer()
        capture_optimizer_base_lrs()

    optimizer_zero_grad()
    torch.cuda.reset_peak_memory_stats(gpu_index)
    peak_gpu_stats.clear()
    log_gpu("budget_start")
    log_gpu("before_train_loop")
    budget_start = time.monotonic()
    train_deadline = budget_start + minutes * 60.0
    train_loop_start = budget_start

    full_decay_start_time: float | None = None

    def compute_lr_multiplier(step_value: int, now: float) -> float:
        nonlocal full_decay_start_time
        if warmup_steps > 0 and step_value <= warmup_steps:
            return step_value / warmup_steps
        if lr_schedule in {"linear", "cosine"}:
            if full_decay_start_time is None:
                full_decay_start_time = now
            decay_seconds = max(train_deadline - full_decay_start_time, 1.0e-9)
            decay_progress = min(1.0, max(0.0, (now - full_decay_start_time) / decay_seconds))
            if lr_schedule == "linear":
                return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - decay_progress)
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        if lr_schedule == "wsd":
            decay_window_seconds = minutes * 60.0 * lr_decay_fraction
            if decay_window_seconds > 0.0:
                decay_start = train_deadline - decay_window_seconds
                if now >= decay_start:
                    decay_progress = min(1.0, max(0.0, (now - decay_start) / decay_window_seconds))
                    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        return 1.0

    step = 0
    tokens = 0
    supervised_tokens_seen = 0
    last_loss = math.nan
    while time.monotonic() < train_deadline:
        optimizer_zero_grad()
        accum_losses: list[float] = []
        accum_supervised_tokens = 0
        accum_loss_tensors: list[tuple[torch.Tensor, int]] = []
        for _ in range(grad_accum):
            batch = next(batch_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            position_ids = batch["position_ids"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=input_ids,
                    attention_mask=None,
                    position_ids=position_ids,
                    labels=labels,
                    use_cache=False,
                )
            accum_losses.append(float(output.loss.detach().cpu()))
            tokens += int(input_ids.numel())
            batch_supervised = _supervised_token_count(labels.detach().cpu())
            if batch_supervised <= 0:
                raise RuntimeError("training batch had no supervised tokens")
            accum_supervised_tokens += batch_supervised
            supervised_tokens_seen += batch_supervised
            accum_loss_tensors.append((output.loss, batch_supervised))

        for loss_tensor, batch_supervised in accum_loss_tensors:
            loss = loss_tensor * (batch_supervised / accum_supervised_tokens)
            loss.backward()

        step += 1
        lr_now = time.monotonic()
        lr_multiplier = compute_lr_multiplier(step, lr_now)
        step_lr = lr * lr_multiplier
        set_optimizer_lr_multiplier(lr_multiplier)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer_step()
        last_loss = float(np.mean(accum_losses))

        if log_every > 0 and (step == 1 or step % log_every == 0):
            now = time.monotonic()
            elapsed_budget_seconds = now - budget_start
            elapsed_train_loop_seconds = now - train_loop_start
            log_metric(
                {
                    "event": "train",
                    "step": step,
                    "train_loss": last_loss,
                    "lr": step_lr,
                    "lr_multiplier": lr_multiplier,
                    "tokens": tokens,
                    "supervised_tokens": supervised_tokens_seen,
                    "step_supervised_tokens": accum_supervised_tokens,
                    "elapsed_compile_warmup_seconds": budget_start - compile_warmup_start,
                    "elapsed_budget_seconds": elapsed_budget_seconds,
                    "elapsed_train_loop_seconds": elapsed_train_loop_seconds,
                    "tokens_per_second": tokens / max(elapsed_budget_seconds, 1.0e-9),
                    "supervised_tokens_per_second": supervised_tokens_seen
                    / max(elapsed_budget_seconds, 1.0e-9),
                    "train_loop_tokens_per_second": tokens / max(elapsed_train_loop_seconds, 1.0e-9),
                    "train_loop_supervised_tokens_per_second": supervised_tokens_seen
                    / max(elapsed_train_loop_seconds, 1.0e-9),
                },
                include_gpu=True,
            )

    budget_end = time.monotonic()
    elapsed_budget_seconds = budget_end - budget_start
    elapsed_train_loop_seconds = budget_end - train_loop_start
    elapsed_compile_warmup_seconds = budget_start - compile_warmup_start
    # Keep post-budget evaluation from triggering a new compiled eval graph.
    model = uncompiled_model
    final_loss = evaluate("final_eval")
    summary = {
        **config,
        **peak_gpu_stats,
        "record_date": dt.date.today().isoformat(),
        "run_dir": str(run_dir),
        "eval_cache": str(eval_path),
        "train_skip_docs": train_skip_docs,
        "eval_supervised_tokens": eval_supervised_tokens,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "steps": step,
        "tokens": tokens,
        "supervised_tokens": supervised_tokens_seen,
        "elapsed_compile_warmup_seconds": elapsed_compile_warmup_seconds,
        "elapsed_budget_seconds": elapsed_budget_seconds,
        "elapsed_train_loop_seconds": elapsed_train_loop_seconds,
        "elapsed_train_seconds": elapsed_budget_seconds,
        "tokens_per_second": tokens / max(elapsed_budget_seconds, 1.0e-9),
        "supervised_tokens_per_second": supervised_tokens_seen / max(elapsed_budget_seconds, 1.0e-9),
        "train_loop_tokens_per_second": tokens / max(elapsed_train_loop_seconds, 1.0e-9),
        "train_loop_supervised_tokens_per_second": supervised_tokens_seen
        / max(elapsed_train_loop_seconds, 1.0e-9),
        "last_train_loss": last_loss,
        "baseline_eval_loss": baseline_loss,
        "final_eval_loss": final_loss,
        "eval_loss_drop": baseline_loss - final_loss,
    }

    if save_final:
        unwrapped_model = getattr(model, "_orig_mod", model)
        final_dir = run_dir / "final_model"
        unwrapped_model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        summary["final_model_dir"] = str(final_dir)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    log_metric({"event": "summary", **summary})
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()

    record_artifacts = None
    if record_description:
        src_path = Path(__file__).resolve()
        record_artifacts = {
            "main.py": src_path.read_text() if src_path.exists() else "",
            "config.json": json.dumps(config, indent=2, default=_json_default) + "\n",
            "summary.json": json.dumps(summary, indent=2, default=_json_default) + "\n",
            "record.txt": _format_record_text(summary),
            "metrics.jsonl": metrics_path.read_text() if metrics_path.exists() else "",
        }

    cache_volume.commit()
    if record_artifacts is not None:
        summary["_record_artifacts"] = record_artifacts
    return summary


@app.local_entrypoint()
def main(
    minutes: float = 0.0,
    seq_len: int = 4096,
    micro_batch_size: int = 0,
    grad_accum: int = 0,
    lr: float = 0.0,
    weight_decay: float = -1.0,
    warmup_steps: int = 20,
    eval_blocks: int = 64,
    eval_micro_batch_size: int = 0,
    seed: int = 1337,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    dataset_id: str = "",
    dataset_config: str = "",
    dataset_revision: str = "",
    cpt_text_field: str = DEFAULT_CPT_TEXT_FIELD,
    data_mode: str = "",
    optimizer_name: str = "auto",
    gradient_checkpointing: str = "auto",
    muon_lr_adjustment: str = "match_rms_adamw",
    muon_quant_block_size: int = DEFAULT_MUON_QUANT_BLOCK_SIZE,
    normuon_beta2: float = DEFAULT_NORMUON_BETA2,
    normuon_eps: float = DEFAULT_NORMUON_EPS,
    muon_lr: float = 0.0,
    adamw_tail_lr: float = 0.0,
    yaqa_mode: str = "static",
    yaqa_calib_sequences: int = 256,
    yaqa_min_scale: float = 0.1,
    yaqa_max_scale: float = 10.0,
    yaqa_eps: float = 1.0e-8,
    lowpass: bool = False,
    lowpass_projector_kind: str = "dct",
    lowpass_keep: int = DEFAULT_LOWPASS_KEEP,
    lowpass_target_filter: str = DEFAULT_LOWPASS_TARGET_FILTER,
    lowpass_min_hidden_dim: int = 8000,
    lowpass_max_hidden_dim: int = 16000,
    lr_schedule: str = "constant",
    lr_decay_fraction: float = 0.1,
    min_lr_ratio: float = 0.0,
    attn_implementation: str = "flex_attention",
    compile_model: bool = True,
    compile_mode: str = "max-autotune-no-cudagraphs",
    compile_warmup: bool = True,
    save_final: bool = False,
    log_every: int = 5,
    wandb_project: str = "modded-continued-training",
    wandb_entity: str = "umd-leans-well",
    wandb_name: str = "",
    wandb_group: str = "",
    wandb_tags: str = "",
    wandb_mode: str = "online",
    track: str = "1",
    record_description: str = "",
    record_contributors: str = "",
) -> None:
    """Run a modded-continued-training track on Modal.

    --track selects the competition track (1=30min, 2=5min, 3=2hr).
    All tracks default to packed CPT full fine-tuning on the canonical
    ConlangCrafter corpus. Pass --data-mode sft for the legacy UltraChat
    SFT path. When --minutes is 0 (default), the track's default budget
    is used. Set --record-description to save a competition record.
    """

    if track not in TRACKS:
        raise ValueError(f"--track must be one of: {', '.join(TRACKS.keys())}")
    if not data_mode:
        data_mode = "cpt"

    if minutes == 0.0:
        minutes = TRACKS[track]["default_minutes"]

    summary = run_track1.remote(
        minutes=minutes,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        grad_accum=grad_accum,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        eval_blocks=eval_blocks,
        eval_micro_batch_size=eval_micro_batch_size,
        seed=seed,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_revision=dataset_revision,
        cpt_text_field=cpt_text_field,
        data_mode=data_mode,  # type: ignore[arg-type]
        optimizer_name=optimizer_name,
        gradient_checkpointing=gradient_checkpointing,  # type: ignore[arg-type]
        muon_lr_adjustment=muon_lr_adjustment,  # type: ignore[arg-type]
        muon_quant_block_size=muon_quant_block_size,
        normuon_beta2=normuon_beta2,
        normuon_eps=normuon_eps,
        muon_lr=muon_lr,
        adamw_tail_lr=adamw_tail_lr,
        yaqa_mode=yaqa_mode,  # type: ignore[arg-type]
        yaqa_calib_sequences=yaqa_calib_sequences,
        yaqa_min_scale=yaqa_min_scale,
        yaqa_max_scale=yaqa_max_scale,
        yaqa_eps=yaqa_eps,
        lowpass=lowpass,
        lowpass_projector_kind=lowpass_projector_kind,  # type: ignore[arg-type]
        lowpass_keep=lowpass_keep,
        lowpass_target_filter=lowpass_target_filter,  # type: ignore[arg-type]
        lowpass_min_hidden_dim=lowpass_min_hidden_dim,
        lowpass_max_hidden_dim=lowpass_max_hidden_dim,
        lr_schedule=lr_schedule,  # type: ignore[arg-type]
        lr_decay_fraction=lr_decay_fraction,
        min_lr_ratio=min_lr_ratio,
        attn_implementation=attn_implementation,
        compile_model=compile_model,
        compile_mode=compile_mode,
        compile_warmup=compile_warmup,
        save_final=save_final,
        log_every=log_every,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
        wandb_mode=wandb_mode,  # type: ignore[arg-type]
        track=track,
        record_description=record_description,
        record_contributors=record_contributors,
    )
    record_artifacts = summary.pop("_record_artifacts", None)
    if record_description and isinstance(record_artifacts, dict):
        record_dir = _write_local_record(summary, record_artifacts)
        print(f"record saved to {record_dir}")
    print(json.dumps(summary, indent=2, default=_json_default))
