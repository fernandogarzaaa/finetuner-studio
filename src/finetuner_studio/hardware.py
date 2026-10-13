"""GPU detection and precision selection.

T4 (compute capability 7.5) has no bfloat16, so "auto" resolves to float16
there and bfloat16 on Ampere and newer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GpuInfo:
    name: str
    total_gb: float
    compute_capability: str
    bf16: bool


def gpus() -> list[GpuInfo]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        major, minor = p.major, p.minor
        out.append(
            GpuInfo(
                name=p.name,
                total_gb=round(p.total_memory / 1024**3, 1),
                compute_capability=f"{major}.{minor}",
                bf16=major >= 8,
            )
        )
    return out


def summary() -> dict[str, Any]:
    found = gpus()
    return {"cuda": bool(found), "gpus": [asdict(g) for g in found]}


def resolve_dtype(requested: str):
    import torch

    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    found = gpus()
    if not found:
        return torch.float32  # CPU: half precision training is not supported
    if all(g.bf16 for g in found):
        return torch.bfloat16
    return torch.float16
