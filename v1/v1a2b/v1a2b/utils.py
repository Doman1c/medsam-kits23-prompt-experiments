from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def parse_int_list(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def parse_float_list(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    return [float(x) for x in value]


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_if_exists(source: str | Path, destination: str | Path) -> bool:
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return True


def case_sort_key(case_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in case_id if ch.isdigit())
    return (int(digits) if digits else 10**9, case_id)
