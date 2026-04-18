import multiprocessing
import os
from threading import Lock
from typing import Any, Dict, List, Optional

_MANAGER: Optional[multiprocessing.Manager] = None
_LABEL_STORE = None
_LABEL_LOCK = Lock()
_SOLVER_SPECS: Optional[List[Dict[str, str]]] = None
_DEFAULT_INDEX = 0


def _ensure_label_store():
    global _MANAGER, _LABEL_STORE
    if _LABEL_STORE is None:
        _MANAGER = multiprocessing.Manager()
        _LABEL_STORE = _MANAGER.dict()
    return _LABEL_STORE


def _load_solver_specs() -> List[Dict[str, str]]:
    global _SOLVER_SPECS, _DEFAULT_INDEX
    if _SOLVER_SPECS is not None:
        return _SOLVER_SPECS

    raw = os.environ.get("REPHRASE_SOLVER_MODELS", "").strip()
    if not raw:
        _SOLVER_SPECS = []
        _DEFAULT_INDEX = 0
        return _SOLVER_SPECS

    root = os.environ.get("REPHRASE_SOLVER_ROOT") or os.environ.get("MODEL_DIR", "")
    specs: List[Dict[str, str]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if os.path.isabs(token):
            path = token
            name = os.path.basename(path.rstrip("/"))
        else:
            path = os.path.join(root, token)
            name = token
        specs.append({"name": name, "path": path})

    _SOLVER_SPECS = specs
    _DEFAULT_INDEX = len(specs) // 2 if specs else 0
    return _SOLVER_SPECS


def solver_pool_enabled() -> bool:
    return len(_load_solver_specs()) > 0


def ensure_solver_store_initialized():
    if solver_pool_enabled():
        _ensure_label_store()


def get_solver_specs() -> List[Dict[str, str]]:
    return list(_load_solver_specs())


def get_solver_count() -> int:
    return len(_load_solver_specs())


def get_default_solver_index() -> int:
    _load_solver_specs()
    return _DEFAULT_INDEX


def _normalize_label(label: int) -> int:
    count = get_solver_count()
    if count == 0:
        return 0
    label = int(label)
    if label < 0:
        return 0
    if label >= count:
        return count - 1
    return label


def _as_key(sample_id: Any) -> Optional[str]:
    if sample_id is None:
        return None
    return str(sample_id)


def solver_label_for_sample(sample_id: Any) -> Optional[int]:
    if not solver_pool_enabled():
        return None
    key = _as_key(sample_id)
    if key is None:
        return None
    store = _ensure_label_store()
    with _LABEL_LOCK:
        try:
            label = store[key]
        except KeyError:
            label = get_default_solver_index()
            store[key] = label
    return int(label)


def shift_solver_label(sample_id: Any, delta: int) -> Optional[int]:
    if not solver_pool_enabled():
        return None
    key = _as_key(sample_id)
    if key is None:
        return None
    store = _ensure_label_store()
    with _LABEL_LOCK:
        try:
            current = store[key]
        except KeyError:
            current = get_default_solver_index()
        updated = _normalize_label(int(current) + int(delta))
        store[key] = updated
    return updated


def set_solver_label(sample_id: Any, label: int) -> Optional[int]:
    if not solver_pool_enabled():
        return None
    key = _as_key(sample_id)
    if key is None:
        return None
    store = _ensure_label_store()
    normalized = _normalize_label(label)
    with _LABEL_LOCK:
        store[key] = normalized
    return normalized


def get_solver_label_counts() -> Dict[int, int]:
    """Get the count of samples assigned to each solver."""
    if not solver_pool_enabled():
        return {}
    store = _ensure_label_store()
    specs = _load_solver_specs()
    counts = {i: 0 for i in range(len(specs))}
    with _LABEL_LOCK:
        for label in store.values():
            label = int(label)
            if label in counts:
                counts[label] += 1
    return counts


def get_solver_label_summary() -> str:
    """Get a summary string of solver label distribution."""
    if not solver_pool_enabled():
        return "solver_pool disabled"
    specs = _load_solver_specs()
    counts = get_solver_label_counts()
    parts = []
    for i, spec in enumerate(specs):
        parts.append(f"{spec['name']}:{counts.get(i, 0)}")
    return ", ".join(parts)
