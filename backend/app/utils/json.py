from __future__ import annotations

from typing import Any

import numpy as np


def make_json_safe(value: Any) -> Any:
    """Convert numpy/pandas-like values inside Plotly payloads to JSON-safe types."""

    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return value
