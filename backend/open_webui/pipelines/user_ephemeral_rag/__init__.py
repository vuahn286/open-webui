"""User ephemeral RAG pipeline configuration helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

PIPELINE_ROOT = Path(__file__).parent
PIPELINE_CONFIG_PATH = PIPELINE_ROOT / "pipeline.yaml"

try:  # pragma: no cover - optional dependency loading
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback path used in tests
    yaml = None  # type: ignore[assignment]


def load_pipeline_config() -> Dict[str, Any]:
    """Return the parsed pipeline configuration for the user ephemeral pipeline."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to parse pipeline.yaml. Install `pyyaml` to use this helper."
        )
    with PIPELINE_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


__all__ = [
    "PIPELINE_CONFIG_PATH",
    "PIPELINE_ROOT",
    "load_pipeline_config",
]
