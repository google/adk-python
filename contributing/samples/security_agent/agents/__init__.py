"""BigQuery ADK Agent Module."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Optional, Tuple

agent: Optional[ModuleType]
root_agent: Optional[object]


def _safe_import_agent() -> Tuple[Optional[ModuleType], Optional[object]]:
    """Import the heavy ADK agent lazily, tolerating missing optional deps."""

    try:
        agent_module = import_module(".agent", package=__name__)
    except ModuleNotFoundError as exc:
        if exc.name in {"google.adk", "google", "google.adk.agents"}:
            return None, None
        raise

    try:
        root = getattr(agent_module, "root_agent")
    except AttributeError:
        root = None

    return agent_module, root


agent, root_agent = _safe_import_agent()

__all__ = []
if agent is not None:
    __all__.append("agent")
if root_agent is not None:
    __all__.append("root_agent")
