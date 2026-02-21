"""Telemetry mixins for agent observability.

Two-layer design for reusability:
- AgentTelemetryMixin: Generic telemetry (any agent with telemetry_config)
- GraphTelemetryMixin: Graph-specific trace toggles (nodes, edges, etc.)
"""

from __future__ import annotations

import random
from typing import Any
from typing import Dict
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..invocation_context import InvocationContext
  from .graph_agent_config import TelemetryConfig


class AgentTelemetryMixin:
  """Generic telemetry mixin for any agent with a telemetry_config field.

  Expects the host class to have:
  - self.telemetry_config: Optional[TelemetryConfig]
  """

  telemetry_config: Any  # Declared for type checking
  name: str  # Provided by host agent class

  def _is_telemetry_enabled(self) -> bool:
    """Check if telemetry is enabled globally."""
    if not self.telemetry_config:
      return True  # Default to enabled if no config
    return bool(self.telemetry_config.enabled)

  def _should_sample(
      self, effective_config: Optional[TelemetryConfig] = None
  ) -> bool:
    """Check if current operation should be sampled.

    Uses random sampling to control telemetry volume.

    Args:
        effective_config: Effective telemetry config (merged parent + own).
                          If None, uses self.telemetry_config.
    """
    config = effective_config or self.telemetry_config
    if not config:
      return True  # Default to 100% sampling if no config
    return bool(random.random() < config.sampling_rate)

  def _get_telemetry_attributes(
      self,
      base_attributes: Dict[str, Any],
      effective_config: Optional[TelemetryConfig] = None,
  ) -> Dict[str, Any]:
    """Get telemetry attributes including custom attributes.

    Args:
        base_attributes: Base attributes for the telemetry event
        effective_config: Effective config. If None, uses self.telemetry_config.

    Returns:
        Combined attributes with additional custom attributes
    """
    config = effective_config or self.telemetry_config
    if not config or not config.additional_attributes:
      return base_attributes

    combined = dict(config.additional_attributes)
    combined.update(base_attributes)
    return combined

  def _get_parent_telemetry_config(
      self, ctx: InvocationContext
  ) -> Optional[Dict[str, Any]]:
    """Get parent telemetry config from agent_states.

    Used for nested agents to inherit telemetry settings from parent.

    Args:
        ctx: Invocation context with agent_states
    """
    if not ctx.agent_states:
      return None
    for agent_name, state_dict in ctx.agent_states.items():
      if agent_name != self.name and isinstance(state_dict, dict):
        config = state_dict.get("telemetry_config_dict")
        if config is not None:
          return dict(config) if isinstance(config, dict) else None
    return None

  def _get_effective_telemetry_config(
      self, ctx: InvocationContext
  ) -> Optional[TelemetryConfig]:
    """Get effective telemetry config by merging parent and own config.

    Own config takes precedence over parent config.

    Args:
        ctx: Invocation context with session state
    """
    parent_config_dict = self._get_parent_telemetry_config(ctx)

    if not parent_config_dict:
      return self.telemetry_config  # type: ignore[no-any-return]

    if not self.telemetry_config:
      from .graph_agent_config import TelemetryConfig

      return TelemetryConfig(**parent_config_dict)

    # Merge: own config takes precedence
    merged_dict = parent_config_dict.copy()
    own_dict = self.telemetry_config.model_dump()
    for key, value in own_dict.items():
      if key == "additional_attributes" and value is not None:
        parent_attrs = merged_dict.get("additional_attributes") or {}
        own_attrs = value or {}
        merged_dict["additional_attributes"] = {**parent_attrs, **own_attrs}
      elif value is not None:
        merged_dict[key] = value

    from .graph_agent_config import TelemetryConfig

    return TelemetryConfig(**merged_dict)


class GraphTelemetryMixin(AgentTelemetryMixin):
  """Graph-specific telemetry toggles.

  Extends AgentTelemetryMixin with granular trace controls for
  graph execution components (nodes, edges, iterations, etc.).
  """

  def _should_trace_nodes(self) -> bool:
    """Check if node execution tracing is enabled."""
    if not self._is_telemetry_enabled():
      return False
    if not self.telemetry_config:
      return True
    return bool(self.telemetry_config.trace_nodes)

  def _should_trace_edges(self) -> bool:
    """Check if edge evaluation tracing is enabled."""
    if not self._is_telemetry_enabled():
      return False
    if not self.telemetry_config:
      return True
    return bool(self.telemetry_config.trace_edges)

  def _should_trace_iterations(self) -> bool:
    """Check if graph iteration metrics are enabled."""
    if not self._is_telemetry_enabled():
      return False
    if not self.telemetry_config:
      return True
    return bool(self.telemetry_config.trace_iterations)

  def _should_trace_parallel_groups(self) -> bool:
    """Check if parallel group execution tracing is enabled."""
    if not self._is_telemetry_enabled():
      return False
    if not self.telemetry_config:
      return True
    return bool(self.telemetry_config.trace_parallel_groups)

  def _should_trace_callbacks(self) -> bool:
    """Check if callback execution tracing is enabled."""
    if not self._is_telemetry_enabled():
      return False
    if not self.telemetry_config:
      return True
    return bool(self.telemetry_config.trace_callbacks)

  def _should_trace_interrupts(self) -> bool:
    """Check if interrupt check tracing is enabled."""
    if not self._is_telemetry_enabled():
      return False
    if not self.telemetry_config:
      return True
    return bool(self.telemetry_config.trace_interrupts)
