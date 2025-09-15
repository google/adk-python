from enum import Enum

from google.adk.cli.adk_web_server import RunAgentRequest


class OptimizationLevel(str, Enum):
  """Enumeration for the available SSE optimization levels."""

  MINIMAL = "minimal"
  BALANCED = "balanced"
  FULL_COMPAT = "full_compat"


class RunAgentRequestOptimized(RunAgentRequest):
  """
  Request model for the enhanced SSE endpoint.
  This can extend the ADK's `RunAgentRequest` if available, or be standalone.
  """

  optimization_level: OptimizationLevel = OptimizationLevel.FULL_COMPAT
