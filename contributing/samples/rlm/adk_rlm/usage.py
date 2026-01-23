"""
Usage tracking for ADK-RLM.

Tracks token usage across multiple models during RLM execution.
"""

from collections import defaultdict

from adk_rlm.types import ModelUsageSummary
from adk_rlm.types import UsageSummary


class UsageTracker:
  """Tracks token usage across multiple models."""

  def __init__(self):
    """Initialize the usage tracker."""
    self._calls: dict[str, int] = defaultdict(int)
    self._input_tokens: dict[str, int] = defaultdict(int)
    self._output_tokens: dict[str, int] = defaultdict(int)

  def add(
      self,
      model: str,
      input_tokens: int = 0,
      output_tokens: int = 0,
  ) -> None:
    """
    Add usage for a model call.

    Args:
        model: The model name.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens used.
    """
    self._calls[model] += 1
    self._input_tokens[model] += input_tokens
    self._output_tokens[model] += output_tokens

  def add_from_response(self, model: str, usage_metadata) -> None:
    """
    Add usage from a Gemini response's usage_metadata.

    Args:
        model: The model name.
        usage_metadata: The usage_metadata from a Gemini response.
    """
    if usage_metadata is None:
      self._calls[model] += 1
      return

    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    self.add(model, input_tokens, output_tokens)

  def get_summary(self) -> UsageSummary:
    """
    Get the aggregated usage summary.

    Returns:
        UsageSummary with per-model usage data.
    """
    model_summaries = {}
    for model in self._calls:
      model_summaries[model] = ModelUsageSummary(
          total_calls=self._calls[model],
          total_input_tokens=self._input_tokens[model],
          total_output_tokens=self._output_tokens[model],
      )
    return UsageSummary(model_usage_summaries=model_summaries)

  def reset(self) -> None:
    """Reset all usage tracking."""
    self._calls.clear()
    self._input_tokens.clear()
    self._output_tokens.clear()

  def merge(self, other: "UsageTracker") -> None:
    """
    Merge usage from another tracker into this one.

    Args:
        other: Another UsageTracker to merge in.
    """
    for model in other._calls:
      self._calls[model] += other._calls[model]
      self._input_tokens[model] += other._input_tokens[model]
      self._output_tokens[model] += other._output_tokens[model]

  @property
  def total_calls(self) -> int:
    """Return total number of calls across all models."""
    return sum(self._calls.values())

  @property
  def total_input_tokens(self) -> int:
    """Return total input tokens across all models."""
    return sum(self._input_tokens.values())

  @property
  def total_output_tokens(self) -> int:
    """Return total output tokens across all models."""
    return sum(self._output_tokens.values())
