"""
Tests for usage tracking.
"""

from adk_rlm.usage import UsageTracker
import pytest


class TestUsageTracker:
  """Tests for UsageTracker."""

  def test_track_single_call(self):
    """Track single call."""
    tracker = UsageTracker()
    tracker.add("gemini-pro", input_tokens=100, output_tokens=50)

    assert tracker.total_calls == 1
    assert tracker.total_input_tokens == 100
    assert tracker.total_output_tokens == 50

  def test_track_multiple_calls_same_model(self):
    """Multiple calls to same model aggregate."""
    tracker = UsageTracker()
    tracker.add("gemini-pro", input_tokens=100, output_tokens=50)
    tracker.add("gemini-pro", input_tokens=200, output_tokens=100)

    assert tracker.total_calls == 2
    assert tracker.total_input_tokens == 300
    assert tracker.total_output_tokens == 150

  def test_track_multiple_models(self):
    """Calls to different models tracked separately."""
    tracker = UsageTracker()
    tracker.add("gemini-pro", input_tokens=100, output_tokens=50)
    tracker.add("gemini-flash", input_tokens=200, output_tokens=100)

    summary = tracker.get_summary()

    assert len(summary.model_usage_summaries) == 2
    assert summary.model_usage_summaries["gemini-pro"].total_calls == 1
    assert summary.model_usage_summaries["gemini-flash"].total_calls == 1

  def test_get_summary(self):
    """Get usage summary."""
    tracker = UsageTracker()
    tracker.add("model1", input_tokens=100, output_tokens=50)
    tracker.add("model2", input_tokens=200, output_tokens=100)

    summary = tracker.get_summary()

    assert "model1" in summary.model_usage_summaries
    assert "model2" in summary.model_usage_summaries
    assert summary.total_calls == 2

  def test_reset(self):
    """Reset clears all tracking."""
    tracker = UsageTracker()
    tracker.add("model", input_tokens=100, output_tokens=50)
    tracker.reset()

    assert tracker.total_calls == 0
    assert tracker.total_input_tokens == 0
    assert tracker.total_output_tokens == 0

  def test_zero_usage(self):
    """No calls returns zeros."""
    tracker = UsageTracker()
    summary = tracker.get_summary()

    assert len(summary.model_usage_summaries) == 0
    assert tracker.total_calls == 0
