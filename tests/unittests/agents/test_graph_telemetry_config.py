"""Tests for GraphAgent telemetry configuration."""

from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph.graph_agent_config import TelemetryConfig
from google.adk.agents.graph.graph_node import GraphNode
import pytest


@pytest.fixture
def simple_graph():
  """Create a simple graph for testing."""

  def simple_function(state, ctx):
    return "test output"

  graph = GraphAgent(name="test_graph", description="Test graph")
  graph.add_node(GraphNode(name="start", function=simple_function))
  graph.add_node(GraphNode(name="end", function=simple_function))
  graph.add_edge("start", "end")
  graph.set_start("start")
  graph.set_end("end")
  return graph


def test_telemetry_config_creation():
  """Test creating TelemetryConfig with defaults."""
  config = TelemetryConfig()

  assert config.enabled is True
  assert config.trace_nodes is True
  assert config.trace_edges is True
  assert config.trace_iterations is True
  assert config.trace_parallel_groups is True
  assert config.trace_callbacks is True
  assert config.trace_interrupts is True
  assert config.sampling_rate == 1.0
  assert config.additional_attributes is None


def test_telemetry_config_custom_values():
  """Test creating TelemetryConfig with custom values."""
  config = TelemetryConfig(
      enabled=True,
      trace_nodes=True,
      trace_edges=False,
      trace_iterations=True,
      trace_parallel_groups=False,
      trace_callbacks=True,
      trace_interrupts=False,
      sampling_rate=0.5,
      additional_attributes={"environment": "test"},
  )

  assert config.enabled is True
  assert config.trace_nodes is True
  assert config.trace_edges is False
  assert config.trace_iterations is True
  assert config.trace_parallel_groups is False
  assert config.trace_callbacks is True
  assert config.trace_interrupts is False
  assert config.sampling_rate == 0.5
  assert config.additional_attributes == {"environment": "test"}


def test_telemetry_config_disabled():
  """Test TelemetryConfig with telemetry disabled."""
  config = TelemetryConfig(enabled=False)

  assert config.enabled is False
  # Other settings should still have their defaults
  assert config.trace_nodes is True
  assert config.trace_edges is True


def test_telemetry_config_sampling_rate_validation():
  """Test TelemetryConfig sampling_rate validation."""
  # Valid sampling rates
  config = TelemetryConfig(sampling_rate=0.0)
  assert config.sampling_rate == 0.0

  config = TelemetryConfig(sampling_rate=1.0)
  assert config.sampling_rate == 1.0

  config = TelemetryConfig(sampling_rate=0.5)
  assert config.sampling_rate == 0.5

  # Invalid sampling rates should raise validation error
  with pytest.raises(Exception):  # Pydantic validation error
    TelemetryConfig(sampling_rate=-0.1)

  with pytest.raises(Exception):  # Pydantic validation error
    TelemetryConfig(sampling_rate=1.1)


def test_graph_agent_telemetry_config_none(simple_graph):
  """Test GraphAgent with no telemetry config (defaults to enabled)."""
  assert simple_graph.telemetry_config is None
  assert simple_graph._is_telemetry_enabled() is True
  assert simple_graph._should_trace_nodes() is True
  assert simple_graph._should_trace_edges() is True
  assert simple_graph._should_trace_iterations() is True
  assert simple_graph._should_trace_parallel_groups() is True
  assert simple_graph._should_trace_callbacks() is True
  assert simple_graph._should_trace_interrupts() is True


def test_graph_agent_telemetry_config_enabled():
  """Test GraphAgent with telemetry config enabled."""
  config = TelemetryConfig(enabled=True)
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  assert graph.telemetry_config is config
  assert graph._is_telemetry_enabled() is True
  assert graph._should_trace_nodes() is True
  assert graph._should_trace_edges() is True


def test_graph_agent_telemetry_config_disabled():
  """Test GraphAgent with telemetry config disabled."""
  config = TelemetryConfig(enabled=False)
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  assert graph.telemetry_config is config
  assert graph._is_telemetry_enabled() is False
  assert graph._should_trace_nodes() is False
  assert graph._should_trace_edges() is False
  assert graph._should_trace_iterations() is False
  assert graph._should_trace_parallel_groups() is False
  assert graph._should_trace_callbacks() is False
  assert graph._should_trace_interrupts() is False


def test_graph_agent_telemetry_selective_tracing():
  """Test GraphAgent with selective tracing enabled."""
  config = TelemetryConfig(
      enabled=True,
      trace_nodes=True,
      trace_edges=False,
      trace_iterations=True,
      trace_parallel_groups=False,
      trace_callbacks=True,
      trace_interrupts=False,
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  assert graph._is_telemetry_enabled() is True
  assert graph._should_trace_nodes() is True
  assert graph._should_trace_edges() is False
  assert graph._should_trace_iterations() is True
  assert graph._should_trace_parallel_groups() is False
  assert graph._should_trace_callbacks() is True
  assert graph._should_trace_interrupts() is False


def test_graph_agent_telemetry_only_nodes():
  """Test GraphAgent with only node tracing enabled."""
  config = TelemetryConfig(
      enabled=True,
      trace_nodes=True,
      trace_edges=False,
      trace_iterations=False,
      trace_parallel_groups=False,
      trace_callbacks=False,
      trace_interrupts=False,
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  assert graph._should_trace_nodes() is True
  assert graph._should_trace_edges() is False
  assert graph._should_trace_iterations() is False
  assert graph._should_trace_parallel_groups() is False
  assert graph._should_trace_callbacks() is False
  assert graph._should_trace_interrupts() is False


def test_graph_agent_telemetry_only_edges():
  """Test GraphAgent with only edge tracing enabled."""
  config = TelemetryConfig(
      enabled=True,
      trace_nodes=False,
      trace_edges=True,
      trace_iterations=False,
      trace_parallel_groups=False,
      trace_callbacks=False,
      trace_interrupts=False,
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  assert graph._should_trace_nodes() is False
  assert graph._should_trace_edges() is True
  assert graph._should_trace_iterations() is False


def test_telemetry_config_additional_attributes():
  """Test TelemetryConfig with additional custom attributes."""
  config = TelemetryConfig(
      additional_attributes={
          "environment": "production",
          "version": "1.2.3",
          "team": "ml-platform",
      }
  )

  assert config.additional_attributes == {
      "environment": "production",
      "version": "1.2.3",
      "team": "ml-platform",
  }


def test_telemetry_config_model_serialization():
  """Test TelemetryConfig serialization/deserialization."""
  config = TelemetryConfig(
      enabled=True,
      trace_nodes=True,
      trace_edges=False,
      sampling_rate=0.75,
      additional_attributes={"env": "test"},
  )

  # Serialize to dict
  config_dict = config.model_dump()
  assert config_dict["enabled"] is True
  assert config_dict["trace_nodes"] is True
  assert config_dict["trace_edges"] is False
  assert config_dict["sampling_rate"] == 0.75
  assert config_dict["additional_attributes"] == {"env": "test"}

  # Deserialize from dict
  new_config = TelemetryConfig(**config_dict)
  assert new_config.enabled is True
  assert new_config.trace_nodes is True
  assert new_config.trace_edges is False
  assert new_config.sampling_rate == 0.75
  assert new_config.additional_attributes == {"env": "test"}


def test_telemetry_disabled_overrides_individual_settings():
  """Test that disabling telemetry overrides individual trace settings."""
  config = TelemetryConfig(
      enabled=False,
      trace_nodes=True,  # Even though this is True
      trace_edges=True,  # Even though this is True
      trace_iterations=True,  # Even though this is True
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  # All tracing should be disabled because enabled=False
  assert graph._is_telemetry_enabled() is False
  assert graph._should_trace_nodes() is False
  assert graph._should_trace_edges() is False
  assert graph._should_trace_iterations() is False


def test_graph_agent_init_with_telemetry_config():
  """Test GraphAgent __init__ accepts telemetry_config."""

  def simple_function(state, ctx):
    return "output"

  config = TelemetryConfig(
      enabled=True, trace_nodes=True, trace_edges=False, sampling_rate=0.5
  )

  graph = GraphAgent(
      name="test_graph",
      description="Test graph with telemetry config",
      telemetry_config=config,
  )

  assert graph.telemetry_config is config
  assert graph.telemetry_config.enabled is True
  assert graph.telemetry_config.trace_nodes is True
  assert graph.telemetry_config.trace_edges is False
  assert graph.telemetry_config.sampling_rate == 0.5


def test_should_sample_with_no_config():
  """Test _should_sample() with no telemetry config (defaults to 100%)."""
  graph = GraphAgent(name="test_graph")

  # No config means 100% sampling
  assert graph._should_sample() is True
  assert graph._should_sample() is True
  assert graph._should_sample() is True


def test_should_sample_with_100_percent():
  """Test _should_sample() with 100% sampling rate."""
  config = TelemetryConfig(sampling_rate=1.0)
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  # Should always return True
  for _ in range(10):
    assert graph._should_sample() is True


def test_should_sample_with_0_percent():
  """Test _should_sample() with 0% sampling rate."""
  config = TelemetryConfig(sampling_rate=0.0)
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  # Should always return False
  for _ in range(10):
    assert graph._should_sample() is False


def test_should_sample_with_50_percent(monkeypatch):
  """Test _should_sample() with 50% sampling rate."""
  import random

  config = TelemetryConfig(sampling_rate=0.5)
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  # Mock random.random() to return controlled values
  mock_values = [0.3, 0.7, 0.4, 0.8, 0.1, 0.9]
  mock_values_iter = iter(mock_values)

  def mock_random():
    return next(mock_values_iter)

  monkeypatch.setattr(random, "random", mock_random)

  # 0.3 < 0.5 → sampled
  assert graph._should_sample() is True
  # 0.7 > 0.5 → not sampled
  assert graph._should_sample() is False
  # 0.4 < 0.5 → sampled
  assert graph._should_sample() is True
  # 0.8 > 0.5 → not sampled
  assert graph._should_sample() is False
  # 0.1 < 0.5 → sampled
  assert graph._should_sample() is True
  # 0.9 > 0.5 → not sampled
  assert graph._should_sample() is False


def test_should_sample_with_25_percent(monkeypatch):
  """Test _should_sample() with 25% sampling rate."""
  import random

  config = TelemetryConfig(sampling_rate=0.25)
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  # Mock random.random() to return controlled values
  mock_values = [0.1, 0.3, 0.2, 0.5, 0.15, 0.9]
  mock_values_iter = iter(mock_values)

  def mock_random():
    return next(mock_values_iter)

  monkeypatch.setattr(random, "random", mock_random)

  # 0.1 < 0.25 → sampled
  assert graph._should_sample() is True
  # 0.3 > 0.25 → not sampled
  assert graph._should_sample() is False
  # 0.2 < 0.25 → sampled
  assert graph._should_sample() is True
  # 0.5 > 0.25 → not sampled
  assert graph._should_sample() is False
  # 0.15 < 0.25 → sampled
  assert graph._should_sample() is True
  # 0.9 > 0.25 → not sampled
  assert graph._should_sample() is False


def test_get_telemetry_attributes_no_config():
  """Test _get_telemetry_attributes() with no telemetry config."""
  graph = GraphAgent(name="test_graph")

  base_attrs = {"graph.node.name": "test_node", "graph.node.type": "agent"}
  result = graph._get_telemetry_attributes(base_attrs)

  # Should return base attributes unchanged
  assert result == base_attrs
  assert result["graph.node.name"] == "test_node"
  assert result["graph.node.type"] == "agent"


def test_get_telemetry_attributes_no_additional():
  """Test _get_telemetry_attributes() with config but no additional_attributes."""
  config = TelemetryConfig(enabled=True, sampling_rate=0.5)
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  base_attrs = {"graph.node.name": "test_node"}
  result = graph._get_telemetry_attributes(base_attrs)

  # Should return base attributes unchanged
  assert result == base_attrs
  assert result["graph.node.name"] == "test_node"


def test_get_telemetry_attributes_with_additional():
  """Test _get_telemetry_attributes() merges additional_attributes."""
  config = TelemetryConfig(
      additional_attributes={"environment": "production", "version": "1.2.3"}
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  base_attrs = {"graph.node.name": "test_node", "graph.node.type": "agent"}
  result = graph._get_telemetry_attributes(base_attrs)

  # Should have both base and additional attributes
  assert result["graph.node.name"] == "test_node"
  assert result["graph.node.type"] == "agent"
  assert result["environment"] == "production"
  assert result["version"] == "1.2.3"
  assert len(result) == 4


def test_get_telemetry_attributes_base_takes_precedence():
  """Test _get_telemetry_attributes() - base attributes take precedence."""
  config = TelemetryConfig(
      additional_attributes={
          "environment": "dev",
          "graph.node.name": "should_be_overwritten",
      }
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  base_attrs = {"graph.node.name": "actual_node", "graph.node.type": "function"}
  result = graph._get_telemetry_attributes(base_attrs)

  # Base attributes should override additional_attributes
  assert result["graph.node.name"] == "actual_node"
  assert result["graph.node.type"] == "function"
  assert result["environment"] == "dev"


def test_get_telemetry_attributes_empty_additional():
  """Test _get_telemetry_attributes() with empty additional_attributes dict."""
  config = TelemetryConfig(additional_attributes={})
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  base_attrs = {"graph.node.name": "test_node"}
  result = graph._get_telemetry_attributes(base_attrs)

  # Should return base attributes unchanged
  assert result == base_attrs


def test_get_telemetry_attributes_complex_values():
  """Test _get_telemetry_attributes() with complex attribute values."""
  config = TelemetryConfig(
      additional_attributes={
          "environment": "staging",
          "version": "2.0.1",
          "team": "ml-platform",
          "region": "us-west-2",
      }
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  base_attrs = {
      "graph.node.name": "complex_node",
      "graph.node.type": "agent",
      "graph.node.iteration": 5,
  }
  result = graph._get_telemetry_attributes(base_attrs)

  # Should have all attributes
  assert result["graph.node.name"] == "complex_node"
  assert result["graph.node.type"] == "agent"
  assert result["graph.node.iteration"] == 5
  assert result["environment"] == "staging"
  assert result["version"] == "2.0.1"
  assert result["team"] == "ml-platform"
  assert result["region"] == "us-west-2"
  assert len(result) == 7


def test_get_effective_telemetry_config_no_parent():
  """Test _get_effective_telemetry_config with no parent config."""
  from unittest import mock

  config = TelemetryConfig(
      sampling_rate=0.5, additional_attributes={"env": "test"}
  )
  graph = GraphAgent(name="test_graph", telemetry_config=config)

  # Mock context with no parent config
  ctx = mock.Mock()
  ctx.agent_states = {}

  effective = graph._get_effective_telemetry_config(ctx)

  # Should return own config
  assert effective is config
  assert effective.sampling_rate == 0.5
  assert effective.additional_attributes == {"env": "test"}


def test_get_effective_telemetry_config_parent_only():
  """Test _get_effective_telemetry_config with only parent config."""
  from unittest import mock

  # Child has no config
  graph = GraphAgent(name="child_graph")

  # Mock context with parent config in agent_states
  ctx = mock.Mock()
  ctx.agent_states = {
      "parent_graph": {
          "telemetry_config_dict": {
              "enabled": True,
              "sampling_rate": 0.3,
              "additional_attributes": {"parent": "true"},
              "trace_nodes": True,
              "trace_edges": True,
              "trace_iterations": True,
              "trace_parallel_groups": True,
              "trace_callbacks": True,
              "trace_interrupts": True,
          }
      }
  }

  effective = graph._get_effective_telemetry_config(ctx)

  # Should inherit parent config
  assert effective is not None
  assert effective.sampling_rate == 0.3
  assert effective.additional_attributes == {"parent": "true"}


def test_get_effective_telemetry_config_merge():
  """Test _get_effective_telemetry_config merges parent and own."""
  from unittest import mock

  # Child has own config
  child_config = TelemetryConfig(
      sampling_rate=0.8, additional_attributes={"child": "true", "env": "dev"}
  )
  graph = GraphAgent(name="child_graph", telemetry_config=child_config)

  # Mock context with parent config in agent_states
  ctx = mock.Mock()
  ctx.agent_states = {
      "parent_graph": {
          "telemetry_config_dict": {
              "enabled": True,
              "sampling_rate": 0.3,
              "additional_attributes": {"parent": "true", "version": "1.0"},
              "trace_nodes": True,
              "trace_edges": True,
              "trace_iterations": True,
              "trace_parallel_groups": True,
              "trace_callbacks": True,
              "trace_interrupts": True,
          }
      }
  }

  effective = graph._get_effective_telemetry_config(ctx)

  # Own config takes precedence
  assert effective.sampling_rate == 0.8

  # Additional attributes should be merged (own takes precedence)
  assert effective.additional_attributes["child"] == "true"
  assert effective.additional_attributes["parent"] == "true"
  assert effective.additional_attributes["env"] == "dev"
  assert effective.additional_attributes["version"] == "1.0"


def test_get_effective_telemetry_config_own_takes_precedence():
  """Test that own config values take precedence over parent."""
  from unittest import mock

  # Child config with specific values
  child_config = TelemetryConfig(
      enabled=True,
      sampling_rate=1.0,
      trace_nodes=False,  # Override parent
      additional_attributes={"env": "prod", "override": "child"},
  )
  graph = GraphAgent(name="child_graph", telemetry_config=child_config)

  # Mock context with parent config in agent_states
  ctx = mock.Mock()
  ctx.agent_states = {
      "parent_graph": {
          "telemetry_config_dict": {
              "enabled": True,
              "sampling_rate": 0.1,  # Should be overridden
              "trace_nodes": True,  # Should be overridden
              "trace_edges": True,
              "trace_iterations": True,
              "trace_parallel_groups": True,
              "trace_callbacks": True,
              "trace_interrupts": True,
              "additional_attributes": {"env": "dev", "parent_only": "value"},
          }
      }
  }

  effective = graph._get_effective_telemetry_config(ctx)

  # Own values take precedence
  assert effective.sampling_rate == 1.0
  assert effective.trace_nodes is False  # Child overrode this
  assert effective.trace_edges is True  # Inherited from parent

  # Attributes merged, own takes precedence
  assert effective.additional_attributes["env"] == "prod"  # Child overrode
  assert effective.additional_attributes["override"] == "child"  # Child only
  assert (
      effective.additional_attributes["parent_only"] == "value"
  )  # Parent only
