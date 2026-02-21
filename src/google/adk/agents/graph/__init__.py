"""Graph-based agent components.

This module contains components for graph-based workflow orchestration:
- GraphAgent: Main graph workflow agent
- GraphState: Domain data container
- GraphAgentState: Execution tracking state (BaseAgentState)
- GraphNode: Node wrapper for agents and functions
- EdgeCondition: Conditional routing between nodes
- StateReducer: State merge strategies
- GraphEvent: Typed events for streaming
- GraphEventType: Event type enumeration
- GraphStreamMode: Stream mode enumeration
- NodeCallbackContext: Context for node lifecycle callbacks
- EdgeCallbackContext: Context for edge condition callbacks
- NodeCallback: Type for node lifecycle callbacks
- EdgeCallback: Type for edge condition callbacks
"""

from __future__ import annotations

from .callbacks import create_nested_observability_callback
from .callbacks import EdgeCallback
from .callbacks import EdgeCallbackContext
from .callbacks import NodeCallback
from .callbacks import NodeCallbackContext
from .evaluation_metrics import graph_path_match
from .evaluation_metrics import node_execution_count
from .evaluation_metrics import state_contains_keys
from .graph_agent import GraphAgent
from .graph_agent_config import GraphAgentConfig
from .graph_agent_config import GraphEdgeConfig
from .graph_agent_config import GraphNodeConfig
from .graph_agent_state import GraphAgentState
from .graph_edge import EdgeCondition
from .graph_events import GraphEvent
from .graph_events import GraphEventType
from .graph_events import GraphStreamMode
from .graph_export import export_execution_timeline
from .graph_export import export_graph_structure
from .graph_export import export_graph_with_execution
from .graph_node import GraphNode
from .graph_rewind import rewind_to_node
from .graph_state import GraphState
from .graph_state import PydanticJSONEncoder
from .graph_state import StateReducer

# Sentinel constants for graph boundaries
START = "__start__"
END = "__end__"

__all__ = [
    "GraphAgent",
    "GraphAgentConfig",
    "GraphAgentState",
    "GraphNodeConfig",
    "GraphEdgeConfig",
    "GraphState",
    "GraphNode",
    "EdgeCondition",
    "StateReducer",
    "PydanticJSONEncoder",
    "GraphEvent",
    "GraphEventType",
    "GraphStreamMode",
    "NodeCallbackContext",
    "EdgeCallbackContext",
    "NodeCallback",
    "EdgeCallback",
    "create_nested_observability_callback",
    "graph_path_match",
    "state_contains_keys",
    "node_execution_count",
    "export_graph_structure",
    "export_graph_with_execution",
    "export_execution_timeline",
    "rewind_to_node",
    "START",
    "END",
]
