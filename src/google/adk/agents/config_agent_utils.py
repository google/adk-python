# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import inspect
import logging
import os
import typing
from typing import Any
from typing import List

from google.genai import types
from pydantic import BaseModel
import yaml

from ..features import experimental
from ..features import FeatureName
from ..tools.tool_configs import ToolConfig
from .base_agent import BaseAgent
from .common_configs import AgentRefConfig
from .common_configs import CodeConfig

logger = logging.getLogger("google_adk." + __name__)


def _is_callback_type(annotation: Any) -> bool:
  """Checks if the type annotation is a callback or list of callbacks."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if origin is typing.Callable:
    return True

  if origin in [typing.Union, getattr(typing, "UnionType", ())]:
    return any(_is_callback_type(arg) for arg in args)

  if origin is list:
    return any(_is_callback_type(arg) for arg in args)

  return False


def _is_schema_type(annotation: Any) -> bool:
  """Checks if the type annotation involves a schema type."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if origin in [typing.Union, getattr(typing, "UnionType", ())]:
    return any(_is_schema_type(arg) for arg in args)

  if origin is type:
    for arg in args:
      if isinstance(arg, type) and issubclass(arg, BaseModel):
        return True

  if annotation is types.SchemaUnion or annotation is types.Schema:
    return True

  return False


def _is_tools_type(annotation: Any) -> bool:
  """Checks if the type annotation is a list of tools."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if origin is list:
    for arg in args:
      from ..tools.base_tool import BaseTool
      from ..tools.base_toolset import BaseToolset

      if isinstance(arg, type) and issubclass(arg, (BaseTool, BaseToolset)):
        return True

      arg_origin = typing.get_origin(arg)
      arg_args = typing.get_args(arg)
      if arg_origin in [typing.Union, getattr(typing, "UnionType", ())]:
        if any(
            isinstance(a, type) and issubclass(a, (BaseTool, BaseToolset))
            for a in arg_args
        ):
          return True
  return False


def _is_sub_agents_type(annotation: Any) -> bool:
  """Checks if the type annotation is a list of agents."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if origin is list:
    for arg in args:
      if isinstance(arg, type) and issubclass(arg, BaseAgent):
        return True
  return False


def _is_workflow_edges_type(annotation: Any) -> bool:
  """Checks if the type annotation is a list of EdgeItem."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if origin is list:
    for arg in args:
      from ..workflow._graph import EdgeItem

      if arg is EdgeItem:
        return True
  return False


def _is_llm_type(annotation: Any) -> bool:
  """Checks if the type annotation involves a BaseLlm type."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  from ..models.base_llm import BaseLlm

  if isinstance(annotation, type) and issubclass(annotation, BaseLlm):
    return True

  if origin in [typing.Union, getattr(typing, "UnionType", ())]:
    return any(_is_llm_type(arg) for arg in args)

  return False


class AgentConfigMapper:
  """Maps YAML data to Agent class fields dynamically."""

  def __init__(self, abs_path: str):
    self.abs_path = abs_path
    self._resolved_nodes_cache = {}

  def _resolve_tools(self, tool_configs: list[ToolConfig]) -> list[Any]:
    """Resolve tools from configuration."""
    from ..tools.base_tool import BaseTool
    from ..tools.base_toolset import BaseToolset

    resolved_tools = []
    for tool_config in tool_configs:
      if "." not in tool_config.name:
        # ADK built-in tools
        module = importlib.import_module("google.adk.tools")
        obj = getattr(module, tool_config.name)
      else:
        # User-defined tools
        module_path, obj_name = tool_config.name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)

      if isinstance(obj, BaseTool) or isinstance(obj, BaseToolset):
        resolved_tools.append(obj)
      elif inspect.isclass(obj) and (
          issubclass(obj, BaseTool) or issubclass(obj, BaseToolset)
      ):
        resolved_tools.append(obj.from_config(tool_config.args, self.abs_path))
      elif callable(obj):
        if tool_config.args:
          resolved_tools.append(obj(tool_config.args))
        else:
          resolved_tools.append(obj)
      else:
        raise ValueError(f"Invalid tool YAML config: {tool_config}.")

    return resolved_tools

  def _resolve_edges(self, value: list[Any]) -> list[Any]:
    """Resolve edges to support agent references."""
    from ..workflow._base_node import BaseNode
    from ..workflow._graph import Edge

    processed_edges = []
    for edge_item in value:
      if isinstance(edge_item, list):
        # It's a tuple of ChainElement
        processed_chain = []
        for element in edge_item:
          processed_chain.append(self._resolve_chain_element(element))
        processed_edges.append(processed_chain)
      elif isinstance(edge_item, dict):
        # Check if it matches Edge fields
        edge_fields = Edge.model_fields
        if all(k in edge_fields for k in edge_item.keys()):
          processed_edge = {}
          for k, v in edge_item.items():
            field = edge_fields[k]
            annotation = field.annotation
            if annotation is BaseNode or (
                isinstance(annotation, type)
                and issubclass(annotation, BaseNode)
            ):
              processed_edge[k] = self._resolve_node_like(v)
            else:
              processed_edge[k] = v
          processed_edge_item = processed_edge
        else:
          # Assume RoutingMap or NodeLike
          processed_edge_item = self._resolve_chain_element(edge_item)
        processed_edges.append(processed_edge_item)
      else:
        processed_edges.append(edge_item)
    return processed_edges

  def _resolve_chain_element(self, element: Any) -> Any:
    """Resolve a chain element in an edge."""
    if isinstance(element, list):
      return [self._resolve_node_like(e) for e in element]
    elif isinstance(element, dict):
      if (
          "name" in element
          or "config_path" in element
          or "agent_class" in element
      ):
        return self._resolve_node_like(element)
      else:
        # Assume RoutingMap
        processed_map = {}
        for k, v in element.items():
          processed_map[k] = self._resolve_chain_element(v)
        return processed_map
    else:
      return self._resolve_node_like(element)

  def _resolve_node_like(self, node_like: Any) -> Any:
    """Resolve a NodeLike item, handling agent references and FunctionNodes."""
    if isinstance(node_like, str):
      if node_like == "START":
        return node_like
      if node_like in self._resolved_nodes_cache:
        return self._resolved_nodes_cache[node_like]

      if node_like.endswith(".yaml"):
        ref = AgentRefConfig(config_path=node_like)
        resolved = resolve_agent_reference(ref, self.abs_path)
        self._resolved_nodes_cache[node_like] = resolved
        return resolved
      else:
        # Assume it's a function reference!
        func_path = node_like
        if func_path.startswith("."):
          # Relative to current package!
          dir_path = os.path.dirname(self.abs_path)
          pkg_name = os.path.basename(dir_path)
          func_path = pkg_name + func_path

        func = resolve_fully_qualified_name(func_path)
        from ..workflow._function_node import FunctionNode

        # Use the function name as node name!
        node_name = func_path.rsplit(".", 1)[-1]
        resolved = FunctionNode(name=node_name, func=func)
        self._resolved_nodes_cache[node_like] = resolved
        return resolved
    elif isinstance(node_like, dict):
      node_id = id(node_like)
      if node_id in self._resolved_nodes_cache:
        return self._resolved_nodes_cache[node_id]

      if "config_path" in node_like:
        ref = AgentRefConfig(**node_like)
        resolved = resolve_agent_reference(ref, self.abs_path)
        self._resolved_nodes_cache[node_id] = resolved
        return resolved

      if "agent_class" in node_like:
        cls_name = node_like.get("agent_class")

        if cls_name == "FunctionNode":
          func_code = node_like.get("func_code")
          if func_code and isinstance(func_code, str):
            func = resolve_fully_qualified_name(func_code)
            from ..workflow._function_node import FunctionNode

            kwargs = {
                k: v
                for k, v in node_like.items()
                if k not in ("agent_class", "func_code")
            }
            resolved = FunctionNode(func=func, **kwargs)
            self._resolved_nodes_cache[node_id] = resolved
            return resolved
        else:
          # Use AgentConfigMapper to map fields!
          mapper = AgentConfigMapper(self.abs_path)
          mapped_kwargs, cls = mapper.map(node_like)
          resolved = cls(**mapped_kwargs)
          self._resolved_nodes_cache[node_id] = resolved
          return resolved

      return node_like
    return node_like

  def map(self, data: dict[str, Any]) -> tuple[dict[str, Any], type[Any]]:
    agent_class_name = data.get("agent_class", "LlmAgent")
    agent_class = _resolve_agent_class(agent_class_name)
    fields = agent_class.model_fields

    kwargs = {}

    for name, value in data.items():
      if name == "agent_class":
        continue

      target_name = name
      is_code_ref = False

      if name.endswith("_code"):
        base_name = name.removesuffix("_code")
        if base_name in fields:
          target_name = base_name
          is_code_ref = True

      if target_name in fields:
        if is_code_ref:
          code_val = value
          if isinstance(code_val, str) and code_val.startswith("."):
            dir_path = os.path.dirname(self.abs_path)
            pkg_name = os.path.basename(dir_path)
            code_val = pkg_name + code_val
          elif isinstance(code_val, dict) and code_val.get(
              "name", ""
          ).startswith("."):
            dir_path = os.path.dirname(self.abs_path)
            pkg_name = os.path.basename(dir_path)
            code_val = dict(code_val)
            code_val["name"] = pkg_name + code_val["name"]

          kwargs[target_name] = resolve_code_reference(
              CodeConfig(**code_val)
              if isinstance(code_val, dict)
              else CodeConfig(name=code_val)
          )
        else:
          kwargs[target_name] = self._map_field(target_name, value, fields)
      else:
        kwargs[name] = value

    return kwargs, agent_class

  def _map_field(self, name: str, value: Any, fields: dict[str, Any]) -> Any:
    field = fields.get(name)
    if not field:
      return value

    annotation = field.annotation

    # Rule 1: Callback
    if _is_callback_type(annotation):
      if isinstance(value, list):
        return resolve_callbacks([
            CodeConfig(**v) if isinstance(v, dict) else CodeConfig(name=v)
            for v in value
        ])
      elif isinstance(value, dict):
        return resolve_code_reference(CodeConfig(**value))
      elif isinstance(value, str):
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if (
            origin in [typing.Union, getattr(typing, "UnionType", ())]
            and str in args
        ):
          return value
        return resolve_code_reference(CodeConfig(name=value))

    # Rule 2: Schemas
    if _is_schema_type(annotation):
      if isinstance(value, dict):
        return resolve_code_reference(CodeConfig(**value))
      elif isinstance(value, str):
        return resolve_code_reference(CodeConfig(name=value))

    # Rule 3: Tools
    if _is_tools_type(annotation) and isinstance(value, list):
      tool_configs = [
          ToolConfig(**v) if isinstance(v, dict) else ToolConfig(name=v)
          for v in value
      ]
      return self._resolve_tools(tool_configs)

    # Rule 4: Sub Agents
    if _is_sub_agents_type(annotation) and isinstance(value, list):
      sub_agents = []
      for sub_agent_config in value:
        ref = (
            AgentRefConfig(**sub_agent_config)
            if isinstance(sub_agent_config, dict)
            else AgentRefConfig(config_path=sub_agent_config)
        )
        sub_agents.append(resolve_agent_reference(ref, self.abs_path))
      return sub_agents

    # Rule 5: LLM (Legacy model mapping or custom LLM)
    if _is_llm_type(annotation) and isinstance(value, dict) and "name" in value:
      return resolve_code_reference(CodeConfig(**value))

    # Rule 6: Workflow Edges
    if _is_workflow_edges_type(annotation) and isinstance(value, list):
      return self._resolve_edges(value)

    return value


@experimental(FeatureName.AGENT_CONFIG)
def from_config(config_path: str) -> BaseAgent:
  """Build agent from a configfile path.

  Args:
    config_path: the path to a YAML config file.

  Returns:
    The created agent instance.

  Raises:
    FileNotFoundError: If config file doesn't exist.
    ValidationError: If config file's content is invalid YAML.
    ValueError: If agent type is unsupported.
  """
  abs_path = os.path.abspath(config_path)
  if not os.path.exists(abs_path):
    raise FileNotFoundError(f"Config file not found: {abs_path}")

  with open(abs_path, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

  if config_data is None:
    config_data = {}
  elif not isinstance(config_data, dict):
    raise ValueError(
        f"Invalid agent config in {abs_path!r}. Expected a dictionary."
    )

  if _ENFORCE_DENYLIST:
    _check_config_for_blocked_keys(config_data, abs_path)

  mapper = AgentConfigMapper(abs_path)
  kwargs, agent_class = mapper.map(config_data)

  return agent_class(**kwargs)


def _resolve_agent_class(agent_class: str) -> type[Any]:
  """Resolve the agent class from its fully qualified name."""
  from ..workflow._base_node import BaseNode

  agent_class_name = agent_class or "LlmAgent"
  if "." not in agent_class_name:
    # Try agents first
    try:
      cls = resolve_fully_qualified_name(
          f"google.adk.agents.{agent_class_name}"
      )
      if inspect.isclass(cls) and issubclass(cls, BaseNode):
        return cls
    except Exception:
      pass

    if agent_class_name == "Workflow":
      from google.adk.workflow import Workflow

      return Workflow
    elif agent_class_name == "FunctionNode":
      from google.adk.workflow import FunctionNode

      return FunctionNode

  agent_class = resolve_fully_qualified_name(agent_class_name)
  if inspect.isclass(agent_class) and issubclass(agent_class, BaseNode):
    return agent_class

  raise ValueError(
      f"Invalid class `{agent_class_name}`. It must be a subclass of BaseNode."
  )


_BLOCKED_YAML_KEYS = frozenset({"args"})
_ENFORCE_DENYLIST = False


def _set_enforce_denylist(value: bool) -> None:
  global _ENFORCE_DENYLIST
  _ENFORCE_DENYLIST = value


def _check_config_for_blocked_keys(node: Any, filename: str) -> None:
  """Recursively check if the configuration contains any blocked keys."""
  if isinstance(node, dict):
    for key, value in node.items():
      if key in _BLOCKED_YAML_KEYS:
        raise ValueError(
            f"Blocked key {key!r} found in {filename!r}. "
            f"The '{key}' field is not allowed in agent configurations "
            "because it can execute arbitrary code."
        )
      _check_config_for_blocked_keys(value, filename)
  elif isinstance(node, list):
    for item in node:
      _check_config_for_blocked_keys(item, filename)


@experimental(FeatureName.AGENT_CONFIG)
def resolve_fully_qualified_name(name: str) -> Any:
  try:
    module_path, obj_name = name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
  except Exception as e:
    raise ValueError(f"Invalid fully qualified name: {name}") from e


@experimental(FeatureName.AGENT_CONFIG)
def resolve_agent_reference(
    ref_config: AgentRefConfig, referencing_agent_config_abs_path: str
) -> BaseAgent:
  """Build an agent from a reference.

  Args:
    ref_config: The agent reference configuration (AgentRefConfig).
    referencing_agent_config_abs_path: The absolute path to the agent config
    that contains the reference.

  Returns:
    The created agent instance.
  """
  if ref_config.config_path:
    if os.path.isabs(ref_config.config_path):
      return from_config(ref_config.config_path)
    else:
      return from_config(
          os.path.join(
              os.path.dirname(referencing_agent_config_abs_path),
              ref_config.config_path,
          )
      )
  elif ref_config.code:
    return _resolve_agent_code_reference(ref_config.code)
  else:
    raise ValueError("AgentRefConfig must have either 'code' or 'config_path'")


def _resolve_agent_code_reference(code: str) -> Any:
  """Resolve a code reference to an actual agent instance.

  Args:
    code: The fully-qualified path to an agent instance.

  Returns:
    The resolved agent instance.

  Raises:
    ValueError: If the agent reference cannot be resolved.
  """
  if "." not in code:
    raise ValueError(f"Invalid code reference: {code}")

  module_path, obj_name = code.rsplit(".", 1)
  module = importlib.import_module(module_path)
  obj = getattr(module, obj_name)

  if callable(obj):
    raise ValueError(f"Invalid agent reference to a callable: {code}")

  if not isinstance(obj, BaseAgent):
    raise ValueError(f"Invalid agent reference to a non-agent instance: {code}")

  return obj


@experimental(FeatureName.AGENT_CONFIG)
def resolve_code_reference(code_config: CodeConfig) -> Any:
  """Resolve a code reference to actual Python object.

  Args:
    code_config: The code configuration (CodeConfig).

  Returns:
    The resolved Python object.

  Raises:
    ValueError: If the code reference cannot be resolved.
  """
  if not code_config or not code_config.name:
    raise ValueError("Invalid CodeConfig.")

  module_path, obj_name = code_config.name.rsplit(".", 1)
  module = importlib.import_module(module_path)
  obj = getattr(module, obj_name)

  return obj


@experimental(FeatureName.AGENT_CONFIG)
def resolve_callbacks(callbacks_config: List[CodeConfig]) -> Any:
  """Resolve callbacks from configuration.

  Args:
    callbacks_config: List of callback configurations (CodeConfig objects).

  Returns:
    List of resolved callback objects.
  """
  return [resolve_code_reference(config) for config in callbacks_config]
