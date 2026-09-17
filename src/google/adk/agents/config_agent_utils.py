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

import collections.abc
import importlib
import inspect
import logging
import os
import sys
from types import UnionType
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
import warnings

from google.genai import types
from pydantic import BaseModel
import yaml

from ..features import experimental
from ..features import FeatureName
from ..tools.tool_configs import ToolConfig
from .base_agent import BaseAgent
from .common_configs import AgentRefConfig
from .common_configs import CodeConfig

if TYPE_CHECKING:
  # For annotations only: the runtime imports stay inside the functions that
  # need it, because google.adk.workflow imports back into google.adk.agents.
  from ..workflow._base_node import BaseNode

logger = logging.getLogger("google_adk." + __name__)


_UNION_ORIGINS = (typing.Union, UnionType)


def _is_union(origin: Any) -> bool:
  """Whether a `typing.get_origin` result is a union.

  Both spellings have to be accepted: `X | Y` reports `types.UnionType` while
  `Optional[X]` reports `typing.Union`. `UnionType` is imported from `types`,
  not `typing`, which has no such attribute -- reading it off `typing` left the
  tuple without it, so every `X | None` annotation (`state_schema`, among
  others) fell through unresolved.
  """
  return origin in _UNION_ORIGINS


def _is_callback_type(annotation: Any) -> bool:
  """Checks if the type annotation is a callback or list of callbacks."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if origin in (typing.Callable, collections.abc.Callable):
    return True

  if _is_union(origin):
    if str in args:
      return False
    return any(_is_callback_type(arg) for arg in args)

  if origin is list:
    return any(_is_callback_type(arg) for arg in args)

  return False


def _is_schema_type(annotation: Any) -> bool:
  """Checks if the type annotation involves a schema type."""
  from ..models.base_llm import BaseLlm
  from ..workflow._base_node import BaseNode

  if isinstance(annotation, type) and issubclass(
      annotation, (BaseNode, BaseLlm)
  ):
    return False

  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if _is_union(origin):
    return any(_is_schema_type(arg) for arg in args)

  if origin is type:
    for arg in args:
      if isinstance(arg, type) and issubclass(arg, BaseModel):
        if not issubclass(arg, (BaseNode, BaseLlm)):
          return True

  if annotation is types.SchemaUnion or annotation is types.Schema:
    return True

  return False


def _is_tools_type(annotation: Any) -> bool:
  """Checks if the type annotation is a list of tools."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if _is_union(origin):
    return any(_is_tools_type(arg) for arg in args)

  if origin is list:
    for arg in args:
      from ..tools.base_tool import BaseTool
      from ..tools.base_toolset import BaseToolset

      if isinstance(arg, type) and issubclass(arg, (BaseTool, BaseToolset)):
        return True

      arg_origin = typing.get_origin(arg)
      arg_args = typing.get_args(arg)
      if _is_union(arg_origin):
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

  if _is_union(origin):
    return any(_is_sub_agents_type(arg) for arg in args)

  if origin is list:
    for arg in args:
      if isinstance(arg, type) and issubclass(arg, BaseAgent):
        return True
  return False


def _is_workflow_edges_type(annotation: Any) -> bool:
  """Checks if the type annotation is a list of EdgeItem."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  if _is_union(origin):
    return any(_is_workflow_edges_type(arg) for arg in args)

  if origin is list:
    for arg in args:
      from ..workflow._graph import Edge
      from ..workflow._graph import EdgeItem

      if arg is EdgeItem or arg == EdgeItem:
        return True
      if isinstance(arg, type) and issubclass(arg, Edge):
        return True
      arg_origin = typing.get_origin(arg)
      arg_args = typing.get_args(arg)
      if _is_union(arg_origin):
        if any(
            (a is EdgeItem or (isinstance(a, type) and issubclass(a, Edge)))
            for a in arg_args
        ):
          return True
  return False


def _is_llm_type(annotation: Any) -> bool:
  """Checks if the type annotation involves a BaseLlm type."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)

  from ..models.base_llm import BaseLlm

  if isinstance(annotation, type) and issubclass(annotation, BaseLlm):
    return True

  if _is_union(origin):
    return any(_is_llm_type(arg) for arg in args)

  return False


class _AgentConfigMapper:
  """Maps YAML data to Agent and Workflow class fields dynamically."""

  def __init__(self, abs_path: str):
    self.abs_path = abs_path
    # Values `map()` resolved that the target's `__init__` cannot accept, but
    # which are real fields on it. Applied after construction by `from_config`
    # rather than dropped, so a key the YAML sets is never silently ignored.
    self.deferred_fields: dict[str, Any] = {}
    # Nodes are cached by name and by the reference that produced them, so a
    # node named once in a chain is the same object when a later edge refers
    # to it again -- which is what keeps START and shared sub-graphs
    # identity-stable across edges.
    self._resolved_nodes_cache: dict[Any, Any] = {}

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
        _validate_module_reference(tool_config.name)
        module_path, obj_name = tool_config.name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)

      if isinstance(obj, BaseTool) or isinstance(obj, BaseToolset):
        resolved_tools.append(obj)
      elif inspect.isclass(obj) and (
          issubclass(obj, BaseTool) or issubclass(obj, BaseToolset)
      ):
        from ..tools.tool_configs import ToolArgsConfig

        resolved_tools.append(
            obj.from_config(tool_config.args or ToolArgsConfig(), self.abs_path)
        )
      elif callable(obj):
        if tool_config.args:
          resolved_tools.append(obj(tool_config.args))
        else:
          resolved_tools.append(obj)
      else:
        raise ValueError(f"Invalid tool YAML config: {tool_config}.")

    return resolved_tools

  def _resolve_edges(self, value: list[Any]) -> list[Any]:
    """Resolve edges to support agent references and graph chains."""
    from ..workflow._graph import Edge

    processed_edges: List[Any] = []
    for edge_item in value:
      if isinstance(edge_item, list):
        # A chain of elements in YAML: [START, node1, node2, ...] -> tuple(...)
        processed_chain = []
        for element in edge_item:
          processed_chain.append(self._resolve_chain_element(element))
        processed_edges.append(tuple(processed_chain))
      elif isinstance(edge_item, tuple):
        processed_chain = []
        for element in edge_item:
          processed_chain.append(self._resolve_chain_element(element))
        processed_edges.append(tuple(processed_chain))
      elif isinstance(edge_item, dict):
        edge_fields = Edge.model_fields
        if (
            all(k in edge_fields for k in edge_item.keys())
            and "from_node" in edge_item
            and "to_node" in edge_item
        ):
          from_node = self._resolve_node_like(edge_item["from_node"])
          to_node = self._resolve_node_like(edge_item["to_node"])
          route = edge_item.get("route")
          processed_edges.append(
              Edge(from_node=from_node, to_node=to_node, route=route)
          )
        else:
          # Assume RoutingMap or NodeLike
          processed_edges.append(self._resolve_chain_element(edge_item))
      elif isinstance(edge_item, Edge):
        processed_edges.append(edge_item)
      else:
        processed_edges.append(edge_item)
    return processed_edges

  def _resolve_chain_element(self, element: Any) -> Any:
    """Resolve a chain element in an edge."""
    if isinstance(element, list):
      # Fan-out in a chain: [node1, node2] -> tuple(node1, node2)
      return tuple(self._resolve_node_like(e) for e in element)
    elif isinstance(element, tuple):
      return tuple(self._resolve_node_like(e) for e in element)
    elif isinstance(element, dict):
      if self._looks_like_a_node(element):
        return self._resolve_node_like(element)
      else:
        # Assume RoutingMap: {route_key: destination}
        processed_map = {}
        for k, v in element.items():
          if isinstance(v, (list, tuple)):
            processed_map[k] = tuple(self._resolve_node_like(e) for e in v)
          else:
            processed_map[k] = self._resolve_node_like(v)
        return processed_map
    else:
      return self._resolve_node_like(element)

  def _looks_like_a_node(self, element: dict[str, Any]) -> bool:
    """Decides whether a mapping in a chain is an inline node or a routing map.

    `agent_class`, `config_path` and `func_code` only ever name a node. `name`
    is ambiguous -- it is how an inline node is spelled, but it is also a
    perfectly good route value -- so it counts only when every other key is a
    field an agent actually has. That keeps `{name: a, other: b}` a routing map
    instead of an LlmAgent missing its required keys.
    """
    from .llm_agent import LlmAgent

    if any(k in element for k in ("agent_class", "config_path", "func_code")):
      return True
    # `{code: "my.module.agent"}` is an AgentRefConfig. A single-entry routing
    # map keyed `code` looks identical until you check the value: a reference
    # is a string, a route destination is a node.
    if len(element) == 1 and isinstance(element.get("code"), str):
      return True
    return "name" in element and all(
        _names_a_field(key, LlmAgent.model_fields) for key in element
    )

  def _resolve_node_like(self, node_like: Any) -> Any:
    """Resolve a NodeLike item, handling agent references and FunctionNodes."""
    from ..workflow._base_node import BaseNode
    from ..workflow._base_node import START
    from ..workflow._function_node import FunctionNode

    if node_like is START or node_like == "START":
      return START

    if isinstance(node_like, BaseNode):
      if node_like.name:
        self._resolved_nodes_cache[node_like.name] = node_like
      return node_like

    if isinstance(node_like, str):
      if node_like in self._resolved_nodes_cache:
        return self._resolved_nodes_cache[node_like]

      if node_like.endswith(".yaml") or node_like.endswith(".yml"):
        ref = AgentRefConfig(config_path=node_like)
        resolved = resolve_agent_reference(ref, self.abs_path)
        self._resolved_nodes_cache[node_like] = resolved
        if hasattr(resolved, "name") and resolved.name:
          self._resolved_nodes_cache[resolved.name] = resolved
        return resolved
      else:
        if "." not in node_like:
          # A bare word is a node name, not a code path. Reaching here means no
          # earlier edge defined it: forward references are not supported, and
          # reporting this as a bad module path sends the reader looking in the
          # wrong place entirely.
          raise ValueError(
              f"Unknown node {node_like!r}. A node has to be defined by an"
              " earlier edge before another edge can name it."
          )
        # Check if it's a function reference or module reference
        func_path = node_like
        if func_path.startswith("."):
          dir_path = os.path.dirname(self.abs_path)
          pkg_name = os.path.basename(dir_path)
          func_path = pkg_name + func_path

        func = resolve_fully_qualified_name(func_path)
        if callable(func) and not inspect.isclass(func):
          node_name = func_path.rsplit(".", 1)[-1]
          resolved = FunctionNode(name=node_name, func=func)
        elif isinstance(func, BaseNode):
          resolved = func
        else:
          # Mirrors `_resolve_agent_code_reference`: refusing here names the
          # offending reference, where accepting it surfaces much later as a
          # graph error that does not mention the config at all.
          raise ValueError(
              f"Invalid node reference {node_like!r}: resolved to a"
              f" {type(func).__name__}, which is neither a callable nor a node."
          )

        self._resolved_nodes_cache[node_like] = resolved
        if hasattr(resolved, "name") and resolved.name:
          self._resolved_nodes_cache[resolved.name] = resolved
        return resolved

    elif isinstance(node_like, dict):
      # Keyed by identity because an inline node definition has no name to key
      # on until it is built. Sound only because the parsed YAML that owns
      # these dicts outlives the mapper: `from_config` holds `config_data`
      # across the whole `map()` call, so no id can be recycled mid-load.
      node_id = id(node_like)
      if node_id in self._resolved_nodes_cache:
        return self._resolved_nodes_cache[node_id]
      if (
          "name" in node_like
          and node_like["name"] in self._resolved_nodes_cache
      ):
        return self._resolved_nodes_cache[node_like["name"]]

      if "config_path" in node_like or (
          "code" in node_like
          and "agent_class" not in node_like
          and "func_code" not in node_like
      ):
        ref = AgentRefConfig(**node_like)
        resolved = resolve_agent_reference(ref, self.abs_path)
        self._resolved_nodes_cache[node_id] = resolved
        if "config_path" in node_like:
          self._resolved_nodes_cache[node_like["config_path"]] = resolved
        if hasattr(resolved, "name") and resolved.name:
          self._resolved_nodes_cache[resolved.name] = resolved
        return resolved

      cls_name = node_like.get("agent_class", "LlmAgent")
      if not isinstance(cls_name, str):
        raise ValueError(f"agent_class must be a string, got {type(cls_name)}")
      cls = _resolve_agent_class(cls_name)

      # No FunctionNode-shaped branch here: `map` already resolves `func_code`
      # like any other `*_code` key, onto the `func` constructor parameter and
      # with the same leading-dot handling. Going through it also means the
      # remaining keys get their fields resolved instead of passed in raw, and
      # that a FunctionNode *subclass* named in `agent_class` is the class that
      # actually gets built.
      resolved = self._build(cls, node_like)

      self._resolved_nodes_cache[node_id] = resolved
      if hasattr(resolved, "name") and resolved.name:
        self._resolved_nodes_cache[resolved.name] = resolved
      if "name" in node_like:
        self._resolved_nodes_cache[node_like["name"]] = resolved
      return resolved

    return node_like

  def map(
      self,
      data: dict[str, Any],
      agent_class: Optional[type[Any]] = None,
  ) -> dict[str, Any]:
    """Map configuration dictionary to constructor keyword arguments."""
    # Reset first: `deferred_fields` is per-call state kept on the mapper, and
    # resolving an inline node re-enters `map`, so a stale set from a nested
    # call must not leak out as this one's.
    self.deferred_fields = {}
    cls = agent_class

    if not cls:
      cls_name = data.get("agent_class", "LlmAgent")
      cls = _resolve_agent_class(cls_name)

    fields = cls.model_fields if issubclass(cls, BaseModel) else {}
    valid_fields = set(fields.keys())
    # Named constructor parameters that are not model fields, e.g.
    # FunctionNode.func. `self` and a `**kwargs` catch-all are excluded: on a
    # pydantic class the signature is `(self, **data)`, so taking every
    # parameter name would make YAML keys called `self` or `data` look valid
    # and fail later with an error naming neither the key nor the class.
    named_params: set[str] = set()
    takes_kwargs = True
    if inspect.isclass(cls):
      params = inspect.signature(cls.__init__).parameters
      named_params = {
          name
          for name, param in params.items()
          if name != "self" and param.kind is not inspect.Parameter.VAR_KEYWORD
      }
      takes_kwargs = any(
          param.kind is inspect.Parameter.VAR_KEYWORD
          for param in params.values()
      )
      valid_fields.update(named_params)
    kwargs = {}

    # Validate against the class's own config schema first. Mapping by
    # reflection reads the keys it recognises and ignores the rest, so without
    # this a key the user misspelled is dropped in silence and yields an agent
    # missing whatever it was meant to set. The schema decides how strict that
    # is: the built-in configs are `extra='forbid'`, while `BaseAgentConfig`
    # is `extra='allow'`, which is what lets a custom agent class carry its
    # own extra keys.
    config_schema = getattr(cls, "config_type", None)
    if not (
        isinstance(config_schema, type) and issubclass(config_schema, BaseModel)
    ):
      config_schema = None
    if config_schema is not None:
      config_schema.model_validate(data)

    unknown_keys = []
    for name, value in data.items():
      if name == "agent_class":
        continue

      target_name = name
      is_code_ref = False

      if name.endswith("_code"):
        base_name = name.removesuffix("_code")
        if base_name in valid_fields:
          target_name = base_name
          is_code_ref = True
      elif name.endswith("_callbacks"):
        singular_name = name.removesuffix("s")
        if singular_name in valid_fields:
          target_name = singular_name

      if target_name in valid_fields:
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
        unknown_keys.append(name)

    if unknown_keys and config_schema is None:
      # The node classes carry no config schema, so nothing above rejected a
      # misspelled key. Reflection would just skip it and hand back a node
      # quietly missing whatever it was meant to set.
      logger.warning(
          "%s does not accept %s; ignoring.", cls.__name__, sorted(unknown_keys)
      )

    if not takes_kwargs:
      # The constructor spells its parameters out -- `FunctionNode` is the
      # built-in example -- so passing a field it does not name is a TypeError.
      # Several of those are still real fields on the node (`description`,
      # which FunctionNode otherwise derives from the function docstring).
      # Hand them to the caller to set after construction instead of
      # discarding them, or a `description:` in the YAML would read as
      # accepted and do nothing.
      #
      # Everything that got this far passed `valid_fields`, so a kwarg the
      # signature does not name is a model field by construction: deferred,
      # never discarded. Keys reflection could not place at all were reported
      # above.
      self.deferred_fields = {
          k: v for k, v in kwargs.items() if k not in named_params
      }
      kwargs = {k: v for k, v in kwargs.items() if k in named_params}

    return kwargs

  def _build(self, cls: type[Any], data: dict[str, Any]) -> Any:
    """Maps `data` onto `cls`, constructs it, and applies deferred fields.

    `deferred_fields` is consumed here rather than left on the mapper, because
    the next `map` call -- including one nested inside this construction --
    would otherwise overwrite it before the caller looked.
    """
    from .base_agent import BaseAgent

    # A subclass owning its own construction owns it here as well. Reaching it
    # only through `config_path` and not through an inline mapping would make
    # the two spellings of the same node behave differently.
    if (
        inspect.isclass(cls)
        and issubclass(cls, BaseAgent)
        and _underlying(cls.from_config)
        is not _underlying(BaseAgent.from_config)
    ):
      return cls.from_config(_config_as_model(cls, data), self.abs_path)

    kwargs = self.map(data, cls)
    deferred = self.deferred_fields
    self.deferred_fields = {}
    node = cls(**kwargs)
    for field_name, field_value in deferred.items():
      node.__pydantic_validator__.validate_assignment(
          node, field_name, field_value
      )
    return node

  def _map_field(self, name: str, value: Any, fields: dict[str, Any]) -> Any:
    """Map a specific field value based on its Pydantic type annotation."""
    field = fields.get(name)
    if not field:
      return value
    annotation = field.annotation

    # Rule 1: Workflow Edges
    if _is_workflow_edges_type(annotation) and isinstance(value, (list, tuple)):
      return self._resolve_edges(list(value))

    # Rule 2: Sub Agents
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

    # Rule 3: Tools
    if _is_tools_type(annotation) and isinstance(value, list):
      tool_configs = [
          ToolConfig(**v) if isinstance(v, dict) else ToolConfig(name=v)
          for v in value
      ]
      return self._resolve_tools(tool_configs)

    # Rule 4: Callback
    if _is_callback_type(annotation):
      if isinstance(value, list):
        return resolve_callbacks([
            CodeConfig(**v) if isinstance(v, dict) else CodeConfig(name=v)
            for v in value
        ])
      elif isinstance(value, dict):
        return resolve_code_reference(CodeConfig(**value))
      elif isinstance(value, str):
        return resolve_code_reference(CodeConfig(name=value))

    # Rule 5: Schemas
    if _is_schema_type(annotation):
      # A schema field can hold a JSON schema outright -- `BaseNode`'s is a
      # `SchemaUnion`, whose first member is `dict` -- so only a mapping shaped
      # like a `CodeConfig` is treated as a reference to import. Anything else
      # is the schema itself and goes through untouched.
      if isinstance(value, dict) and "name" in value:
        return resolve_code_reference(CodeConfig(**value))
      elif isinstance(value, str):
        return resolve_code_reference(CodeConfig(name=value))

    # Rule 6: LLM (Legacy model mapping or custom LLM)
    if _is_llm_type(annotation) and isinstance(value, dict) and "name" in value:
      return resolve_code_reference(CodeConfig(**value))

    return value


def _names_a_field(key: str, fields: dict[str, Any]) -> bool:
  """Whether a YAML key addresses a field, directly or through a suffix.

  `model_code` and `before_agent_callbacks` are spellings of `model` and
  `before_agent_callback`; neither appears in `model_fields`, so a check
  against that alone would not recognise them.
  """
  if key in fields:
    return True
  if key.endswith("_code") and key.removesuffix("_code") in fields:
    return True
  if key.endswith("_callbacks") and key.removesuffix("s") in fields:
    return True
  return False


def _underlying(method: Any) -> Any:
  """Returns the plain function behind a bound classmethod.

  Bound classmethods compare by `__self__` as well, so two subclasses sharing
  one inherited implementation still compare unequal; the functions do not.
  """
  return getattr(method, "__func__", method)


def _config_as_model(
    agent_class: type[Any], config_data: dict[str, Any]
) -> Any:
  """Returns `config_data` validated into the class's declared config type.

  The `from_config` and `_parse_config` hooks were always handed a parsed
  config model rather than the raw mapping, so both call sites go through here
  to keep that contract. A class without a pydantic `config_type` gets the
  mapping unchanged.
  """
  config_type = getattr(agent_class, "config_type", None)
  if isinstance(config_type, type) and issubclass(config_type, BaseModel):
    return config_type.model_validate(config_data)
  return config_data


@experimental(FeatureName.AGENT_CONFIG)
def from_config(config_path: str) -> BaseNode:
  """Build agent or workflow node from a YAML config file path.

  Args:
    config_path: the path to a YAML config file.

  Returns:
    The created agent or workflow instance.

  Raises:
    FileNotFoundError: If config file doesn't exist.
    ValidationError: If config file's content is invalid YAML or schema.
    ValueError: If agent type is unsupported.
  """
  abs_path = os.path.abspath(config_path)
  if not os.path.exists(abs_path):
    raise FileNotFoundError(f"Config file not found: {abs_path}")

  with open(abs_path, "r", encoding="utf-8") as f:
    config_data = yaml.safe_load(f)

  if config_data is None:
    raise ValueError(f"Invalid agent config in {abs_path!r}. File is empty.")
  elif not isinstance(config_data, dict):
    raise ValueError(
        f"Invalid agent config in {abs_path!r}. Expected a dictionary."
    )

  if _ENFORCE_YAML_KEY_DENYLIST:
    _check_config_for_blocked_keys(config_data, abs_path)

  agent_class_name = config_data.get("agent_class", "LlmAgent")
  agent_class = _resolve_agent_class(agent_class_name)

  # A subclass that overrides `from_config` owns its own construction, so
  # delegate rather than building the class here -- otherwise the override,
  # and the `_parse_config` hook it reaches, would never run.
  from .base_agent import BaseAgent

  if issubclass(agent_class, BaseAgent) and _underlying(
      agent_class.from_config
  ) is not _underlying(BaseAgent.from_config):
    return agent_class.from_config(
        _config_as_model(agent_class, config_data), abs_path
    )

  mapper = _AgentConfigMapper(abs_path)
  kwargs = mapper.map(config_data, agent_class)

  # The mapper replaced the per-class parsers, not the `_parse_config` hook, so
  # a subclass overriding it still gets the last word on its kwargs. Invoking it
  # here is what keeps those subclasses working: `BaseAgent.from_config`, which
  # used to call it, is no longer on this path.
  if issubclass(agent_class, BaseAgent) and _underlying(
      agent_class._parse_config  # pylint: disable=protected-access
  ) is not _underlying(BaseAgent._parse_config):
    # The hook's own `@deprecated` cannot reach an override -- the override is
    # a different, undecorated function -- so the warning is raised here, where
    # the overriding class is known.
    warnings.warn(
        f"{agent_class.__name__}._parse_config is deprecated and will be"
        " removed in a future version. Declare the fields directly on the"
        " class instead; the loader maps them onto it by reflection.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs = agent_class._parse_config(  # pylint: disable=protected-access
        _config_as_model(agent_class, config_data), abs_path, kwargs
    )

  # `_resolve_agent_class` is typed `type[Any]` because the class is named at
  # runtime, so the instance needs narrowing back to the declared return type.
  node: BaseNode = agent_class(**kwargs)
  for field_name, field_value in mapper.deferred_fields.items():
    # Validated rather than `setattr`: these fields were held back from the
    # constructor, so this is the only place their declared type is enforced
    # at all. A plain assignment would let a `description:` holding a list
    # settle onto a field declared `str`.
    node.__pydantic_validator__.validate_assignment(
        node, field_name, field_value
    )
  return node


def _resolve_agent_class(agent_class: str) -> type[Any]:
  """Resolve the agent class from its fully qualified name or shorthand."""
  from ..workflow._base_node import BaseNode

  agent_class_name = agent_class or "LlmAgent"
  if "." not in agent_class_name:
    # Probe the built-in packages directly rather than through
    # `resolve_fully_qualified_name`, which funnels every failure into
    # ValueError and so cannot distinguish "the shorthand is not in this
    # package" from "this package failed to import". Only the first should
    # fall through to the next candidate; the second is the user's own broken
    # module and must surface as itself. The package names are literals, so
    # no denylist check is needed here.
    for package in ("google.adk.agents", "google.adk.workflow"):
      try:
        cls = getattr(importlib.import_module(package), agent_class_name)
      except AttributeError:
        continue
      if inspect.isclass(cls) and issubclass(cls, BaseNode):
        return cls

  agent_class_obj = resolve_fully_qualified_name(agent_class_name)
  if inspect.isclass(agent_class_obj) and issubclass(agent_class_obj, BaseNode):
    return agent_class_obj

  raise ValueError(
      f"Invalid class `{agent_class_name}`. It must be a subclass of BaseNode."
  )


_BLOCKED_YAML_KEYS = frozenset({"args"})
_ENFORCE_YAML_KEY_DENYLIST = False


def _set_enforce_yaml_key_denylist(value: bool) -> None:
  global _ENFORCE_YAML_KEY_DENYLIST
  _ENFORCE_YAML_KEY_DENYLIST = value


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


_ENFORCE_DENYLIST = True

# Agent configs never need the standard library: they name the agent's own
# package, google.adk, or a third-party integration. So block all of it. Listing
# only the scary modules does not work, because cProfile.run, timeit.timeit and
# trace.Trace.run all execute a string you hand them, and each Python release
# can add more.
_STDLIB_MODULES = frozenset(sys.stdlib_module_names) | frozenset(
    sys.builtin_module_names  # Redundant on stock CPython, not custom builds.
)

# Extra names to block. Everything above the LOAD-BEARING line below is already
# covered by _STDLIB_MODULES and is kept only to spell out the threat model.
_BLOCKED_MODULES = frozenset({
    # Process / OS execution
    "os",
    "posix",  # Unix alias: posix.system is os.system
    "nt",  # Windows alias: nt.system is os.system
    "subprocess",
    "_posixsubprocess",
    "sys",
    "builtins",
    "importlib",
    "shutil",
    "signal",
    "multiprocessing",
    "threading",
    # Dynamic code evaluation
    "code",
    "codeop",
    "compileall",
    "runpy",
    # Native / unsafe extensions
    "ctypes",
    # Network access
    "socket",
    "_socket",
    "http",
    "urllib",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "xmlrpc",
    "asyncio",
    # Filesystem / serialisation
    "tempfile",
    "pathlib",
    "shelve",
    "pickle",
    "marshal",
    # Interactive / side-effect modules
    "webbrowser",
    "antigravity",
    "pty",
    "pdb",
    "profile",
    # LOAD-BEARING, keep these. They are not in sys.stdlib_module_names on
    # every Python we support, so this set is all that blocks them.
    #
    # Modules dropped from the standard library that you can still import:
    # distutils comes back through setuptools' shim and its spawn() runs a
    # subprocess, and the rest have "standard-*" packages on PyPI. commands is
    # a Python 2 leftover.
    "asynchat",
    "asyncore",
    "cgi",
    "commands",
    "crypt",
    "distutils",
    "imp",
    "mailcap",
    "nntplib",
    "pipes",
    "smtpd",
    "telnetlib",
    "uu",
    # CPython's own test packages, which most installs ship. They can start a
    # subprocess (test.support.script_helper) and execute source (_testcapi).
    "_testcapi",
    "_testinternalcapi",
    "test",
    # Hard, always-installed third-party dependencies of adk-python itself
    # (or common transitive dependencies) that ship exec-capable
    # deserialization entry points. A denylist still cannot cover third-party
    # packages in general (the loader resolves them by name, and any of the
    # many packages an integration might install could have its own gadget),
    # but these are common enough that they are blocked outright rather than
    # left to the general third-party gap.
    #
    # yaml.unsafe_load (and yaml.load without an explicit safe Loader, and
    # yaml.full_load) and ruamel.yaml equivalents construct arbitrary Python
    # objects from the YAML document they are given, via tags such as
    # !!python/object/apply:os.system. A reference to any of these as a
    # code-reference field's target -- with the YAML content supplied as
    # the function's argument at call time -- is a direct RCE primitive
    # requiring no other preconditions.
    "ruamel",
    "yaml",
})


def _validate_module_reference(fully_qualified_name: str) -> None:
  """Validate that a module reference does not target a blocked module.

  Args:
    fully_qualified_name: The fully-qualified Python name to validate (e.g.
      ``"my_package.my_module.my_func"``).

  Raises:
    ValueError: If the top-level module is part of the Python standard library
      or is in ``_BLOCKED_MODULES``.
  """
  if not _ENFORCE_DENYLIST:
    return
  # Extract the top-level package from the fully-qualified name.
  top_module = fully_qualified_name.split(".")[0]
  if top_module in _BLOCKED_MODULES or top_module in _STDLIB_MODULES:
    raise ValueError(
        f"Blocked module reference: {fully_qualified_name!r}. Agent "
        f"configurations cannot import from '{top_module}'. The Python "
        "standard library is blocked in full because too much of it can "
        "execute arbitrary code. Reference your own agent package, "
        "'google.adk', or a third-party package instead."
    )


def _set_enforce_denylist(value: bool) -> None:
  global _ENFORCE_DENYLIST
  _ENFORCE_DENYLIST = value


@experimental(FeatureName.AGENT_CONFIG)
def resolve_fully_qualified_name(name: str) -> Any:
  try:
    module_path, obj_name = name.rsplit(".", 1)
    _validate_module_reference(name)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
  except Exception as e:
    raise ValueError(f"Invalid fully qualified name: {name}") from e


@experimental(FeatureName.AGENT_CONFIG)
def resolve_agent_reference(
    ref_config: AgentRefConfig, referencing_agent_config_abs_path: str
) -> BaseNode:
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
      raise ValueError(
          "Absolute paths are not allowed in AgentRefConfig config_path:"
          f" {ref_config.config_path!r}"
      )
    agent_dir = os.path.dirname(referencing_agent_config_abs_path)
    resolved_path = os.path.realpath(
        os.path.join(agent_dir, ref_config.config_path)
    )
    canonical_agent_dir = os.path.realpath(agent_dir)
    if (
        os.path.commonpath([canonical_agent_dir, resolved_path])
        != canonical_agent_dir
    ):
      raise ValueError(
          f"Path traversal detected: config_path {ref_config.config_path!r}"
          " resolves outside the agent directory"
      )
    return from_config(resolved_path)
  elif ref_config.code:
    return _resolve_agent_code_reference(ref_config.code)
  else:
    raise ValueError("AgentRefConfig must have either 'code' or 'config_path'")


def _resolve_agent_code_reference(code: str) -> BaseNode:
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

  _validate_module_reference(code)
  module_path, obj_name = code.rsplit(".", 1)
  module = importlib.import_module(module_path)
  obj = getattr(module, obj_name)

  from ..workflow._base_node import BaseNode

  # A class is callable but is not a BaseNode *instance*, so it is rejected
  # either way; the branches only exist to say which mistake was made.
  if inspect.isclass(obj):
    raise ValueError(
        f"Invalid agent reference to a class: {code}. Reference an agent"
        " instance, or name the class in `agent_class` instead."
    )

  if callable(obj):
    raise ValueError(f"Invalid agent reference to a callable: {code}")

  if not isinstance(obj, BaseNode):
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

  _validate_module_reference(code_config.name)
  if "." not in code_config.name:
    raise ValueError(f"Invalid code reference: {code_config.name}")
  module_path, obj_name = code_config.name.rsplit(".", 1)
  try:
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)
  except ModuleNotFoundError as e:
    # The docstring has always promised ValueError for a reference that does
    # not resolve, and callers rely on it -- including agent classes that
    # resolved their own code references before the mapper took this over.
    #
    # `e.name` is the module Python could not find. Only rewrite the error when
    # that is the module the reference named: if the reference resolved and it
    # is the user's own module that imports something missing, reporting an
    # "invalid name" would send them looking in the wrong place.
    if (
        e.name is None
        or module_path == e.name
        or module_path.startswith(f"{e.name}.")
    ):
      raise ValueError(
          f"Invalid fully qualified name: {code_config.name}"
      ) from e
    raise
  except AttributeError as e:
    # The module imported cleanly but holds no such symbol.
    raise ValueError(f"Invalid fully qualified name: {code_config.name}") from e


@experimental(FeatureName.AGENT_CONFIG)
def resolve_callbacks(callbacks_config: List[CodeConfig]) -> Any:
  """Resolve callbacks from configuration.

  Args:
    callbacks_config: List of callback configurations (CodeConfig objects).

  Returns:
    List of resolved callback objects.
  """
  return [resolve_code_reference(config) for config in callbacks_config]
