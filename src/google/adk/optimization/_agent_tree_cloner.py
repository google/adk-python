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

"""Clones an ADK agent hierarchy with dotted parameter overrides applied."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any
from typing import TypeAlias

from google.genai import types
from pydantic import BaseModel

from ..agents.base_agent import BaseAgent
from ..flows.llm_flows.core._utils import copy_http_options
from ..planners.built_in_planner import BuiltInPlanner
from ..tools.agent_tool import AgentTool

_ChildParams: TypeAlias = dict[str, dict[str, Any]]


def _mode_tool_ids(agent: BaseAgent) -> set[int]:
  """Returns IDs of tools auto-built for single_turn or task modes."""
  return {id(tool) for tool in getattr(agent, "_constructed_mode_tools", ())}


def _copy_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
  """Recursively copies dicts without touching non-mapping leaf objects."""
  return {
      k: _copy_mapping(v) if isinstance(v, Mapping) else v
      for k, v in mapping.items()
  }


def _partition_agent_params(
    agent: BaseAgent,
    flat_params: Mapping[str, Any],
) -> tuple[dict[str, Any], _ChildParams, _ChildParams]:
  """Routes canonical parameter paths to `agent`, `sub_agents`, or `tools`."""
  fields = set(type(agent).model_fields)
  constructed_ids = _mode_tool_ids(agent)
  raw_subs = flat_params.get("sub_agents", agent.sub_agents) or []
  sub_children = {
      child.name: child for child in raw_subs if isinstance(child, BaseAgent)
  }
  sub_ids = {id(child) for child in sub_children.values()}
  raw_tools = flat_params.get("tools", getattr(agent, "tools", None)) or []
  tool_children = {
      tool.agent.name: tool.agent
      for tool in raw_tools
      if isinstance(tool, AgentTool)
      and id(tool) not in constructed_ids
      and id(tool.agent) not in sub_ids
      and isinstance(tool.agent, BaseAgent)
  }

  direct: dict[str, Any] = {}
  sub_params: dict[str, dict[str, Any]] = {}
  tool_params: dict[str, dict[str, Any]] = {}
  for key, value in flat_params.items():
    if key.startswith("sub_agents."):
      child_name, _, rest = key.removeprefix("sub_agents.").partition(".")
      if child_name not in sub_children:
        raise ValueError(
            f"Parameter {key!r}: agent {agent.name!r} has no sub-agent named"
            f" {child_name!r}. Sub-agents: {sorted(sub_children) or 'none'}."
        )
      if not rest:
        raise ValueError(
            f"Parameter {key!r} names sub-agent {child_name!r} but no field."
        )
      sub_params.setdefault(child_name, {})[rest] = value
    elif key.startswith("tools."):
      child_name, _, rest = key.removeprefix("tools.").partition(".")
      if child_name not in tool_children:
        hint = (
            f" ({child_name!r} is in sub_agents; use"
            f" 'sub_agents.{child_name}.{rest}'.)"
            if child_name in sub_children and rest
            else ""
        )
        raise ValueError(
            f"Parameter {key!r}: agent {agent.name!r} has no AgentTool agent"
            f" named {child_name!r}.{hint} AgentTools:"
            f" {sorted(tool_children) or 'none'}."
        )
      if not rest:
        raise ValueError(
            f"Parameter {key!r} names AgentTool {child_name!r} but no field."
        )
      tool_params.setdefault(child_name, {})[rest] = value
    elif key.startswith("generate_content_config."):
      if "generate_content_config" not in fields:
        raise ValueError(
            f"Parameter {key!r}: {type(agent).__name__} ({agent.name!r}) has"
            " no generate_content_config field."
        )
      if not key.removeprefix("generate_content_config."):
        raise ValueError(f"Parameter {key!r} names no config field.")
      direct[key] = value
    elif "." in key:
      head, _, _ = key.partition(".")
      if head in fields:
        raise ValueError(
            f"Parameter {key!r} sets a field inside {head!r}; only"
            " 'generate_content_config.<field>', 'sub_agents.<Name>.<field>',"
            " and 'tools.<Name>.<field>' support dotted paths."
        )
      raise ValueError(
          f"Parameter {key!r} is not valid on {agent.name!r}. Prefix child"
          " agents with 'sub_agents.' or 'tools.' and generation settings"
          " with 'generate_content_config.'."
      )
    elif key not in fields:
      raise ValueError(
          f"Parameter {key!r} is not a field of {type(agent).__name__}"
          f" ({agent.name!r}); use 'generate_content_config.{key}' for"
          " generation settings."
      )
    else:
      direct[key] = value
  return direct, sub_params, tool_params


def _expand_dotted_mapping(flat_params: Mapping[str, Any]) -> dict[str, Any]:
  """Expands dotted `GenerateContentConfig` sub-paths into nested dicts."""
  nested: dict[str, Any] = {}
  leaf_paths: set[str] = set()
  for key in sorted(flat_params, key=lambda path: path.count(".")):
    cursor = nested
    parts = key.split(".")
    for depth, part in enumerate(parts[:-1]):
      prefix = ".".join(parts[: depth + 1])
      if prefix in leaf_paths:
        raise ValueError(
            f"Parameter 'generate_content_config.{key}' sets a field inside"
            f" 'generate_content_config.{prefix}', which the grid also sets"
            " as a whole."
        )
      cursor = cursor.setdefault(part, {})
    value = flat_params[key]
    cursor[parts[-1]] = (
        _copy_mapping(value) if isinstance(value, Mapping) else value
    )
    leaf_paths.add(key)
  return nested


def _resolve_field_name(model_cls: type[BaseModel], key: str) -> str:
  """Maps a Pydantic field alias (such as camelCase) to its field name."""
  if key in model_cls.model_fields:
    return key
  for name, field_info in model_cls.model_fields.items():
    if field_info.alias == key:
      return name
  return key


def _merge_nested_value(current: Any, override: Any, prefix: str) -> Any:
  """Recursively merges a mapping override onto a BaseModel or Mapping."""
  if not isinstance(override, Mapping):
    return override
  if isinstance(current, BaseModel):
    model_cls = type(current)
    merged = {name: getattr(current, name) for name in current.model_fields_set}
    seen: set[str] = set()
    for raw_key, sub_value in override.items():
      key = _resolve_field_name(model_cls, raw_key)
      if key in seen:
        raise ValueError(
            f"Duplicate override for 'generate_content_config.{prefix}{key}'"
            " via both field name and alias."
        )
      seen.add(key)
      merged[key] = _merge_nested_value(
          merged.get(key), sub_value, f"{prefix}{key}."
      )
    return merged
  if isinstance(current, Mapping):
    merged = dict(current)
    for key, sub_value in override.items():
      merged[key] = _merge_nested_value(
          merged.get(key), sub_value, f"{prefix}{key}."
      )
    return merged
  return _copy_mapping(override)


def _merge_http_options(
    current: Any, override: Any
) -> types.HttpOptions | None:
  """Merges an HttpOptions override while keeping live client objects."""
  if override is None or isinstance(override, types.HttpOptions):
    return override
  if not isinstance(override, Mapping):
    return types.GenerateContentConfig.model_validate(
        {"http_options": override}
    ).http_options
  if not isinstance(current, types.HttpOptions):
    return types.HttpOptions.model_validate(_copy_mapping(override))
  merged: dict[str, Any] = {}
  for raw_key, sub_value in override.items():
    key = _resolve_field_name(types.HttpOptions, raw_key)
    if key in merged:
      raise ValueError(
          "Duplicate override for"
          f" 'generate_content_config.http_options.{key}' via both field name"
          " and alias."
      )
    merged[key] = _merge_nested_value(
        getattr(current, key, None), sub_value, f"http_options.{key}."
    )
  validated = types.HttpOptions.model_validate(merged)
  http_updates = {
      name: getattr(validated, name) for name in validated.model_fields_set
  }
  return copy_http_options(current).model_copy(update=http_updates)


def _copy_generate_content_config(
    config: types.GenerateContentConfig,
    update: Mapping[str, Any] | None = None,
) -> types.GenerateContentConfig:
  """Deep-copies config while keeping live clients on http_options."""
  updates = dict(update) if update else {}
  has_http_update = "http_options" in updates
  raw_http = updates.pop("http_options", config.http_options)
  had_http_field = has_http_update or "http_options" in config.model_fields_set
  updates["http_options"] = None
  copied = config.model_copy(update=updates).model_copy(deep=True)
  if isinstance(raw_http, types.HttpOptions):
    copied.http_options = copy_http_options(raw_http)
  elif has_http_update:
    copied.http_options = raw_http
  elif not had_http_field:
    copied.model_fields_set.discard("http_options")
  return copied


def _validate_agent_config(
    agent: BaseAgent,
    config: types.GenerateContentConfig | None,
) -> None:
  """Runs the agent class's generate_content_config validator when present."""
  if config is None:
    return
  validator = getattr(type(agent), "validate_generate_content_config", None)
  if callable(validator):
    validator(config)


def _sync_builtin_planner(
    agent: BaseAgent,
    update_dict: dict[str, Any],
    final_config: types.GenerateContentConfig | None,
) -> None:
  """Keeps BuiltInPlanner.thinking_config aligned with a swept config."""
  planner = getattr(agent, "planner", None)
  if "planner" in update_dict or not isinstance(planner, BuiltInPlanner):
    return
  thinking = None if final_config is None else final_config.thinking_config
  update_dict["planner"] = (
      None
      if thinking is None
      else BuiltInPlanner(thinking_config=thinking.model_copy(deep=True))
  )


def _apply_generate_content_config(
    agent: BaseAgent,
    update_dict: dict[str, Any],
) -> None:
  """Applies whole or dotted `generate_content_config` updates on `agent`."""
  if "generate_content_config" not in type(agent).model_fields:
    return
  dotted = {
      key.removeprefix("generate_content_config."): update_dict.pop(key)
      for key in list(update_dict)
      if key.startswith("generate_content_config.")
  }
  has_whole = "generate_content_config" in update_dict
  if has_whole and dotted:
    raise ValueError(
        "Cannot set both 'generate_content_config' as a whole and"
        f" 'generate_content_config.{next(iter(dotted))}' in the same grid."
    )

  base = getattr(agent, "generate_content_config", None)
  config_cls = (
      type(base)
      if isinstance(base, types.GenerateContentConfig)
      else types.GenerateContentConfig
  )
  if has_whole:
    swept = update_dict["generate_content_config"]
    if swept is None:
      final_config = None
    elif isinstance(swept, types.GenerateContentConfig):
      final_config = _copy_generate_content_config(swept)
    else:
      payload = _copy_mapping(swept) if isinstance(swept, Mapping) else swept
      final_config = _copy_generate_content_config(
          config_cls.model_validate(payload)
      )
    _validate_agent_config(agent, final_config)
    update_dict["generate_content_config"] = final_config
    _sync_builtin_planner(agent, update_dict, final_config)
    return

  if not dotted:
    update_dict["generate_content_config"] = (
        None if base is None else _copy_generate_content_config(base)
    )
    return

  planner = getattr(agent, "planner", None)
  planner_tc = (
      planner.thinking_config if isinstance(planner, BuiltInPlanner) else None
  )
  updates = _expand_dotted_mapping(dotted)
  merged: dict[str, Any] = {}
  has_http_update = False
  merged_http: types.HttpOptions | None = None
  for raw_key, value in updates.items():
    key = _resolve_field_name(config_cls, raw_key)
    if key == "http_options":
      if has_http_update:
        raise ValueError(
            "Duplicate override for 'generate_content_config.http_options' via"
            " both field name and alias."
        )
      has_http_update = True
      merged_http = _merge_http_options(
          getattr(base, "http_options", None), value
      )
      continue
    if key in merged:
      raise ValueError(
          f"Duplicate override for 'generate_content_config.{key}' via both"
          " field name and alias."
      )
    current = getattr(base, key, None)
    if key == "thinking_config" and current is None:
      current = planner_tc
    merged[key] = _merge_nested_value(current, value, f"{key}.")

  seed = (
      {
          name: getattr(base, name)
          for name in base.model_fields_set
          if name != "http_options"
      }
      if isinstance(base, types.GenerateContentConfig)
      else {}
  )
  validated = config_cls.model_validate({**seed, **merged})
  config_updates: dict[str, Any] = {
      name: getattr(validated, name) for name in validated.model_fields_set
  }
  if has_http_update:
    config_updates["http_options"] = merged_http
  final_config = (
      _copy_generate_content_config(
          validated,
          update={"http_options": merged_http} if has_http_update else None,
      )
      if base is None
      else _copy_generate_content_config(base, update=config_updates)
  )
  _validate_agent_config(agent, final_config)
  update_dict["generate_content_config"] = final_config
  if "thinking_config" in merged:
    _sync_builtin_planner(agent, update_dict, final_config)


def _copy_agent_tool(tool: AgentTool, cloned_agent: BaseAgent) -> AgentTool:
  """Copies an AgentTool around `cloned_agent` and keeps metadata in sync."""
  orig_agent = tool.agent
  cloned_tool = copy.copy(tool)
  cloned_tool.agent = cloned_agent
  if cloned_tool.name == orig_agent.name:
    cloned_tool.name = cloned_agent.name
  if cloned_tool.description == orig_agent.description:
    cloned_tool.description = cloned_agent.description
  return cloned_tool


def _clone_agent_tool(tool: AgentTool, params: Mapping[str, Any]) -> AgentTool:
  """Copies an AgentTool around a clone of its wrapped agent."""
  cloned_child = clone_agent_with_params(tool.agent, params)
  # AgentTool runs its agent as a separate root, so it must have no parent.
  cloned_child.parent_agent = None
  return _copy_agent_tool(tool, cloned_child)


def clone_agent_with_params(
    agent: BaseAgent,
    flat_params: Mapping[str, Any],
) -> BaseAgent:
  """Clones `agent` and its descendants with `flat_params` applied."""
  update_dict, sub_params, tool_params = _partition_agent_params(
      agent, flat_params
  )
  raw_model = update_dict.get("model", getattr(agent, "model", None))
  if isinstance(raw_model, BaseModel):
    cloned_llm = raw_model.model_copy()
    if hasattr(cloned_llm, "_additional_args"):
      cloned_llm._additional_args = dict(cloned_llm._additional_args)
    update_dict["model"] = cloned_llm

  # Clone children even when `sub_agents` itself is swept: cloning re-parents
  # them, which must never touch the caller's agents or another candidate's.
  sub_agents = update_dict.get("sub_agents", agent.sub_agents)
  cloned_subs: dict[int, BaseAgent] = {}
  if sub_agents:
    cloned_list = [
        clone_agent_with_params(child, sub_params.get(child.name, {}))
        for child in sub_agents
    ]
    update_dict["sub_agents"] = cloned_list
    cloned_subs = {
        id(orig): cloned
        for orig, cloned in zip(sub_agents, cloned_list, strict=True)
    }

  tools = update_dict.get("tools", getattr(agent, "tools", None)) or []
  constructed_ids = _mode_tool_ids(agent)
  if any(
      isinstance(tool, AgentTool) and id(tool) not in constructed_ids
      for tool in tools
  ):
    new_tools: list[Any] = []
    for tool in tools:
      if not isinstance(tool, AgentTool) or id(tool) in constructed_ids:
        new_tools.append(tool)
      elif id(tool.agent) in cloned_subs:
        new_tools.append(_copy_agent_tool(tool, cloned_subs[id(tool.agent)]))
      elif isinstance(tool.agent, BaseAgent):
        new_tools.append(
            _clone_agent_tool(tool, tool_params.get(tool.agent.name, {}))
        )
      else:
        new_tools.append(tool)
    update_dict["tools"] = new_tools

  _apply_generate_content_config(agent, update_dict)
  planner = update_dict.get("planner", getattr(agent, "planner", None))
  if isinstance(planner, BuiltInPlanner):
    update_dict["planner"] = BuiltInPlanner(
        thinking_config=planner.thinking_config.model_copy(deep=True)
    )
  return agent.clone(update=update_dict)
