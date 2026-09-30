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

# pylint: disable=g-import-not-at-top,protected-access

"""Toolset for discovering, viewing, and executing agent skills."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
from enum import Enum
import hashlib
import json
import logging
import mimetypes
from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import Any
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING
import warnings

from google.genai import types
from typing_extensions import override

from ..agents.readonly_context import ReadonlyContext
from ..code_executors.base_code_executor import BaseCodeExecutor
from ..code_executors.code_execution_utils import CodeExecutionInput
from ..features import FeatureName
from ..features import is_feature_enabled
from ..skills import models
from ..skills import prompt
from ..skills import SkillRegistry
from ..telemetry import _hallucination
from ..telemetry import _instrumentation
from ..utils import instructions_utils
from .base_tool import BaseTool
from .base_toolset import BaseToolset
from .base_toolset import ToolPredicate
from .function_tool import FunctionTool
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..agents.llm_agent import ToolUnion
  from ..environment._base_environment import BaseEnvironment
  from ..models.llm_request import LlmRequest

logger = logging.getLogger("google_adk." + __name__)

_DEFAULT_SCRIPT_TIMEOUT = 300
_MAX_SKILL_PAYLOAD_BYTES = 16 * 1024 * 1024  # 16 MB

# Message used for the "Content Injection" pattern.
_BINARY_FILE_DETECTED_MSG = (
    "Binary file detected. The content has been injected into the"
    " conversation history for you to analyze."
)


_LIST_SKILLS_TOOL_NAME = "list_skills"
_SEARCH_SKILLS_TOOL_NAME = "search_skills"
_LOAD_SKILL_TOOL_NAME = "load_skill"
_UNLOAD_SKILL_TOOL_NAME = "unload_skill"
_LOAD_SKILL_RESOURCE_TOOL_NAME = "load_skill_resource"
_RUN_SKILL_SCRIPT_TOOL_NAME = "run_skill_script"

# Stands in for the instructions of a skill that is no longer active.
_UNLOADED_SKILL_STATUS = "unloaded"
_UNLOADED_SKILL_NOTICE = (
    "This skill has been unloaded. Its instructions no longer apply and the"
    " tools it contributed are no longer available. Load it again if you need"
    " them."
)


def _activated_skills_state_key(agent_name: str) -> str:
  """Returns the session state key holding an agent's activated skill names."""
  return f"_adk_activated_skill_{agent_name}"


def _read_activated_skills(state: Any, agent_name: str) -> list[str]:
  """Returns a mutable copy of an agent's activated skill names."""
  return list(state.get(_activated_skills_state_key(agent_name)) or [])


def _write_activated_skills(
    state: Any, agent_name: str, skill_names: list[str]
) -> None:
  """Stores an agent's activated skill names."""
  # Assign rather than mutate in place so the state delta is recorded.
  state[_activated_skills_state_key(agent_name)] = skill_names


_DEFAULT_MAX_ACTIVE_SKILLS = 5


class SkillLifecycleMode(Enum):
  """How long a skill stays active once loaded."""

  PERSISTENT = "persistent"
  """Stays active for the rest of the session unless explicitly unloaded.

  The default, and what every skill did before this setting existed. A
  persistent skill neither counts against `max_active_skills` nor is evicted
  by it.
  """

  BOUNDED = "bounded"
  """As PERSISTENT, but also subject to the `max_active_skills` cap.

  Loading a bounded skill past the cap releases the bounded skill that was
  least recently loaded. Reloading one counts as a use.
  """

  EPHEMERAL = "ephemeral"
  """Released once the turn that loaded it ends.

  Active for every model step of that turn, so the model can finish what it
  loaded the skill for, and gone from the next turn on. Loading it again buys
  another turn. Bounded by time, so the `max_active_skills` cap ignores it.

  Not supported under `run_live`, which runs the whole bidi stream as one
  invocation: with no turn boundary to expire on, an ephemeral skill there
  behaves as PERSISTENT and stays active until the stream ends.
  """


@dataclasses.dataclass(frozen=True)
class SkillLifecycleConfig:
  """How long a toolset's skills stay active once loaded.

  Attributes:
    enabled: False leaves every skill active for the whole session, whatever the
      rest of this says.
    default_mode: Lifecycle for any skill not in `skill_overrides`. PERSISTENT
      is how skills behaved before this config existed.
    max_active_skills: How many BOUNDED skills may be active at once, per agent.
      PERSISTENT and EPHEMERAL skills never count against it.
    skill_overrides: Per-skill lifecycles. Names need not be registered locally,
      so a registry skill can be listed here too.
    revalidate_skills: Recheck each request that the active skills still say
      what they did when loaded, and re-state the ones that changed. Off by
      default: a registry lookup per active skill per turn, and re-stating one
      invalidates the prompt cache.
  """

  enabled: bool = True
  default_mode: SkillLifecycleMode = SkillLifecycleMode.PERSISTENT
  max_active_skills: int = _DEFAULT_MAX_ACTIVE_SKILLS
  skill_overrides: dict[str, SkillLifecycleMode] = dataclasses.field(
      default_factory=dict
  )
  revalidate_skills: bool = False

  def __post_init__(self) -> None:
    if self.max_active_skills < 1:
      raise ValueError(
          "`max_active_skills` must be at least 1, got"
          f" {self.max_active_skills}."
      )


def _skill_lifecycle_state_key(agent_name: str) -> str:
  """Returns the session state key holding an agent's lifecycle records.

  Deliberately not under the `_adk_activated_skill_` prefix: consumers
  elsewhere scan for that prefix and read every match as a list of names.
  """
  return f"_adk_skill_meta_{agent_name}"


def _read_lifecycle_records(
    state: Any, agent_name: str
) -> dict[str, dict[str, Any]]:
  """Returns a mutable copy of an agent's per-skill lifecycle records."""
  records = state.get(_skill_lifecycle_state_key(agent_name))
  if not isinstance(records, dict):
    return {}
  return {
      name: dict(record)
      for name, record in records.items()
      if isinstance(record, dict)
  }


def _is_expired(record: dict[str, Any], invocation_id: str | None) -> bool:
  """Whether a lifecycle record has outlived the turn that created it.

  An invocation is a turn: every model step and tool call the user's message
  sets off shares its id. So an ephemeral skill is expired as soon as some
  other invocation asks. Except under `run_live`, where the whole stream is
  one invocation and nothing ever expires.

  A record with no id was activated without one, which only a hand-built
  context does. Left active, since guessing would release a skill in use.
  """
  if record.get("lifecycle") != SkillLifecycleMode.EPHEMERAL.value:
    return False
  activated_in = record.get("activated_in")
  return bool(activated_in) and activated_in != invocation_id


class SkillDiscoveryMode(Enum):
  """How the local skill catalog is disclosed to the model."""

  LAZY = "lazy"
  """The model discovers skills by calling `list_skills` (default).

  Costs a model turn before the first `load_skill`, and keeps the system
  instruction free of skill names. Preferable for a large or changing catalog.
  """

  EAGER = "eager"
  """The catalog is injected into the system instruction as XML.

  The `list_skills` tool is not offered, and the model can call `load_skill`
  straight away. Preferable for a small, stable catalog, where the discovery
  turn costs more than the names do. Registry skills are unaffected: they are
  still reachable only through `search_skills`.
  """


def _skill_content_hash(skill: models.Skill) -> str:
  """Returns a digest of everything a skill can tell the model to do.

  Instructions, frontmatter and every resource, so editing a reference or a
  script is a new version too. Names and lengths go into the hash as well, so
  moving content between two files does not hash the same.
  """
  hasher = hashlib.sha256()

  def _feed(*values: str | bytes) -> None:
    for value in values:
      data = value.encode("utf-8") if isinstance(value, str) else value
      hasher.update(len(data).to_bytes(8, "big"))
      hasher.update(data)

  _feed(skill.name, skill.instructions)
  _feed(
      json.dumps(
          skill.frontmatter.model_dump(mode="json"),
          sort_keys=True,
          default=str,
      )
  )
  resources = skill.resources
  for kind, names, get in (
      ("references", resources.list_references(), resources.get_reference),
      ("assets", resources.list_assets(), resources.get_asset),
  ):
    for name in sorted(names):
      content = get(name)
      if content is None:
        continue
      _feed(kind, name, content)
  for name in sorted(resources.list_scripts()):
    script = resources.get_script(name)
    if script is None or script.src is None:
      continue
    _feed("scripts", name, script.src)

  return hasher.hexdigest()


async def _write_skill_resources_to_env(
    skill: models.Skill, skill_dir: Path, env: BaseEnvironment
) -> None:
  """Writes every resource of a skill into its folder in the environment."""
  write_tasks = []

  def _write(*path_parts: str, content: str | bytes) -> None:
    write_tasks.append(
        env.write_file(
            cast(
                Path,
                PurePosixPath(skill_dir.joinpath(*path_parts).as_posix()),
            ),
            content,
        )
    )

  for ref_name in skill.resources.list_references():
    content = skill.resources.get_reference(ref_name)
    if content is not None:
      _write("references", ref_name, content=content)
  for asset_name in skill.resources.list_assets():
    content = skill.resources.get_asset(asset_name)
    if content is not None:
      _write("assets", asset_name, content=content)
  for scr_name in skill.resources.list_scripts():
    scr = skill.resources.get_script(scr_name)
    if scr is not None and scr.src is not None:
      _write("scripts", scr_name, content=scr.src)

  if write_tasks:
    await asyncio.gather(*write_tasks)


def _prune_unloaded_skill_instructions(
    contents: list[types.Content] | None,
    load_skill_tool_name: str,
    active_skills: set[str],
) -> list[str]:
  """Strips the instructions of unloaded skills out of a request's history.

  A `load_skill` response carries the whole SKILL.md body and stays in the
  transcript after the skill is released, so the model keeps reading rules for
  a skill whose tools are gone. Each one is replaced by a short notice. It is
  rewritten rather than dropped because the API wants a response for every
  function call, so removing the part would orphan its call.

  Args:
    contents: The request's conversation history, rewritten in place by
      replacing whole `Content` objects.
    load_skill_tool_name: Name of the load tool as it appears in the history,
      prefix included.
    active_skills: Skills still active for this agent. Anything else that was
      loaded counts as released.

  Returns:
    The skills stripped, in the order they appear, once per response.
  """
  if not contents:
    return []

  pruned: list[str] = []
  for index, content in enumerate(contents):
    new_parts: list[types.Part] | None = None
    for part_index, part in enumerate(content.parts or []):
      function_response = part.function_response
      if (
          function_response is None
          or function_response.name != load_skill_tool_name
      ):
        continue
      response = function_response.response
      if not isinstance(response, dict) or "instructions" not in response:
        continue
      skill_name = response.get("skill_name")
      # Without a name there is no telling which skill this is, so leave it.
      if not isinstance(skill_name, str) or skill_name in active_skills:
        continue
      if new_parts is None:
        new_parts = list(content.parts or [])
      new_parts[part_index] = part.model_copy(
          update={
              "function_response": function_response.model_copy(
                  update={
                      "response": {
                          "skill_name": skill_name,
                          "status": _UNLOADED_SKILL_STATUS,
                          "detail": _UNLOADED_SKILL_NOTICE,
                      }
                  }
              )
          }
      )
      pruned.append(skill_name)
    if new_parts is not None:
      # A request's contents are shallow copies of the session's events, so
      # this response dict is the one history holds. Replace, never edit.
      contents[index] = content.model_copy(update={"parts": new_parts})
  return pruned


def _build_skill_system_instruction(
    *,
    prefix: str | None = None,
    allowed_tools: set[str] | frozenset[str] | None = None,
    skills_folder: Path | None = None,
    script_execution_enabled: bool = True,
    unload_enabled: bool = False,
) -> str:
  """Builds the skill guidance injected into the model's system instruction.

  Args:
    prefix: Optional tool name prefix, matching the toolset's.
    allowed_tools: Optional set of base tool names available after filtering.
      When None, documents all skill tools (historical default, used for
      ``DEFAULT_SKILL_SYSTEM_INSTRUCTION``). When provided, documents all skill
      tools, and explicitly forbids calling filtered-out tools.
    skills_folder: Where skill resources are materialized, when running in an
      environment.
    script_execution_enabled: Whether scripts can actually be run. When False,
      `run_skill_script` is not offered to the model either, so advertising it
      here would promise a capability that always fails.
    unload_enabled: Whether the lifecycle feature is on. When it is,
      `unload_skill` is documented here, and named in the ban notice if
      `allowed_tools` filters it out.

  Returns:
    The system instruction text.
  """
  p = f"{prefix}_" if prefix else ""
  skills_folder_posix = (
      skills_folder.as_posix() if skills_folder is not None else None
  )
  scripts_bullet = (
      "- **scripts/** (Optional): Executable scripts that can be run via "
      "bash.\n\n"
      if script_execution_enabled
      else (
          "- **scripts/** (Optional): Scripts bundled with the skill. You"
          f" cannot run them; use `{p}{_LOAD_SKILL_RESOURCE_TOOL_NAME}` to read"
          " one and follow it yourself.\n\n"
      )
  )

  steps = [
      (
          "If a skill seems relevant to the current user query, you MUST use "
          f"the `{p}{_LOAD_SKILL_TOOL_NAME}` tool with"
          ' `skill_name="<SKILL_NAME>"` to read '
          "its full instructions before proceeding."
      ),
      (
          "Once you have read the instructions, follow them exactly as "
          "documented before replying to the user. For example, If the "
          "instruction lists multiple steps, please make sure you complete all "
          "of them in order."
      ),
      (
          f"The `{p}{_LOAD_SKILL_RESOURCE_TOOL_NAME}` tool is for viewing files"
          " within a skill's directory (e.g., `references/*`, `assets/*`,"
          " `scripts/*`). It is ONLY for skill-bundled files — do NOT use it"
          " to access documents or files provided by the user at runtime. Do"
          " NOT use other tools to access skill files."
      ),
  ]
  if script_execution_enabled:
    steps.append(
        f"Use `{p}{_RUN_SKILL_SCRIPT_TOOL_NAME}` to run scripts from a skill's"
        f" `scripts/` directory. Use `{p}{_LOAD_SKILL_RESOURCE_TOOL_NAME}` to"
        " view script content first if needed."
    )
  steps.append(
      f"If `{p}{_LOAD_SKILL_RESOURCE_TOOL_NAME}` returns any error, do not"
      " retry any path. Report the error to the user and stop."
  )
  if script_execution_enabled:
    steps.append(
        f"If `{p}{_RUN_SKILL_SCRIPT_TOOL_NAME}` returns an error (for example "
        "`SCRIPT_NOT_FOUND`), do not retry the same script or guess a "
        "different script path. Report the error to the user and stop."
    )
  steps.append(
      "Loading a skill only retrieves its instructions; it does NOT complete"
      f" your turn. After a `{p}{_LOAD_SKILL_TOOL_NAME}` call returns, continue"
      " in the SAME turn: call whatever tools the skill's steps require"
      " (search, data retrieval, render), then write your reply. Never end"
      " your turn with an empty response right after loading a skill."
  )
  if unload_enabled:
    steps.append(
        "Once a skill's task is finished and you no longer need its"
        f" instructions or its tools, call `{p}{_UNLOAD_SKILL_TOOL_NAME}` with"
        ' `skill_name="<SKILL_NAME>"` to release it. Only unload a skill you'
        " are done with: its tools stop being available, and you would have to"
        f" `{p}{_LOAD_SKILL_TOOL_NAME}` it again to use them. Unloading is"
        " optional; never unload a skill just because you loaded it."
    )
  if script_execution_enabled and skills_folder_posix is not None:
    steps.append(
        "NOTE ON ENVIRONMENT EXECUTION: When using"
        f" `{p}{_RUN_SKILL_SCRIPT_TOOL_NAME}` with the `command` parameter, all"
        " skill resources (including scripts and assets) are materialized in"
        " the execution environment under"
        f" `{skills_folder_posix}/<skill_name>/`. Always specify file and"
        " script paths relative to or starting with"
        f" `{skills_folder_posix}/<skill_name>/` (e.g.,"
        f" `{skills_folder_posix}/<skill_name>/scripts/<script_name>`)."
    )

  instruction = (
      "You can use specialized 'skills' to help you with complex tasks. "
      "You MUST use the skill tools to interact with these skills.\n\n"
      "Skills are folders of instructions and resources that extend your "
      "capabilities for specialized tasks. Each skill folder contains:\n"
      "- **SKILL.md** (required): The main instruction file with skill "
      "metadata and detailed markdown instructions.\n"
      "- **references/** (Optional): Additional documentation or examples for "
      "skill usage.\n"
      "- **assets/** (Optional): Templates, scripts or other resources used by "
      "the skill.\n"
      + scripts_bullet
      + "This is very important:\n\n"
      + "".join(f"{i}. {step}\n" for i, step in enumerate(steps, start=1))
  )

  if allowed_tools is not None:
    bannable = [
        _RUN_SKILL_SCRIPT_TOOL_NAME,
        _LOAD_SKILL_RESOURCE_TOOL_NAME,
        _LOAD_SKILL_TOOL_NAME,
        _LIST_SKILLS_TOOL_NAME,
    ]
    if unload_enabled:
      bannable.append(_UNLOAD_SKILL_TOOL_NAME)
    banned = []
    for tool_name in bannable:
      if tool_name not in allowed_tools:
        banned.append(f"`{p}{tool_name}`")
    if banned:
      banned_csv = ", ".join(banned)
      instruction += (
          f"\n\nNote: The following tools are NOT available: {banned_csv}."
          " Do NOT call them. After loading a skill (if available), apply"
          " its instructions in context and write your final reply as"
          " normal model text. Never wrap the user-facing answer inside a"
          " tool call.\n"
      )

  return instruction


class ListSkillsTool(BaseTool):
  """Tool to list all available skills."""

  TOOL_NAME = _LIST_SKILLS_TOOL_NAME

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name=self.TOOL_NAME,
        description=(
            "Lists all available skills with their names and descriptions."
        ),
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skills = self._toolset._list_skills()
    return prompt.format_skills_as_xml(skills)


class SearchSkillsTool(BaseTool):
  """Tool to search for relevant skills in the registry."""

  TOOL_NAME = _SEARCH_SKILLS_TOOL_NAME

  def __init__(self, toolset: "SkillToolset"):
    if not toolset._registry:
      raise ValueError("SearchSkillsTool requires a configured skill registry.")
    description = toolset._registry.search_tool_description() or (
        "Searches for relevant skills in the registry based on a semantic or"
        " keyword query."
    )
    super().__init__(
        name=self.TOOL_NAME,
        description=description,
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    properties = {
        "query": {
            "type": "string",
            "description": "Semantic or keyword search query.",
        },
    }
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": properties,
            "required": ["query"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    query = args.get("query")
    if not query:
      return {
          "error": "Argument 'query' is required.",
          "error_code": "INVALID_ARGUMENTS",
      }
    try:
      results = await self._toolset._registry.search_skills(query=query)
      formatted_results = []
      for r in results:
        if r.name in self._toolset._skills:
          logger.warning(
              "Skill naming conflict: skill '%s' already exists locally."
              " Registry skill is filtered.",
              r.name,
          )
          continue
        formatted_results.append(r.model_dump())
      return formatted_results
    except Exception as e:
      return {
          "error": f"Failed to search skills from registry: {e}",
          "error_code": "REGISTRY_ERROR",
      }


class LoadSkillTool(BaseTool):
  """Tool to load a skill's instructions."""

  TOOL_NAME = _LOAD_SKILL_TOOL_NAME

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name=self.TOOL_NAME,
        description="Loads the SKILL.md instructions for a given skill.",
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                },
            },
            "required": ["skill_name"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name: str | None = args.get("skill_name")
    if not skill_name:
      return {
          "error": "Argument 'skill_name' is required.",
          "error_code": "INVALID_ARGUMENTS",
      }

    skill_telemetry = _instrumentation.track_skill_load(
        _hallucination.MaybeHallucinated(skill_name)
    )

    try:
      skill = await self._toolset._get_or_fetch_skill(
          skill_name, tool_context.invocation_id
      )
    except Exception as e:
      return {
          "error": f"Failed to fetch skill '{skill_name}' from registry: {e}",
          "error_code": "REGISTRY_ERROR",
      }

    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    skill_telemetry.skill = skill
    # If we loaded a skill, it's not hallucinated, so we can confirm it.
    skill_telemetry.skill_name = _hallucination.ConfirmedNotHallucinated(
        skill.name
    )

    # Record skill activation in agent state for tool resolution.
    evicted = self._toolset._record_activation(
        tool_context.state,
        tool_context.agent_name,
        skill_name,
        tool_context.invocation_id,
        skill,
    )

    instructions = skill.instructions
    if skill.frontmatter.metadata.get("adk_inject_state"):
      instructions = await instructions_utils.inject_session_state(
          instructions,
          tool_context,
      )

    result = {
        "skill_name": skill_name,
        "instructions": instructions,
        "frontmatter": skill.frontmatter.model_dump(),
    }
    if evicted:
      # Tell the model, rather than letting the declarations quietly vanish.
      result["unloaded_skills"] = evicted
    if self._toolset._lifecycle_for(skill_name) is SkillLifecycleMode.EPHEMERAL:
      result["lifecycle_notice"] = (
          "This skill is released at the end of the current turn. Do what it"
          " is needed for now; in a later turn, load it again."
      )
    return result

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get("error"):
      error_code = response.get("error_code")
      return error_code if error_code else "TOOL_ERROR"
    return None


class UnloadSkillTool(BaseTool):
  """Tool to release an active skill.

  Drops the skill from the agent's activated-skill state, so the tools it
  contributed via ``adk_additional_tools`` stop being declared. Later requests
  replace the instructions it was loaded with by a short notice; the session's
  own events keep them. Nothing is re-fetched, so this also works for a skill
  that has left the registry.

  Known limitation: the activated-skill list is rewritten wholesale, so
  parallel writes to it race. The deltas merge per key and the last call in the
  batch wins, so two unloads issued together can leave one of the skills active
  while both report success. ``load_skill`` can lose an activation the same
  way.
  """

  TOOL_NAME = _UNLOAD_SKILL_TOOL_NAME

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name=self.TOOL_NAME,
        description=(
            "Unloads an active skill once its task is complete, releasing its"
            " dynamic tools from the context."
        ),
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to unload.",
                },
            },
            "required": ["skill_name"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Drops a skill from the calling agent's activated-skill list.

    Args:
      args: Tool arguments. ``skill_name`` (required) is the skill to release.
      tool_context: Context of the call; its session state holds the list.

    Returns:
      ``{"skill_name": str, "unloaded": True, "active_skills": list[str]}`` on
      success, listing what stays active. On failure, ``{"error": str,
      "error_code": str}``, where ``error_code`` is ``INVALID_ARGUMENTS``
      (``skill_name`` missing or empty) or ``SKILL_NOT_ACTIVE``.
    """
    skill_name: str | None = args.get("skill_name")
    if not skill_name:
      return {
          "error": "Argument 'skill_name' is required.",
          "error_code": "INVALID_ARGUMENTS",
      }

    if not self._toolset.unload_skill(tool_context, skill_name):
      return {
          "error": f"Skill '{skill_name}' is not active.",
          "error_code": "SKILL_NOT_ACTIVE",
      }

    return {
        "skill_name": skill_name,
        "unloaded": True,
        "active_skills": self._toolset.list_active_skills(tool_context),
    }

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get("error"):
      error_code = response.get("error_code")
      return error_code if error_code else "TOOL_ERROR"
    return None


class LoadSkillResourceTool(BaseTool):
  """Tool to load resources (references, assets, or scripts) from a skill."""

  TOOL_NAME = _LOAD_SKILL_RESOURCE_TOOL_NAME

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name=self.TOOL_NAME,
        description=(
            "Loads a resource file (from references/, assets/, or"
            " scripts/) from within a skill."
        ),
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "The relative path to the resource (e.g.,"
                        " 'references/my_doc.md', 'assets/template.txt',"
                        " or 'scripts/setup.sh')."
                    ),
                },
            },
            "required": ["skill_name", "file_path"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    skill_name: str | None = args.get("skill_name")
    file_path: str | None = args.get("file_path")

    if not skill_name or not file_path:
      errors = []
      if not skill_name:
        errors.append("Argument 'skill_name' is required.")
      if not file_path:
        errors.append("Argument 'file_path' is required.")
      return {
          "error": "\n".join(errors),
          "error_code": "INVALID_ARGUMENTS",
      }

    skill_telemetry = _instrumentation.track_skill_resource_load(
        _hallucination.MaybeHallucinated(skill_name),
        _hallucination.MaybeHallucinated(file_path),
    )

    try:
      skill = await self._toolset._get_or_fetch_skill(
          skill_name, tool_context.invocation_id
      )
    except Exception as e:
      return {
          "error": f"Failed to fetch skill '{skill_name}' from registry: {e}",
          "error_code": "REGISTRY_ERROR",
      }

    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    skill_telemetry.skill = skill
    # If we loaded a skill, it's not hallucinated, so we can confirm it.
    skill_telemetry.skill_name = _hallucination.ConfirmedNotHallucinated(
        skill.name
    )

    content = None
    if file_path.startswith("references/"):
      ref_name = file_path[len("references/") :]
      content = skill.resources.get_reference(ref_name)
    elif file_path.startswith("assets/"):
      asset_name = file_path[len("assets/") :]
      content = skill.resources.get_asset(asset_name)
    elif file_path.startswith("scripts/"):
      script_name = file_path[len("scripts/") :]
      script = skill.resources.get_script(script_name)
      if script is not None:
        content = script.src
    else:
      return {
          "error": (
              "Path must start with 'references/', 'assets/', or 'scripts/'."
          ),
          "error_code": "INVALID_RESOURCE_PATH",
      }

    if content is None:
      # Invocation-scoped failure counter. Counts RESOURCE_NOT_FOUND across ALL
      # paths so the guard fires even when the LLM hallucinates a different path
      # on each retry. The `temp:` prefix prevents persistence to durable
      # session storage; invocation_id isolates in-memory backends.
      counter_key = f"temp:_adk_skill_resource_not_found_count_{tool_context.invocation_id}"
      fail_count = int(tool_context.state.get(counter_key) or 0) + 1
      tool_context.state[counter_key] = fail_count
      if fail_count > 1:
        return {
            "error": (
                f"Resource '{file_path}' not found in skill '{skill_name}'."
                f" This is resource lookup failure #{fail_count} this"
                " invocation. Do not retry any path — report the error to"
                " the user and stop."
            ),
            "error_code": "RESOURCE_NOT_FOUND_FATAL",
        }
      return {
          "error": f"Resource '{file_path}' not found in skill '{skill_name}'.",
          "error_code": "RESOURCE_NOT_FOUND",
      }

    if content is not None:
      # If we found the resource, it's not hallucinated, so we can confirm it.
      skill_telemetry.resource_path = _hallucination.ConfirmedNotHallucinated(
          file_path
      )

    if isinstance(content, bytes):
      return {
          "skill_name": skill_name,
          "file_path": file_path,
          "status": _BINARY_FILE_DETECTED_MSG,
      }

    return {
        "skill_name": skill_name,
        "file_path": file_path,
        "content": content,
    }

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get("error"):
      error_code = response.get("error_code")
      return error_code if error_code else "TOOL_ERROR"
    return None

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: Any
  ) -> None:
    """Injects binary content into the LLM request if the model viewed it."""
    await super().process_llm_request(
        tool_context=tool_context, llm_request=llm_request
    )

    if not llm_request.contents:
      return

    # Check for LoadSkillResource calls on binary files in the last turn
    for part in llm_request.contents[-1].parts:
      if not part.function_response or part.function_response.name != self.name:
        continue

      response = part.function_response.response or {}
      if response.get("status") != _BINARY_FILE_DETECTED_MSG:
        continue

      skill_name = response.get("skill_name")
      file_path = response.get("file_path")
      if not skill_name or not file_path:
        continue

      try:
        skill = await self._toolset._get_or_fetch_skill(
            skill_name, tool_context.invocation_id
        )
      except Exception as e:
        logger.warning(
            "Failed to fetch skill '%s' from registry during LLM request"
            " processing: %s",
            skill_name,
            e,
        )
        continue

      if not skill:
        continue

      # Find the binary content
      content = None
      if file_path.startswith("references/"):
        ref_name = file_path[len("references/") :]
        content = skill.resources.get_reference(ref_name)
      elif file_path.startswith("assets/"):
        asset_name = file_path[len("assets/") :]
        content = skill.resources.get_asset(asset_name)

      if not isinstance(content, bytes):
        continue

      # Determine mime type based on extension
      mime_type, _ = mimetypes.guess_type(file_path)
      if not mime_type:
        mime_type = "application/octet-stream"

      # Append binary content to llm_request
      llm_request.contents.append(
          types.Content(
              role="user",
              parts=[
                  types.Part.from_text(
                      text=f"The content of binary file '{file_path}' is:"
                  ),
                  types.Part(
                      inline_data=types.Blob(
                          data=content,
                          mime_type=mime_type,
                      )
                  ),
              ],
          )
      )


class _SkillScriptCodeExecutor:
  """A helper that materializes skill files and executes scripts."""

  _base_executor: BaseCodeExecutor
  _script_timeout: int

  _WRAPPER_START_TEMPLATE = """
import os
import tempfile
import sys
import json as _json
import subprocess
import runpy
_files = {files_dict!r}
def _materialize_and_run():
  _orig_cwd = os.getcwd()
  with tempfile.TemporaryDirectory() as td:
    for rel_path, content in _files.items():
      norm_rel = os.path.normpath(rel_path)
      if norm_rel.startswith('..') or os.path.isabs(norm_rel):
        raise PermissionError(
            'Path traversal blocked in skill file: ' + rel_path
        )
      full_path = os.path.join(os.path.abspath(td), norm_rel)
      os.makedirs(os.path.dirname(full_path), exist_ok=True)
      mode = 'wb' if isinstance(content, bytes) else 'w'
      with open(full_path, mode,
        encoding='utf-8' if mode == 'w' else None
      ) as f:
        f.write(content)
    os.chdir(td)
    try:
"""

  _WRAPPER_END_TEMPLATE = """
    finally:
      os.chdir(_orig_cwd)
_materialize_and_run()
"""

  _WRAPPER_PYTHON_TEMPLATE = """
      sys.argv = {argv_list!r}
      sys.path.insert(0, os.path.dirname(os.path.abspath({file_path!r})))
      try:
        runpy.run_path({file_path!r}, run_name='__main__')
      except SystemExit as e:
        if e.code is not None and e.code != 0:
          raise e
"""

  _WRAPPER_SHELL_TEMPLATE = """
      try:
        _r = subprocess.run(
          {arr!r},
          capture_output=True,
          text=True,
          # Keep shell output decoding independent from the host locale.
          encoding='utf-8',
          errors='replace',
          timeout={timeout!r},
          cwd=td,
        )
        print('\\n' + _json.dumps({{
            '__shell_result__': True,
            'stdout': _r.stdout,
            'stderr': _r.stderr,
            'returncode': _r.returncode,
        }}))
      except subprocess.TimeoutExpired as _e:
        print('\\n' + _json.dumps({{
            '__shell_result__': True,
            'stdout': _e.stdout or '',
            'stderr': 'Timed out after {timeout}s',
            'returncode': -1,
            'timeout': True,
        }}))
"""

  def __init__(self, base_executor: BaseCodeExecutor, script_timeout: int):
    self._base_executor = base_executor
    self._script_timeout = script_timeout

  async def execute_script_async(
      self,
      invocation_context: Any,
      skill: models.Skill,
      file_path: str,
      script_args: dict[str, Any] | list[str] | None,
      short_options: dict[str, Any] | None = None,
      positional_args: list[str] | None = None,
      skill_telemetry: _instrumentation.SkillScriptExecutionTelemetry | None = (
          None
      ),
  ) -> dict[str, Any]:
    """Prepares and executes the script using the base executor.

    Args:
      invocation_context: The context for execution.
      skill: The skill containing the script.
      file_path: Relative path to the script file (e.g., 'scripts/myscript.py'
        or 'myscript.py').
      script_args: Optional arguments to pass to the script. Can be a dict of
        long options or a list of strings.
      short_options: Optional short options (single hyphen) as key-value pairs.
      positional_args: Optional positional arguments.
      skill_telemetry: Optional telemetry object to record script execution
        details.

    Returns:
      A dictionary containing execution results (stdout, stderr, status).
    """
    code = self._build_wrapper_code(
        skill, file_path, script_args, short_options, positional_args
    )
    if code is None:
      if "." in file_path:
        ext_msg = f"'.{file_path.rsplit('.', 1)[-1]}'"
      else:
        ext_msg = "(no extension)"
      return {
          "error": (
              f"Unsupported script type {ext_msg}."
              " Supported types: .py, .sh, .bash"
          ),
          "error_code": "UNSUPPORTED_SCRIPT_TYPE",
      }

    try:
      # Execute the self-contained script using the underlying executor
      result = await asyncio.to_thread(
          self._base_executor.execute_code,
          invocation_context,
          CodeExecutionInput(code=code),
      )

      stdout = result.stdout
      stderr = result.stderr

      # The status the script exited with, or None when nothing reported one.
      rc: int | None = None
      is_shell = "." in file_path and file_path.rsplit(".", 1)[-1].lower() in (
          "sh",
          "bash",
      )
      if is_shell:
        # A shell script runs as a child of the wrapper, so the wrapper's own
        # status says nothing about it. Both streams come back serialized as
        # JSON through stdout; that envelope carries the script's status.
        if stdout:
          parsed = None
          try:
            parsed = json.loads(stdout)
          except (json.JSONDecodeError, ValueError):
            # Fallback: try parsing line by line to find the shell result
            # envelope. This handles cases where other processes or Python
            # startup scripts printed warnings to stdout.
            for line in reversed(stdout.splitlines()):
              line = line.strip()
              if line.startswith("{") and line.endswith("}"):
                try:
                  candidate = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                  continue
                if isinstance(candidate, dict) and candidate.get(
                    "__shell_result__"
                ):
                  parsed = candidate
                  break

          if isinstance(parsed, dict) and parsed.get("__shell_result__"):
            stdout = parsed.get("stdout", "")
            stderr = parsed.get("stderr", "")
            rc = parsed.get("returncode", 0)
            if rc != 0 and not parsed.get("timeout", False):
              exit_code_message = f"Exit code {rc}"
              stderr = (
                  f"{stderr.rstrip()}\n{exit_code_message}"
                  if stderr
                  else exit_code_message
              )
          else:
            logger.warning(
                "No shell execution envelope found in stdout for '%s' from"
                " skill '%s'.",
                file_path,
                skill.name,
            )
      else:
        # A Python script runs in the wrapper process itself, so the process
        # the executor ran exited with the script's own status. Executors that
        # cannot report one leave this None and fall back to stderr below.
        rc = result.exit_code

      status = "success"
      if rc is not None and rc != 0:
        status = "error"
      elif stderr and not stdout:
        status = "error"
        # Reached only when the executor reported no status: an inference, and
        # never an override of a status the run actually reported.
        if rc is None:
          rc = 1
      elif stderr:
        status = "warning"

      if skill_telemetry is not None:
        skill_telemetry.script_exit_code = rc

      return {
          "skill_name": skill.name,
          "file_path": file_path,
          "stdout": stdout,
          "stderr": stderr,
          "status": status,
      }
    except SystemExit as e:
      if skill_telemetry is not None:
        if e.code is None:
          skill_telemetry.script_exit_code = 0
        elif isinstance(e.code, int):
          skill_telemetry.script_exit_code = e.code
        else:
          skill_telemetry.script_exit_code = 1
      if e.code in (None, 0):
        return {
            "skill_name": skill.name,
            "file_path": file_path,
            "stdout": "",
            "stderr": "",
            "status": "success",
        }
      return {
          "error": (
              f"Failed to execute script '{file_path}':"
              f" exited with code {e.code}"
          ),
          "error_code": "EXECUTION_ERROR",
      }
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.exception(
          "Error executing script '%s' from skill '%s'",
          file_path,
          skill.name,
      )
      short_msg = str(e)
      if len(short_msg) > 200:
        short_msg = short_msg[:200] + "..."
      return {
          "error": (
              "Failed to execute script"
              f" '{file_path}':\n{type(e).__name__}:"
              f" {short_msg}"
          ),
          "error_code": "EXECUTION_ERROR",
      }

  def _build_wrapper_code(
      self,
      skill: models.Skill,
      file_path: str,
      script_args: dict[str, Any] | list[str] | None,
      short_options: dict[str, Any] | None = None,
      positional_args: list[str] | None = None,
  ) -> str | None:
    """Builds a self-extracting Python script."""
    ext = ""
    if "." in file_path:
      ext = file_path.rsplit(".", 1)[-1].lower()

    if not file_path.startswith("scripts/"):
      file_path = f"scripts/{file_path}"

    files_dict = {}
    for ref_name in skill.resources.list_references():
      content = skill.resources.get_reference(ref_name)
      if content is not None:
        files_dict[f"references/{ref_name}"] = content

    for asset_name in skill.resources.list_assets():
      content = skill.resources.get_asset(asset_name)
      if content is not None:
        files_dict[f"assets/{asset_name}"] = content

    for scr_name in skill.resources.list_scripts():
      scr = skill.resources.get_script(scr_name)
      if scr is not None and scr.src is not None:
        files_dict[f"scripts/{scr_name}"] = scr.src

    total_size = sum(
        len(v) if isinstance(v, (str, bytes)) else 0
        for v in files_dict.values()
    )
    if total_size > _MAX_SKILL_PAYLOAD_BYTES:
      logger.warning(
          "Skill '%s' resources total %d bytes, exceeding"
          " the recommended limit of %d bytes.",
          skill.name,
          total_size,
          _MAX_SKILL_PAYLOAD_BYTES,
      )

    # Build the boilerplate extract string
    code = self._WRAPPER_START_TEMPLATE.format(
        files_dict=files_dict,
    )

    if ext == "py":
      argv_list = [file_path]
      if isinstance(script_args, list):
        argv_list.extend(str(v) for v in script_args)
      else:
        if isinstance(script_args, dict):
          for k, v in script_args.items():
            argv_list.extend([f"--{k}", str(v)])

        if short_options:
          for k, v in short_options.items():
            argv_list.extend([f"-{k}", str(v)])

        if positional_args:
          argv_list.append("--")
          argv_list.extend(str(v) for v in positional_args)

      code += self._WRAPPER_PYTHON_TEMPLATE.format(
          argv_list=argv_list,
          file_path=file_path,
      )
    elif ext in ("sh", "bash"):
      arr = ["bash", file_path]
      if isinstance(script_args, list):
        arr.extend(str(v) for v in script_args)
      else:
        if isinstance(script_args, dict):
          for k, v in script_args.items():
            arr.extend([f"--{k}", str(v)])

        if short_options:
          for k, v in short_options.items():
            arr.extend([f"-{k}", str(v)])

        if positional_args:
          arr.append("--")
          arr.extend(positional_args)
      timeout = self._script_timeout
      code += self._WRAPPER_SHELL_TEMPLATE.format(
          arr=arr,
          timeout=timeout,
      )
    else:
      return None

    code += self._WRAPPER_END_TEMPLATE
    return code


class RunSkillScriptTool(BaseTool):
  """Tool to execute scripts from a skill's scripts/ directory."""

  TOOL_NAME = _RUN_SKILL_SCRIPT_TOOL_NAME

  def __init__(self, toolset: "SkillToolset"):
    super().__init__(
        name=self.TOOL_NAME,
        description="Executes a script from a skill's scripts/ directory.",
    )
    self._toolset = toolset

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    if self._toolset._env is not None:
      return types.FunctionDeclaration(
          name=self.name,
          description=self.description,
          parameters_json_schema={
              "type": "object",
              "properties": {
                  "skill_name": {
                      "type": "string",
                      "description": "The name of the skill.",
                  },
                  "file_path": {
                      "type": "string",
                      "description": (
                          "The relative path to the script (e.g.,"
                          " 'scripts/setup.py')."
                      ),
                  },
                  "command": {
                      "type": "string",
                      "description": (
                          "The command to execute in the environment."
                      ),
                  },
              },
              "required": ["skill_name", "file_path", "command"],
          },
      )
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "The relative path to the script (e.g.,"
                        " 'scripts/setup.py')."
                    ),
                },
                "args": {
                    "anyOf": [
                        {"type": "object"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": (
                        "Optional arguments to pass to the script as key-value"
                        " pairs (long options) or as a list of strings. If"
                        " specified as a list, it is treated as the complete"
                        " list of arguments, and 'short_options' and"
                        " 'positional_args' must not be provided."
                    ),
                },
                "short_options": {
                    "type": "object",
                    "description": (
                        "Optional short options (single hyphen) to pass to the"
                        " script as key-value pairs. Must not be provided if"
                        " 'args' is a list."
                    ),
                },
                "positional_args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional positional arguments to pass to the script."
                        " Must not be provided if 'args' is a list."
                    ),
                },
            },
            "required": ["skill_name", "file_path"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    # Standardized arguments: skill_name and file_path.
    skill_name: str | None = args.get("skill_name")
    file_path: str | None = args.get("file_path")
    command: str | None = args.get("command")
    script_args: Any = args.get("args")
    short_options: Any = args.get("short_options")
    positional_args: Any = args.get("positional_args")

    if not skill_name or not file_path:
      errors = []
      if not skill_name:
        errors.append("Argument 'skill_name' is required.")
      if not file_path:
        errors.append("Argument 'file_path' is required.")
      return {
          "error": "\n".join(errors),
          "error_code": "INVALID_ARGUMENTS",
      }

    skill_telemetry = _instrumentation.track_skill_script_execution(
        _hallucination.MaybeHallucinated(skill_name),
        _hallucination.MaybeHallucinated(file_path),
    )

    env = self._toolset._env
    if env is not None:
      if command is None or not isinstance(command, str) or not command:
        return {
            "error": "Argument 'command' is required and must be a string.",
            "error_code": "INVALID_ARGUMENTS",
        }
    else:
      errors = []
      if script_args is not None and not isinstance(script_args, (dict, list)):
        errors.append(
            "'args' must be a JSON object (dict) or a list of strings,"
            f" got {type(script_args).__name__}."
        )

      if short_options is not None and not isinstance(short_options, dict):
        errors.append(
            "'short_options' must be a JSON object (dict),"
            f" got {type(short_options).__name__}."
        )

      if positional_args is not None and not isinstance(positional_args, list):
        errors.append(
            "'positional_args' must be a list of strings,"
            f" got {type(positional_args).__name__}."
        )

      if isinstance(script_args, list) and (short_options or positional_args):
        errors.append(
            "Cannot specify 'short_options' or 'positional_args' when 'args'"
            " is a list."
        )

      if errors:
        return {
            "error": "\n".join(errors),
            "error_code": "INVALID_ARGUMENTS",
        }

    try:
      skill = await self._toolset._get_or_fetch_skill(
          skill_name, tool_context.invocation_id
      )
    except Exception as e:
      return {
          "error": f"Failed to fetch skill '{skill_name}' from registry: {e}",
          "error_code": "REGISTRY_ERROR",
      }

    if not skill:
      return {
          "error": f"Skill '{skill_name}' not found.",
          "error_code": "SKILL_NOT_FOUND",
      }

    skill_telemetry.skill = skill
    # If we loaded the skill, it's not hallucinated, so we can confirm it.
    skill_telemetry.skill_name = _hallucination.ConfirmedNotHallucinated(
        skill.name
    )

    if file_path.startswith("scripts/"):
      script = skill.resources.get_script(file_path[len("scripts/") :])
    else:
      script = skill.resources.get_script(file_path)

    if script is None:
      # Invocation-scoped failure counter. Counts SCRIPT_NOT_FOUND across ALL
      # paths so the guard fires even when the LLM hallucinates a different
      # script path on each retry. The `temp:` prefix prevents persistence to
      # durable session storage; invocation_id isolates in-memory backends.
      counter_key = (
          f"temp:_adk_skill_script_not_found_count_{tool_context.invocation_id}"
      )
      fail_count = int(tool_context.state.get(counter_key) or 0) + 1
      tool_context.state[counter_key] = fail_count
      if fail_count > 1:
        return {
            "error": (
                f"Script '{file_path}' not found in skill '{skill_name}'."
                f" This is script lookup failure #{fail_count} this"
                " invocation. Do not retry any script path — report the"
                " error to the user and stop."
            ),
            "error_code": "SCRIPT_NOT_FOUND_FATAL",
        }
      return {
          "error": f"Script '{file_path}' not found in skill '{skill_name}'.",
          "error_code": "SCRIPT_NOT_FOUND",
      }

    if script is not None:
      # If we found the script, we can mark the path as not hallucinated.
      skill_telemetry.script_path = _hallucination.ConfirmedNotHallucinated(
          file_path
      )

    if env is not None:
      try:
        await self._ensure_skill_materialized_in_env(skill, file_path, env)
        result = await env.execute(
            command=cast(str, command),
            timeout=self._toolset._script_timeout,
        )
        skill_telemetry.script_exit_code = result.exit_code
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
        }
      except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception(
            "Error executing script '%s' from skill '%s' in environment",
            file_path,
            skill.name,
        )
        short_msg = str(e)
        if len(short_msg) > 200:
          short_msg = short_msg[:200] + "..."
        return {
            "error": (
                "Failed to execute script"
                f" '{file_path}' in environment:\n{type(e).__name__}:"
                f" {short_msg}"
            ),
            "error_code": "EXECUTION_ERROR",
        }

    # Resolve code executor: toolset-level first, then agent fallback
    code_executor = self._toolset._code_executor
    if code_executor is None:
      agent = tool_context._invocation_context.agent
      if hasattr(agent, "code_executor"):
        code_executor = agent.code_executor
    if code_executor is None:
      return {
          "error": (
              "Neither Environment nor CodeExecutor is configured. An"
              " environment or code executor is required to run scripts."
          ),
          "error_code": "NO_CODE_EXECUTOR",
      }

    script_executor = _SkillScriptCodeExecutor(
        code_executor, self._toolset._script_timeout  # pylint: disable=protected-access
    )
    return await script_executor.execute_script_async(
        tool_context._invocation_context,  # pylint: disable=protected-access
        skill,
        file_path,
        script_args,
        short_options,
        positional_args,  # pylint: disable=protected-access
        skill_telemetry,
    )

  async def _ensure_skill_materialized_in_env(
      self, skill: models.Skill, file_path: str, env: BaseEnvironment
  ) -> None:
    # JIT Materialization: Check if the script exists in the environment.
    # If not, write all skill resources (including scripts) to the environment.
    skills_folder = self._toolset.skills_folder
    if skills_folder is None:
      raise RuntimeError(
          "skills_folder is not set and no environment working_dir available."
      )
    skill_dir = skills_folder / skill.name
    if not file_path.startswith("scripts/"):
      rel_script = f"scripts/{file_path}"
    else:
      rel_script = file_path
    script_path = skill_dir / rel_script

    try:
      await env.read_file(cast(Path, PurePosixPath(script_path.as_posix())))
      script_exists = True
    except FileNotFoundError:
      script_exists = False

    if not script_exists:
      logger.info(
          "Materializing skill resources for %s in environment", skill.name
      )
      await _write_skill_resources_to_env(skill, skill_dir, env)

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get("error"):
      error_code = response.get("error_code")
      return error_code if error_code else "TOOL_ERROR"
    if isinstance(response, dict) and response.get("status", "") == "error":
      return "SKILL_SCRIPT_EXECUTION_ERROR"
    return None


class SkillToolset(BaseToolset):
  """A toolset for managing and interacting with agent skills."""

  def __init__(
      self,
      skills: list[models.Skill] | None = None,
      *,
      registry: SkillRegistry | None = None,
      code_executor: BaseCodeExecutor | None = None,
      environment: BaseEnvironment | None = None,
      skills_folder: Path | str | None = None,
      script_timeout: int = _DEFAULT_SCRIPT_TIMEOUT,
      additional_tools: list[ToolUnion] | None = None,
      tool_name_prefix: str | None = None,
      tool_filter: ToolPredicate | list[str] | None = None,
      discovery_mode: SkillDiscoveryMode = SkillDiscoveryMode.LAZY,
      lifecycle_config: SkillLifecycleConfig | None = None,
  ):
    """Initializes the SkillToolset.

    Args:
      skills: List of skills to register.
      registry: Optional skill registry for dynamic loading.
      code_executor: Optional code executor for script execution.
      environment: Optional environment for executing scripts.
      skills_folder: Optional absolute path where skills are stored in the
        environment filesystem. Defaults to 'skills' under the environment's
        working directory.
      script_timeout: Timeout in seconds for shell script execution via
        subprocess.run. Defaults to 300 seconds. Does not apply to Python
        scripts executed via exec().
      additional_tools: Optional list of `BaseTool` or `BaseToolset` instances
        to be made available to the agent when certain skills are activated.
      tool_name_prefix: Optional prefix to prepend to tool names.
      tool_filter: Optional filter to select specific tools.
      discovery_mode: How the local catalog reaches the model. Defaults to
        `LAZY`, where it calls `list_skills`. `EAGER` drops that tool and
        injects the catalog into the system instruction instead.
      lifecycle_config: How long skills stay active once loaded, how many may
        be at once, and whether they are rechecked against their definition.
        Defaults to leaving every skill active for the rest of the session,
        which is how skills behaved before this existed.

    Raises:
      ValueError: If both `code_executor` and `environment` are given, or on a
        duplicate skill name or a relative `skills_folder`.
    """
    super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)

    config = lifecycle_config or SkillLifecycleConfig()
    # Copy the overrides: editing the config later must not change a session
    # already running.
    self._lifecycle_config = dataclasses.replace(
        config, skill_overrides=dict(config.skill_overrides)
    )
    # Nothing is ephemeral for most callers, and then the lifecycle records
    # are never touched at all.
    self._tracks_ephemeral_skills = config.enabled and (
        SkillLifecycleMode.EPHEMERAL
        in ({config.default_mode} | set(config.skill_overrides.values()))
    )
    self._revalidate_skills = config.enabled and config.revalidate_skills
    # Digest last written into the sandbox, per skill, so a stale skill is
    # rewritten once rather than every request.
    self._materialized_skill_hashes: dict[str, str] = {}

    skills = skills or []

    # Check for duplicate skill names
    seen: set[str] = set()
    for skill in skills:
      if skill.name in seen:
        raise ValueError(f"Duplicate skill name '{skill.name}'.")
      seen.add(skill.name)

    self._skills = {skill.name: skill for skill in skills}
    self._registry = registry
    self._code_executor = code_executor
    self._env = environment
    if code_executor and environment:
      raise ValueError("Cannot have both code_executor and environment")
    self._skills_folder: Path | None = None
    if skills_folder is not None:
      if environment is None:
        raise ValueError("Cannot specify skills_folder without an environment")
      is_absolute = (
          PurePosixPath(skills_folder).is_absolute()
          or PureWindowsPath(skills_folder).is_absolute()
      )
      if not is_absolute:
        raise ValueError(
            f"`skills_folder` must be an absolute path: '{skills_folder}'"
        )
      self._skills_folder = Path(skills_folder)
    self._script_timeout = script_timeout
    # Needed for mid-turn reloading of skill tools.
    self._use_invocation_cache = False
    # Cache fetched remote skill definitions per turn to reduce requests to registry
    self._fetched_skill_cache: collections.OrderedDict[
        str,
        dict[str, models.Skill | asyncio.Future[models.Skill | None] | None],
    ] = collections.OrderedDict()
    self._max_cache_turns = 16

    self._provided_tools_by_name = {}
    self._provided_toolsets = []
    for tool_union in additional_tools or []:
      if isinstance(tool_union, BaseToolset):
        self._provided_toolsets.append(tool_union)
      elif isinstance(tool_union, BaseTool):
        self._provided_tools_by_name[tool_union.name] = tool_union
      elif callable(tool_union):
        ft = FunctionTool(tool_union)
        self._provided_tools_by_name[ft.name] = ft

    self._discovery_mode = discovery_mode
    self._warned_on_filtered_list_skills = False
    self._lifecycle_enabled = is_feature_enabled(FeatureName.SKILL_LIFECYCLE)

    # Initialize core skill tools
    self._tools: list[BaseTool] = []
    if discovery_mode is SkillDiscoveryMode.LAZY:
      self._tools.append(ListSkillsTool(self))
    self._tools.extend([
        LoadSkillTool(self),
        LoadSkillResourceTool(self),
        RunSkillScriptTool(self),
    ])
    if self._registry:
      self._tools.append(SearchSkillsTool(self))
    if self._lifecycle_enabled:
      self._tools.append(UnloadSkillTool(self))

  @property
  def skills_folder(self) -> Path | None:
    """The path where skills are materialized in the environment filesystem."""
    if self._skills_folder is not None:
      return self._skills_folder
    if self._env is not None:
      return self._env.working_dir / "skills"
    return None

  def _has_script_execution(self, context: ReadonlyContext | None) -> bool:
    """Whether scripts can be run; an unknown agent counts as yes."""
    if self._env is not None or self._code_executor is not None:
      return True
    agent = getattr(
        getattr(context, "_invocation_context", None), "agent", None
    )
    if agent is None:
      return True
    return getattr(agent, "code_executor", None) is not None

  def _offers_script_tool(self, context: ReadonlyContext | None) -> bool:
    """Whether `run_skill_script` can succeed; anything unknown counts as yes."""
    if not self._has_script_execution(context):
      return False
    if (
        getattr(context, "_invocation_context", None) is None
        or self._registry is not None
    ):
      return True
    return any(skill.resources.list_scripts() for skill in self._list_skills())

  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Returns the list of tools in this toolset."""
    dynamic_tools = await self._resolve_additional_tools_from_state(
        readonly_context
    )
    all_tools = self._tools + dynamic_tools
    if not self._offers_script_tool(readonly_context):
      all_tools = [
          t for t in all_tools if not isinstance(t, RunSkillScriptTool)
      ]
    return [t for t in all_tools if self._is_tool_selected(t, readonly_context)]

  async def _resolve_additional_tools_from_state(
      self, readonly_context: ReadonlyContext | None
  ) -> list[BaseTool]:
    """Resolves tools listed in the "adk_additional_tools" metadata of skills."""

    if not readonly_context:
      return []

    activated_skills = self._active_skills(
        readonly_context.state,
        readonly_context.agent_name,
        readonly_context.invocation_id,
    )

    if not activated_skills:
      return []

    additional_tool_names = set()
    for skill_name in activated_skills:
      skill = await self._get_or_fetch_skill(
          skill_name, readonly_context.invocation_id
      )
      if skill:
        additional_tools = skill.frontmatter.metadata.get(
            "adk_additional_tools"
        )
        if additional_tools:
          additional_tool_names.update(additional_tools)

    if not additional_tool_names:
      return []

    # Collect all candidate tools from both individual tools and toolsets
    candidate_tools = self._provided_tools_by_name.copy()
    if self._provided_toolsets:
      ts_results = await asyncio.gather(
          *(
              ts.get_tools_with_prefix(readonly_context)
              for ts in self._provided_toolsets
          ),
          return_exceptions=True,
      )
      for toolset, ts_tools in zip(self._provided_toolsets, ts_results):
        if isinstance(ts_tools, Exception):
          logger.warning(
              "Skipping toolset %s while resolving skill additional tools: %s",
              type(toolset).__name__,
              ts_tools,
              exc_info=ts_tools,
          )
          continue
        if isinstance(ts_tools, BaseException):
          raise ts_tools
        for t in ts_tools:
          candidate_tools[t.name] = t

    resolved_tools = []
    existing_tool_names = {t.name for t in self._tools}
    for name in additional_tool_names:
      if name in candidate_tools:
        tool = candidate_tools[name]
        if tool.name in existing_tool_names:
          logger.error(
              "Tool name collision: tool '%s' already exists.", tool.name
          )
          continue
        resolved_tools.append(tool)
        existing_tool_names.add(tool.name)

    return resolved_tools

  def _get_skill(self, skill_name: str) -> models.Skill | None:
    """Retrieves a skill by name."""
    return self._skills.get(skill_name)

  async def _get_or_fetch_skill(
      self, skill_name: str, invocation_id: str | None = None
  ) -> models.Skill | None:
    """Retrieves a skill by name, falling back to the registry if configured."""
    skill = self._get_skill(skill_name)
    if skill:
      return skill

    if not self._registry:
      return None

    if invocation_id:
      if invocation_id not in self._fetched_skill_cache:
        # Enforce bounded cache (FIFO eviction)
        if len(self._fetched_skill_cache) >= self._max_cache_turns:
          self._fetched_skill_cache.popitem(last=False)
        self._fetched_skill_cache[invocation_id] = {}

      turn_cache = self._fetched_skill_cache[invocation_id]
      if skill_name in turn_cache:
        cached = turn_cache[skill_name]
        if isinstance(cached, asyncio.Future):
          return await cached
        return cached

      loop = asyncio.get_running_loop()
      fut = loop.create_future()
      turn_cache[skill_name] = fut

      try:
        skill = await self._registry.get_skill(name=skill_name)
        fut.set_result(skill)
        turn_cache[skill_name] = skill
        return skill
      except Exception as e:
        fut.set_exception(e)
        fut.exception()
        turn_cache.pop(skill_name, None)
        raise

    return await self._registry.get_skill(name=skill_name)

  def _list_skills(self) -> list[models.Skill]:
    """Lists all available skills."""
    return list(self._skills.values())

  @property
  def skills(self) -> list[models.Skill]:
    """Returns the list of available skills."""
    return self._list_skills()

  def list_active_skills(self, ctx: ReadonlyContext) -> list[str]:
    """Returns the skills active for `ctx`'s agent, least recently loaded first.

    That is activation order, except that reloading a skill that is not
    PERSISTENT moves it to the end — the order the cap evicts in. An EPHEMERAL
    skill loaded in an earlier turn has been released and is not listed.

    Args:
      ctx: A context for the running agent. `ToolContext` is one.

    Returns:
      The active skill names. Activation is recorded by name, so a name here
      is not guaranteed to still resolve against the registry.
    """
    return self._active_skills(ctx.state, ctx.agent_name, ctx.invocation_id)

  async def load_skill(self, ctx: ToolContext, skill_name: str) -> bool:
    """Activates a skill for `ctx`'s agent without going through the model.

    Activation is what registers the skill's `adk_additional_tools`. Unlike the
    `load_skill` tool, this does not put the skill's instructions into the
    conversation: the model gets the tools without being told what they are
    for, so supply that guidance yourself.

    Args:
      ctx: A context for the running agent, e.g. the `ToolContext` a tool or
        callback was handed.
      skill_name: The skill to activate.

    The lifecycle applies here exactly as it does to the `load_skill` tool:
    loading a BOUNDED skill can release other bounded ones, reloading an active
    one counts as a use, and an EPHEMERAL skill gets the turn `ctx` is in. That
    is not reported back, so a caller who needs to know should read
    `list_active_skills`.

    Reloading an already active skill does not refresh what `revalidate_skills`
    compares against, because nothing here is said to the model: a skill that
    changed goes on being re-stated until the model loads it itself.

    Returns:
      True if the skill was activated, False if it already was. An already
      active skill is reported without consulting the registry, so this works
      for one that has since been removed from it.

    Raises:
      ValueError: If no such skill is available locally or in the registry.
      Exception: Whatever the registry raises if the lookup itself fails.
    """
    if skill_name in self._active_skills(
        ctx.state, ctx.agent_name, ctx.invocation_id
    ):
      self._record_activation(
          ctx.state, ctx.agent_name, skill_name, ctx.invocation_id
      )
      return False

    skill = await self._get_or_fetch_skill(skill_name, ctx.invocation_id)
    if skill is None:
      raise ValueError(f"Skill '{skill_name}' not found.")

    # The fetch suspends, so re-read: a concurrent activation may have written
    # the list since.
    was_active = skill_name in self._active_skills(
        ctx.state, ctx.agent_name, ctx.invocation_id
    )
    self._record_activation(
        ctx.state, ctx.agent_name, skill_name, ctx.invocation_id, skill
    )
    return not was_active

  def unload_skill(self, ctx: ToolContext, skill_name: str) -> bool:
    """Deactivates a skill for `ctx`'s agent, releasing its dynamic tools.

    Later requests replace the instructions it was loaded with by a short
    notice; the session's own events keep them. Synchronous, unlike
    `load_skill`, because deactivation never consults the registry — so it also
    works for a skill that has since been removed from one.

    Args:
      ctx: A context for the running agent, e.g. the `ToolContext` a tool or
        callback was handed.
      skill_name: The skill to deactivate.

    Returns:
      True if the skill was deactivated, False if it was not active. An
      EPHEMERAL skill whose turn has passed was already released, so it reports
      False, but what it left behind in state is cleaned up all the same.
    """
    stored_skills = _read_activated_skills(ctx.state, ctx.agent_name)
    if skill_name not in stored_skills:
      return False
    was_active = skill_name in self._active_skills(
        ctx.state, ctx.agent_name, ctx.invocation_id
    )
    stored_skills.remove(skill_name)
    _write_activated_skills(ctx.state, ctx.agent_name, stored_skills)
    self._forget_lifecycle_records(ctx.state, ctx.agent_name, stored_skills)
    return was_active

  def _lifecycle_for(self, skill_name: str) -> SkillLifecycleMode:
    """Returns the lifecycle configured for a skill.

    The only reader of the config, so `enabled=False` is enough to switch the
    whole feature off.
    """
    config = self._lifecycle_config
    if not config.enabled:
      return SkillLifecycleMode.PERSISTENT
    return config.skill_overrides.get(skill_name, config.default_mode)

  def _active_skills(
      self, state: Any, agent_name: str, invocation_id: str | None
  ) -> list[str]:
    """Returns the skills active for an agent this invocation.

    Read-only, so `get_tools` can call it with a read-only context. Expired
    skills are filtered out here and dropped from state by the next
    activation.

    Args:
      state: Session state to read.
      agent_name: The agent whose skills to report.
      invocation_id: The invocation asking. An ephemeral skill loaded in
        another one has expired.

    Returns:
      The active skill names, oldest activation first.
    """
    activated_skills = _read_activated_skills(state, agent_name)
    if not activated_skills or not self._tracks_ephemeral_skills:
      return activated_skills
    records = _read_lifecycle_records(state, agent_name)
    return [
        name
        for name in activated_skills
        if not _is_expired(records.get(name, {}), invocation_id)
    ]

  def _record_activation(
      self,
      state: Any,
      agent_name: str,
      skill_name: str,
      invocation_id: str | None,
      skill: models.Skill | None = None,
  ) -> list[str]:
    """Marks a skill active for an agent and releases what that displaces.

    Args:
      state: The session state to record activation in.
      agent_name: The agent the skill is being activated for.
      skill_name: The skill being activated.
      invocation_id: The invocation doing the activating, which is the turn an
        ephemeral skill gets.
      skill: The definition being activated, when the caller has it. Recorded
        as a digest, so a later request can tell that the skill has changed
        since.

    Returns:
      The skills released, oldest first: ephemeral leftovers from an earlier
      turn, then whatever the cap evicted.
    """
    stored_skills = _read_activated_skills(state, agent_name)
    activated_skills = self._active_skills(state, agent_name, invocation_id)
    # Not the skill being loaded: it is getting a fresh turn, so reporting it
    # as released in the same breath would contradict itself.
    expired = [
        name
        for name in stored_skills
        if name not in activated_skills and name != skill_name
    ]
    lifecycle = self._lifecycle_for(skill_name)

    if skill_name not in activated_skills:
      activated_skills.append(skill_name)
    elif lifecycle is SkillLifecycleMode.PERSISTENT:
      # The record may still need refreshing, which is what stops a reload
      # being re-stated.
      if not expired and not self._records_for(lifecycle, skill):
        return []
    else:
      # Reloading is a use: move it to the end so the cap spares it.
      activated_skills.remove(skill_name)
      activated_skills.append(skill_name)
    evicted = self._evict_over_cap(activated_skills)
    _write_activated_skills(state, agent_name, activated_skills)
    self._record_lifecycle(
        state, agent_name, skill_name, lifecycle, invocation_id, skill
    )
    self._forget_lifecycle_records(state, agent_name, activated_skills)
    return expired + evicted

  def _records_for(
      self, lifecycle: SkillLifecycleMode, skill: models.Skill | None
  ) -> dict[str, Any]:
    """Returns the lifecycle record to store for an activation, if any.

    A persistent skill nobody revalidates needs no bookkeeping, and the cap
    reads the activation order it already has.
    """
    record: dict[str, Any] = {}
    if lifecycle is SkillLifecycleMode.EPHEMERAL:
      record["lifecycle"] = lifecycle.value
    if self._revalidate_skills and skill is not None:
      record["version_hash"] = _skill_content_hash(skill)
    return record

  def _record_lifecycle(
      self,
      state: Any,
      agent_name: str,
      skill_name: str,
      lifecycle: SkillLifecycleMode,
      invocation_id: str | None,
      skill: models.Skill | None = None,
  ) -> None:
    """Notes what a skill was when it was loaded, and which turn that was."""
    record = self._records_for(lifecycle, skill)
    if not record:
      return
    if "lifecycle" in record:
      record["activated_in"] = invocation_id
    records = _read_lifecycle_records(state, agent_name)
    if "version_hash" not in record:
      # Re-activating without the definition in hand must not drop the digest
      # the skill was loaded with.
      loaded_hash = records.get(skill_name, {}).get("version_hash")
      if loaded_hash:
        record["version_hash"] = loaded_hash
    records[skill_name] = record
    state[_skill_lifecycle_state_key(agent_name)] = records

  def _forget_lifecycle_records(
      self, state: Any, agent_name: str, activated_skills: list[str]
  ) -> None:
    """Drops records for skills that are no longer active.

    The only place they are pruned. A request cannot do it: the tool context
    built there is thrown away with the request, so its state delta never
    reaches an event. A tool call's delta is committed, so cleanup rides along
    with the next activation.
    """
    state_key = _skill_lifecycle_state_key(agent_name)
    records = _read_lifecycle_records(state, agent_name)
    kept = {
        name: record
        for name, record in records.items()
        if name in activated_skills
    }
    if len(kept) != len(records):
      state[state_key] = kept

  def _evict_over_cap(self, activated_skills: list[str]) -> list[str]:
    """Drops the oldest bounded skills over the cap, editing the list in place.

    Only bounded skills count against `max_active_skills` and only they are
    evicted, so a persistent or ephemeral skill can neither be dropped nor push
    one out.
    """
    bounded = [
        name
        for name in activated_skills
        if self._lifecycle_for(name) is SkillLifecycleMode.BOUNDED
    ]
    overflow = len(bounded) - self._lifecycle_config.max_active_skills
    if overflow <= 0:
      return []

    evicted = bounded[:overflow]
    dropped = set(evicted)
    activated_skills[:] = [
        name for name in activated_skills if name not in dropped
    ]
    return evicted

  def clone_with_updated_skills(
      self, skills: list[models.Skill]
  ) -> SkillToolset:
    """Creates a new SkillToolset with identical configuration but modified skills."""
    additional_tools = (
        list(self._provided_tools_by_name.values()) + self._provided_toolsets
    )
    return SkillToolset(
        skills=skills,
        registry=self._registry,
        code_executor=self._code_executor,
        environment=self._env,
        skills_folder=self._skills_folder,
        script_timeout=self._script_timeout,
        additional_tools=additional_tools,
        tool_name_prefix=self.tool_name_prefix,
        tool_filter=self.tool_filter,
        discovery_mode=self._discovery_mode,
        lifecycle_config=self._lifecycle_config,
    )

  def _inject_catalog(self, selected_core_tools: set[str]) -> bool:
    """Whether to write the local catalog into the system instruction."""
    if self._discovery_mode is SkillDiscoveryMode.EAGER:
      return True
    if _LIST_SKILLS_TOOL_NAME in selected_core_tools:
      return False
    # A tool_filter that hides list_skills used to imply eager disclosure.
    # Kept so those callers keep a way to discover skills, but the mode is now
    # how you ask for this.
    # FutureWarning rather than DeprecationWarning so callers see it by default.
    if not self._warned_on_filtered_list_skills:
      self._warned_on_filtered_list_skills = True
      warnings.warn(
          "Filtering out `list_skills` to inject the skill catalog into the"
          " system instruction is deprecated. Pass"
          " `discovery_mode=SkillDiscoveryMode.EAGER` instead; a future release"
          " will let tool_filter remove the tool without changing the prompt.",
          FutureWarning,
          stacklevel=2,
      )
    return True

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Processes the outgoing LLM request to include available skills."""
    if self._env is not None and not self._env.is_initialized:
      await self._env.initialize()
    selected_core_tools = {
        t.name for t in self._tools if self._is_tool_selected(t, tool_context)
    }

    instructions = [
        _build_skill_system_instruction(
            prefix=self.tool_name_prefix,
            allowed_tools=selected_core_tools,
            skills_folder=self.skills_folder,
            script_execution_enabled=self._offers_script_tool(tool_context),
            unload_enabled=self._lifecycle_enabled,
        )
    ]

    if self._inject_catalog(selected_core_tools):
      skills = self._list_skills()
      skills_xml = prompt.format_skills_as_xml(skills)
      instructions.append(skills_xml)

    if self._registry and _SEARCH_SKILLS_TOOL_NAME in selected_core_tools:
      p = f"{self.tool_name_prefix}_" if self.tool_name_prefix else ""
      instructions.append(
          "\nIf the locally available skills are not sufficient to complete "
          f"your task, you can use the `{p}{_SEARCH_SKILLS_TOOL_NAME}` tool to"
          " discover additional skills from the registry."
      )

    if self._revalidate_skills:
      instructions.extend(await self._restate_changed_skills(tool_context))

    llm_request.append_instructions(instructions)

    if self._lifecycle_enabled:
      self._prune_unloaded_skills(tool_context, llm_request)

  def _prune_unloaded_skills(
      self, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Drops released skills' instructions from the outgoing request."""
    active_skills = set(self.list_active_skills(tool_context))
    p = f"{self.tool_name_prefix}_" if self.tool_name_prefix else ""
    pruned = _prune_unloaded_skill_instructions(
        llm_request.contents,
        f"{p}{_LOAD_SKILL_TOOL_NAME}",
        active_skills,
    )
    if pruned:
      logger.debug(
          "Pruned instructions for unloaded skills: %s", ", ".join(pruned)
      )

  async def _restate_changed_skills(
      self, tool_context: ToolContext
  ) -> list[str]:
    """Returns fresh instructions for active skills whose definition changed.

    A session can outlive several releases of the skills it loaded, and what
    the model was told is fixed in the transcript. Each request compares the
    current definition against the digest taken at load time.

    Only `load_skill` refreshes that digest, so a changed skill is re-stated
    on every request until the model reloads it -- the same text each time, so
    the prompt stays stable.

    Args:
      tool_context: Context of the request being built.

    Returns:
      One instruction block per changed skill, in activation order.
    """
    records = _read_lifecycle_records(
        tool_context.state, tool_context.agent_name
    )
    if not records:
      return []

    to_check = []
    for skill_name in self._active_skills(
        tool_context.state, tool_context.agent_name, tool_context.invocation_id
    ):
      loaded_hash = records.get(skill_name, {}).get("version_hash")
      if loaded_hash:
        to_check.append((skill_name, loaded_hash))
    if not to_check:
      return []

    # Together, so the first request of a turn does not wait out one registry
    # round trip per active skill in series.
    fetched = await asyncio.gather(
        *(
            self._get_or_fetch_skill(name, tool_context.invocation_id)
            for name, _ in to_check
        ),
        return_exceptions=True,
    )

    p = f"{self.tool_name_prefix}_" if self.tool_name_prefix else ""
    instructions = []
    for (skill_name, loaded_hash), skill in zip(to_check, fetched):
      if isinstance(skill, Exception):
        # A registry that is down is no reason to fail the turn.
        logger.warning(
            "Could not revalidate skill '%s': %s",
            skill_name,
            skill,
            exc_info=skill,
        )
        continue
      if isinstance(skill, BaseException):
        raise skill
      if skill is None or _skill_content_hash(skill) == loaded_hash:
        continue
      try:
        await self._rematerialize_skill(skill)
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Still worth re-stating the skill; only the sandbox copy is stale.
        logger.warning(
            "Could not refresh skill '%s' in the environment: %s",
            skill_name,
            e,
            exc_info=e,
        )
      logger.info(
          "Skill '%s' changed since it was loaded; re-stating it.", skill_name
      )
      instructions.append(
          f"\nThe skill `{skill_name}` has changed since it was loaded. These"
          " instructions replace the ones the earlier"
          f" `{p}{_LOAD_SKILL_TOOL_NAME}` response"
          f" gave you:\n\n{skill.instructions}"
      )
    return instructions

  async def _rematerialize_skill(self, skill: models.Skill) -> None:
    """Rewrites a changed skill's resources into the environment.

    Scripts are copied into the sandbox on first run and left there, so
    without this the model would run the old copy of a script it has just been
    re-briefed on.

    Only writes. `BaseEnvironment` has no delete, so a resource the new version
    dropped stays on disk; the re-stated instructions no longer mention it.
    """
    env = self._env
    skills_folder = self.skills_folder
    if env is None or skills_folder is None:
      return
    content_hash = _skill_content_hash(skill)
    if self._materialized_skill_hashes.get(skill.name) == content_hash:
      return
    if not env.is_initialized:
      await env.initialize()
    await _write_skill_resources_to_env(skill, skills_folder / skill.name, env)
    self._materialized_skill_hashes[skill.name] = content_hash

  @override
  async def close(self) -> None:
    """Performs cleanup and releases resources held by the toolset."""
    if self._env is not None and self._env.is_initialized:
      await self._env.close()
    for turn_cache in self._fetched_skill_cache.values():
      for cached in turn_cache.values():
        if isinstance(cached, asyncio.Future) and not cached.done():
          cached.cancel()
    self._fetched_skill_cache.clear()
    await super().close()


DEFAULT_SKILL_SYSTEM_INSTRUCTION = _build_skill_system_instruction()
