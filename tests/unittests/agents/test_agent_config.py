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

import logging
import ntpath
import os
from pathlib import Path
from textwrap import dedent
from typing import Any
from typing import Literal
from typing import Type
from unittest import mock
import warnings

from google.adk.agents import config_agent_utils
from google.adk.agents.agent_config import agent_config_discriminator
from google.adk.agents.agent_config import AgentConfig
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent_config import BaseAgentConfig
from google.adk.agents.common_configs import AgentRefConfig
from google.adk.agents.common_configs import CodeConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.llm_agent_config import LlmAgentConfig
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel
from pydantic import ValidationError
import pytest
import yaml


def test_agent_config_discriminator_default_is_llm_agent(tmp_path: Path):
  yaml_content = """\
name: search_agent
model: gemini-2.5-flash
description: a sample description
instruction: a fake instruction
tools:
  - name: google_search
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert config.root.agent_class == "LlmAgent"


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "LlmAgent",
        "google.adk.agents.LlmAgent",
        "google.adk.agents.llm_agent.LlmAgent",
    ],
)
def test_agent_config_discriminator_llm_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: search_agent
model: gemini-2.5-flash
description: a sample description
instruction: a fake instruction
tools:
  - name: google_search
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "LoopAgent",
        "google.adk.agents.LoopAgent",
        "google.adk.agents.loop_agent.LoopAgent",
    ],
)
def test_agent_config_discriminator_loop_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents: []
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LoopAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "ParallelAgent",
        "google.adk.agents.ParallelAgent",
        "google.adk.agents.parallel_agent.ParallelAgent",
    ],
)
def test_agent_config_discriminator_parallel_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents: []
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, ParallelAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    "agent_class_value",
    [
        "SequentialAgent",
        "google.adk.agents.SequentialAgent",
        "google.adk.agents.sequential_agent.SequentialAgent",
    ],
)
def test_agent_config_discriminator_sequential_agent(
    agent_class_value: str, tmp_path: Path
):
  yaml_content = f"""\
agent_class: {agent_class_value}
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
sub_agents: []
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, SequentialAgent)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    ("agent_class_value", "expected_agent_type"),
    [
        ("LoopAgent", LoopAgent),
        ("google.adk.agents.LoopAgent", LoopAgent),
        ("google.adk.agents.loop_agent.LoopAgent", LoopAgent),
        ("ParallelAgent", ParallelAgent),
        ("google.adk.agents.ParallelAgent", ParallelAgent),
        ("google.adk.agents.parallel_agent.ParallelAgent", ParallelAgent),
        ("SequentialAgent", SequentialAgent),
        ("google.adk.agents.SequentialAgent", SequentialAgent),
        ("google.adk.agents.sequential_agent.SequentialAgent", SequentialAgent),
    ],
)
def test_agent_config_discriminator_with_sub_agents(
    agent_class_value: str, expected_agent_type: Type[BaseAgent], tmp_path: Path
):
  # Create sub-agent config files
  sub_agent_dir = tmp_path / "sub_agents"
  sub_agent_dir.mkdir()
  sub_agent_config = """\
name: sub_agent_{index}
model: gemini-2.5-flash
description: a sub agent
instruction: sub agent instruction
"""
  (sub_agent_dir / "sub_agent1.yaml").write_text(
      sub_agent_config.format(index=1)
  )
  (sub_agent_dir / "sub_agent2.yaml").write_text(
      sub_agent_config.format(index=2)
  )
  yaml_content = f"""\
agent_class: {agent_class_value}
name: main_agent
description: main agent with sub agents
sub_agents:
  - config_path: sub_agents/sub_agent1.yaml
  - config_path: sub_agents/sub_agent2.yaml
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, expected_agent_type)
  assert config.root.agent_class == agent_class_value


@pytest.mark.parametrize(
    ("agent_class_value", "expected_agent_type"),
    [
        ("LlmAgent", LlmAgent),
        ("google.adk.agents.LlmAgent", LlmAgent),
        ("google.adk.agents.llm_agent.LlmAgent", LlmAgent),
    ],
)
def test_agent_config_discriminator_llm_agent_with_sub_agents(
    agent_class_value: str, expected_agent_type: Type[BaseAgent], tmp_path: Path
):
  # Create sub-agent config files
  sub_agent_dir = tmp_path / "sub_agents"
  sub_agent_dir.mkdir()
  sub_agent_config = """\
name: sub_agent_{index}
model: gemini-2.5-flash
description: a sub agent
instruction: sub agent instruction
"""
  (sub_agent_dir / "sub_agent1.yaml").write_text(
      sub_agent_config.format(index=1)
  )
  (sub_agent_dir / "sub_agent2.yaml").write_text(
      sub_agent_config.format(index=2)
  )
  yaml_content = f"""\
agent_class: {agent_class_value}
name: main_agent
model: gemini-2.5-flash
description: main agent with sub agents
instruction: main agent instruction
sub_agents:
  - config_path: sub_agents/sub_agent1.yaml
  - config_path: sub_agents/sub_agent2.yaml
"""
  config_file = tmp_path / "test_config.yaml"
  config_file.write_text(yaml_content)

  config = AgentConfig.model_validate(yaml.safe_load(yaml_content))
  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, expected_agent_type)
  assert config.root.agent_class == agent_class_value


def test_agent_config_model_code_resolves_preconfigured_client(tmp_path: Path):
  """model_code references a pre-built model instance by fully qualified name.

  Configured clients (custom api_base, etc.) must be constructed in Python
  and referenced from YAML; YAML cannot pass constructor arguments.
  """
  preconfigured = LiteLlm(
      model="kimi/k2", api_base="https://proxy.litellm.ai/v1"
  )

  yaml_content = """\
name: managed_api_agent
description: Agent using LiteLLM managed endpoint
instruction: Respond concisely.
model_code:
  name: my_library.clients.my_litellm
"""
  config_file = tmp_path / "litellm_agent.yaml"
  config_file.write_text(yaml_content)

  with mock.patch.object(
      config_agent_utils,
      "resolve_code_reference",
      return_value=preconfigured,
  ):
    agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert agent.model is preconfigured


def test_agent_config_discriminator_custom_agent():
  class MyCustomAgentConfig(BaseAgentConfig):
    agent_class: Literal["mylib.agents.MyCustomAgent"] = (
        "mylib.agents.MyCustomAgent"
    )
    other_field: str

  yaml_content = """\
agent_class: mylib.agents.MyCustomAgent
name: CodePipelineAgent
description: Executes a sequence of code writing, reviewing, and refactoring.
other_field: other value
"""
  config_data = yaml.safe_load(yaml_content)

  config = AgentConfig.model_validate(config_data)

  # pylint: disable=unidiomatic-typecheck Needs exact class matching.
  assert type(config.root) is BaseAgentConfig
  assert config.root.agent_class == "mylib.agents.MyCustomAgent"
  assert config.root.model_extra == {"other_field": "other value"}

  my_custom_config = MyCustomAgentConfig.model_validate(
      config.root.model_dump()
  )
  assert my_custom_config.other_field == "other value"


def test_from_config_passes_extra_yaml_fields_to_custom_agent_constructor(
    tmp_path: Path,
):
  """Custom agent fields in YAML reach the constructor without a custom config_type.

  Mirrors the 1.x AgentConfigMapper behavior: a custom agent subclass with
  extra Pydantic fields declared on the agent (not on a config_type) can
  populate those fields directly from YAML.
  """

  class MyCustomAgent(BaseAgent):
    custom_field: str = ""

  yaml_content = """\
agent_class: mylib.agents.MyCustomAgent
name: custom_agent
description: a custom agent
custom_field: hello from yaml
"""
  config_file = tmp_path / "custom_agent.yaml"
  config_file.write_text(yaml_content)

  with mock.patch.object(
      config_agent_utils,
      "resolve_fully_qualified_name",
      return_value=MyCustomAgent,
  ):
    agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, MyCustomAgent)
  assert agent.custom_field == "hello from yaml"


def test_from_config_ignores_extra_yaml_fields_not_on_agent(tmp_path: Path):
  """Extra YAML keys that don't map to constructor params are silently dropped."""

  class MyCustomAgent(BaseAgent):
    custom_field: str = ""

  yaml_content = """\
agent_class: mylib.agents.MyCustomAgent
name: custom_agent
description: a custom agent
custom_field: kept
unknown_field: dropped
"""
  config_file = tmp_path / "custom_agent.yaml"
  config_file.write_text(yaml_content)

  with mock.patch.object(
      config_agent_utils,
      "resolve_fully_qualified_name",
      return_value=MyCustomAgent,
  ):
    agent = config_agent_utils.from_config(str(config_file))

  assert agent.custom_field == "kept"
  assert not hasattr(agent, "unknown_field")


@pytest.mark.parametrize(
    ("config_rel_path", "child_rel_path", "child_name", "instruction"),
    [
        (
            Path("main.yaml"),
            Path("sub_agents/child.yaml"),
            "child_agent",
            "I am a child agent",
        ),
        (
            Path("level1/level2/nested_main.yaml"),
            Path("sub/nested_child.yaml"),
            "nested_child",
            "I am nested",
        ),
    ],
)
def test_resolve_agent_reference_resolves_relative_paths(
    config_rel_path: Path,
    child_rel_path: Path,
    child_name: str,
    instruction: str,
    tmp_path: Path,
):
  """Verify resolve_agent_reference resolves relative sub-agent paths."""
  config_file = tmp_path / config_rel_path
  config_file.parent.mkdir(parents=True, exist_ok=True)

  child_config_path = config_file.parent / child_rel_path
  child_config_path.parent.mkdir(parents=True, exist_ok=True)
  child_config_path.write_text(dedent(f"""
          agent_class: LlmAgent
          name: {child_name}
          model: gemini-2.5-flash
          instruction: {instruction}
          """).lstrip())

  config_file.write_text(dedent(f"""
          agent_class: LlmAgent
          name: main_agent
          model: gemini-2.5-flash
          instruction: I am the main agent
          sub_agents:
            - config_path: {child_rel_path.as_posix()}
          """).lstrip())

  ref_config = AgentRefConfig(config_path=child_rel_path.as_posix())
  agent = config_agent_utils.resolve_agent_reference(
      ref_config, str(config_file)
  )

  assert agent.name == child_name

  config_dir = os.path.dirname(str(config_file.resolve()))
  assert config_dir == str(config_file.parent.resolve())

  expected_child_path = os.path.join(config_dir, *child_rel_path.parts)
  assert os.path.exists(expected_child_path)


def test_resolve_agent_reference_uses_windows_dirname():
  """Ensure Windows-style config references resolve via os.path.dirname."""
  ref_config = AgentRefConfig(config_path="sub\\child.yaml")
  recorded: dict[str, str] = {}

  def fake_from_config(path: str):
    recorded["path"] = path
    return "sentinel"

  with (
      mock.patch.object(
          config_agent_utils,
          "from_config",
          autospec=True,
          side_effect=fake_from_config,
      ),
      mock.patch.object(config_agent_utils.os, "path", ntpath),
  ):
    referencing = r"C:\workspace\agents\main.yaml"
    result = config_agent_utils.resolve_agent_reference(ref_config, referencing)

  expected_path = ntpath.join(
      ntpath.dirname(referencing), ref_config.config_path
  )
  assert result == "sentinel"
  assert recorded["path"] == expected_path


def test_resolve_agent_reference_blocks_absolute_path():
  """Verify resolve_agent_reference raises ValueError for absolute paths."""
  ref_config = AgentRefConfig(config_path="/etc/passwd")
  with pytest.raises(
      ValueError,
      match="Absolute paths are not allowed in AgentRefConfig config_path",
  ):
    config_agent_utils.resolve_agent_reference(
        ref_config, "/workspace/main.yaml"
    )


def test_resolve_agent_reference_blocks_path_traversal():
  """Verify resolve_agent_reference raises ValueError for path traversal."""
  ref_config = AgentRefConfig(config_path="../outside.yaml")
  with pytest.raises(ValueError, match="Path traversal detected"):
    config_agent_utils.resolve_agent_reference(
        ref_config, "/workspace/agents/main.yaml"
    )


# --- Security tests: module blocklist for YAML agent config code references ---


def test_resolve_code_reference_blocks_os_when_enforced():
  """Verify resolve_code_reference blocks os module directly."""
  from google.adk.agents.common_configs import CodeConfig

  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils.resolve_code_reference(CodeConfig(name="os.system"))


def test_resolve_code_reference_wraps_a_name_that_does_not_resolve():
  """An unresolvable reference raises ValueError, as the docstring promises.

  The mapper routes `Callable`-typed fields through here, so a bad name in
  YAML must not surface as a raw ModuleNotFoundError to callers that have
  always been told to expect ValueError.
  """
  with pytest.raises(ValueError, match="Invalid fully qualified name"):
    config_agent_utils.resolve_code_reference(
        CodeConfig(name="invalid.tool.name")
    )


def test_resolve_code_reference_wraps_a_missing_attribute():
  """A real module with no such symbol is unresolvable in the same way."""
  with pytest.raises(ValueError, match="Invalid fully qualified name"):
    config_agent_utils.resolve_code_reference(
        CodeConfig(name="google.adk.agents.config_agent_utils.no_such_symbol")
    )


def test_resolve_fully_qualified_name_blocks_subprocess_when_enforced():
  """Verify resolve_fully_qualified_name blocks subprocess module.

  resolve_fully_qualified_name wraps all exceptions in
  ValueError("Invalid fully qualified name: ..."), so we check the wrapper
  and verify the __cause__ carries the blocklist message.
  """
  with pytest.raises(
      ValueError, match="Invalid fully qualified name"
  ) as exc_info:
    config_agent_utils.resolve_fully_qualified_name("subprocess.Popen")
  assert "Blocked module reference" in str(exc_info.value.__cause__)


def test_allowed_module_passes_when_enforced(tmp_path: Path):
  """Verify that google.adk modules are NOT blocked by the module denylist."""
  # This should NOT raise — google.adk modules must remain allowed
  result = config_agent_utils.resolve_fully_qualified_name(
      "google.adk.agents.llm_agent.LlmAgent"
  )
  assert result is LlmAgent


@pytest.mark.parametrize(
    "blocked_module",
    [
        "os.system",
        "posix.system",
        "nt.system",
        "subprocess.call",
        "_posixsubprocess.fork_exec",
        "socket.socket",
        "_socket.socket",
        "builtins.exec",
    ],
)
def test_resolve_agent_code_reference_blocks_when_enforced(
    blocked_module: str,
):
  """Verify _resolve_agent_code_reference blocks dangerous modules."""
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils._resolve_agent_code_reference(blocked_module)


@pytest.mark.parametrize(
    "blocked_ref",
    [
        "os.system",
        "posix.system",
        "nt.system",
        "subprocess.call",
        "_posixsubprocess.fork_exec",
        "socket.socket",
        "_socket.socket",
        "builtins.exec",
        "pickle.loads",
    ],
)
def test_resolve_tools_blocks_dangerous_modules(blocked_ref: str):
  """Verify _resolve_tools blocks dangerous modules for user-defined tools."""
  from google.adk.tools.tool_configs import ToolConfig

  tool_config = ToolConfig(name=blocked_ref)
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils._AgentConfigMapper("/fake/path.yaml")._resolve_tools(
        [tool_config]
    )


def test_resolve_tools_allows_builtin_adk_tools():
  """Verify _resolve_tools allows ADK built-in tools (no dot in name)."""
  from google.adk.tools.tool_configs import ToolConfig

  # Built-in tools have no dot — they import from google.adk.tools
  tool_config = ToolConfig(name="google_search")
  # Should NOT raise — this is a safe, hardcoded import path
  mapper = config_agent_utils._AgentConfigMapper("/fake/path.yaml")
  resolved = mapper._resolve_tools([tool_config])
  assert len(resolved) == 1


@pytest.mark.parametrize(
    "blocked_ref",
    [
        "ftplib.FTP",
        "smtplib.SMTP",
        "xmlrpc.client",
        "telnetlib.Telnet",
        "poplib.POP3",
        "imaplib.IMAP4",
        "asyncio.run",
        "pathlib.Path",
    ],
)
def test_newly_blocked_network_modules_are_rejected(blocked_ref: str):
  """Verify newly added network-capable modules are blocked.

  resolve_fully_qualified_name wraps errors, so we check the cause.
  """
  with pytest.raises(
      ValueError, match="Invalid fully qualified name"
  ) as exc_info:
    config_agent_utils.resolve_fully_qualified_name(blocked_ref)
  assert "Blocked module reference" in str(exc_info.value.__cause__)


# Standard library functions that will run whatever code you hand them. The old
# denylist happened to list profile but not cProfile, and missed all the rest.
# One entry per module, since the check only looks at the top-level name.
_EXEC_CAPABLE_STDLIB_REFS = [
    "cProfile.run",
    "profile.run",
    "timeit.timeit",
    "pydoc.pipepager",
    "trace.Trace",
    "doctest.testmod",
    "bdb.Bdb",
    "py_compile.compile",
]

# These are not in sys.stdlib_module_names on every Python we support, so
# _BLOCKED_MODULES is the only thing rejecting them.
_LOAD_BEARING_NON_STDLIB_REFS = [
    "distutils.spawn.spawn",
    "test.support.script_helper.spawn_python",
    "_testcapi.run_stringflags",
    "pipes.quote",
    "telnetlib.Telnet",
]


@pytest.mark.parametrize("blocked_ref", _EXEC_CAPABLE_STDLIB_REFS)
def test_resolve_code_reference_blocks_exec_capable_stdlib(blocked_ref: str):
  """Exec-capable stdlib modules are rejected as code references."""
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils.resolve_code_reference(CodeConfig(name=blocked_ref))


@pytest.mark.parametrize("blocked_ref", _EXEC_CAPABLE_STDLIB_REFS)
def test_resolve_tools_blocks_exec_capable_stdlib(blocked_ref: str):
  """Exec-capable stdlib modules are rejected as user-defined tools.

  This is the path the reported exploit takes: upload an agent YAML whose only
  tool is `cProfile.run`, then replay a saved test session, which dispatches a
  recorded functionCall straight to the resolved tool.
  """
  from google.adk.tools.tool_configs import ToolConfig

  tool_config = ToolConfig(name=blocked_ref)
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils._AgentConfigMapper("/fake/path.yaml")._resolve_tools(
        [tool_config]
    )


@pytest.mark.parametrize(
    "blocked_ref",
    [
        "json.loads",
        "base64.b64decode",
        "string.capwords",
        "gc.collect",
        "operator.attrgetter",
    ],
)
def test_harmless_looking_stdlib_modules_are_also_blocked(blocked_ref: str):
  """The whole standard library is off-limits, not just the scary parts.

  Blocking all of it is what keeps this closed against ways to run code that
  future Python releases add.
  """
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils.resolve_code_reference(CodeConfig(name=blocked_ref))


@pytest.mark.parametrize("blocked_ref", _LOAD_BEARING_NON_STDLIB_REFS)
def test_modules_dropped_from_the_stdlib_are_still_blocked(blocked_ref: str):
  """Covers the modules the standard library rule misses.

  They stay importable from a shim or a PyPI backport, so without the explicit
  denylist they come back as a way to run code.
  """
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils.resolve_code_reference(CodeConfig(name=blocked_ref))


def test_third_party_module_reference_is_not_blocked():
  """Non-stdlib packages stay resolvable so integrations keep working.

  A compatibility guarantee for integrations like langchain, not a security
  assertion: third-party packages are still resolvable by name.
  """
  result = config_agent_utils.resolve_fully_qualified_name("pydantic.BaseModel")
  assert result is BaseModel


# yaml is a hard, always-installed dependency of adk-python itself (not an
# optional integration), and ruamel is a common transitive dependency.
# Both ship exec-capable deserialization entry points.
_YAML_UNSAFE_LOADER_REFS = [
    "yaml.unsafe_load",
    "yaml.load",
    "yaml.full_load",
]

_RUAMEL_UNSAFE_LOADER_REFS = [
    "ruamel.yaml.round_trip_load",
]


@pytest.mark.parametrize(
    "blocked_ref", _YAML_UNSAFE_LOADER_REFS + _RUAMEL_UNSAFE_LOADER_REFS
)
def test_resolve_code_reference_blocks_yaml_and_ruamel_deserialization(
    blocked_ref: str,
):
  """yaml and ruamel's unsafe/full loaders are rejected as code references."""
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils.resolve_code_reference(CodeConfig(name=blocked_ref))


@pytest.mark.parametrize(
    "blocked_ref", _YAML_UNSAFE_LOADER_REFS + _RUAMEL_UNSAFE_LOADER_REFS
)
def test_resolve_tools_blocks_yaml_and_ruamel_deserialization(blocked_ref: str):
  """yaml and ruamel's unsafe/full loaders are rejected as user-defined tools."""
  from google.adk.tools.tool_configs import ToolConfig

  tool_config = ToolConfig(name=blocked_ref)
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils._AgentConfigMapper("/fake/path.yaml")._resolve_tools(
        [tool_config]
    )


_YAML_SAFE_LOOKING_REFS = [
    "yaml.safe_load",
    "yaml.dump",
    "yaml.SafeLoader",
]

_RUAMEL_SAFE_LOOKING_REFS = [
    "ruamel.yaml.safe_load",
    "ruamel.yaml.dump",
]


@pytest.mark.parametrize(
    "blocked_ref", _YAML_SAFE_LOOKING_REFS + _RUAMEL_SAFE_LOOKING_REFS
)
def test_harmless_looking_yaml_and_ruamel_references_are_also_blocked(
    blocked_ref: str,
):
  """The whole yaml and ruamel modules are off-limits, not just the scary parts.

  Blocking them in full locks in the module-wide intent and prevents
  future bypasses if new unsafe loaders are added or if safe-looking names
  are refactored to resolve differently.
  """
  with pytest.raises(ValueError, match="Blocked module reference"):
    config_agent_utils.resolve_code_reference(CodeConfig(name=blocked_ref))


def test_denylist_can_be_disabled():
  """Verify _set_enforce_denylist(False) disables module blocking."""
  config_agent_utils._set_enforce_denylist(False)
  try:
    # os.getcwd is a real, importable reference — should succeed
    result = config_agent_utils.resolve_fully_qualified_name("os.getcwd")
    assert callable(result)
  finally:
    config_agent_utils._set_enforce_denylist(True)


def test_from_config_blocks_args_key_when_enforced(tmp_path: Path):
  """Loading blocks the 'args' key when enforcement is on."""
  config_file = tmp_path / "agent.yaml"
  config_file.write_text("name: my_agent\nargs:\n  key: value\n")
  config_agent_utils._set_enforce_yaml_key_denylist(True)
  try:
    with pytest.raises(ValueError) as exc_info:
      config_agent_utils.from_config(str(config_file))
    assert "Blocked key 'args' found" in str(exc_info.value)
  finally:
    config_agent_utils._set_enforce_yaml_key_denylist(False)


# --- Discriminator contract ---------------------------------------------


@pytest.mark.parametrize(
    ("config_data", "expected_tag"),
    [
        ({"agent_class": "LlmAgent"}, "LlmAgent"),
        ({"agent_class": "LoopAgent"}, "LoopAgent"),
        ({"agent_class": "ParallelAgent"}, "ParallelAgent"),
        ({"agent_class": "SequentialAgent"}, "SequentialAgent"),
        # Omitting agent_class means LlmAgent, per the field's documentation.
        ({"name": "no_agent_class"}, "LlmAgent"),
        # Anything the framework does not own falls back to the open-ended
        # BaseAgentConfig, which keeps the unknown keys in model_extra.
        ({"agent_class": "mylib.agents.MyAgent"}, "BaseAgent"),
        # A fully qualified name for a built-in class is still user-defined as
        # far as the union is concerned: only the bare names are tagged.
        ({"agent_class": "google.adk.agents.LlmAgent"}, "BaseAgent"),
    ],
)
def test_agent_config_discriminator_maps_agent_class_to_tag(
    config_data: dict, expected_tag: str
):
  """The discriminator picks the union member from the agent_class key."""
  assert agent_config_discriminator(config_data) == expected_tag


@pytest.mark.parametrize(
    "malformed_config",
    [None, "name: my_agent", [{"name": "my_agent"}], 42],
)
def test_agent_config_discriminator_rejects_non_mapping(malformed_config: Any):
  """A config that is not a mapping has no agent_class and must be rejected."""
  with pytest.raises(ValueError, match="Invalid agent config"):
    agent_config_discriminator(malformed_config)


def test_from_config_rejects_empty_yaml_file(tmp_path: Path):
  """An empty YAML file loads as None; it must not be treated as an LlmAgent."""
  config_file = tmp_path / "empty.yaml"
  config_file.write_text("")

  with pytest.raises(ValueError, match="Invalid agent config"):
    config_agent_utils.from_config(str(config_file))


# --- AgentRefConfig exactly-one-of validation ---------------------------


def test_agent_ref_config_rejects_both_code_and_config_path():
  """A reference naming both sources is ambiguous and must be rejected."""
  with pytest.raises(
      ValueError, match="Only one of `code` or `config_path` should be provided"
  ):
    AgentRefConfig(code="my_library.agents.my_agent", config_path="sub.yaml")


def test_agent_ref_config_rejects_neither_code_nor_config_path():
  """A reference naming no source points at nothing and must be rejected."""
  with pytest.raises(
      ValueError,
      match="Exactly one of `code` or `config_path` must be provided",
  ):
    AgentRefConfig()


@pytest.mark.parametrize(
    ("kwargs", "expected_code", "expected_config_path"),
    [
        (
            {"code": "my_library.agents.my_agent"},
            "my_library.agents.my_agent",
            None,
        ),
        ({"config_path": "sub.yaml"}, None, "sub.yaml"),
    ],
)
def test_agent_ref_config_accepts_exactly_one_source(
    kwargs: dict, expected_code: str, expected_config_path: str
):
  """Exactly one source is the valid shape, and the other stays None."""
  ref_config = AgentRefConfig(**kwargs)

  assert ref_config.code == expected_code
  assert ref_config.config_path == expected_config_path


# --- LlmAgentConfig validation ------------------------------------------


def test_llm_agent_config_rejects_model_and_model_code_together():
  """`model` and `model_code` are two ways to say the same thing."""
  with pytest.raises(
      ValueError,
      match=(
          r"Only one of `model` or `model_code` should be set, but both were"
          r" provided\. Got model='gemini-2\.5-flash' and"
          r" model_code=CodeConfig\(name='my_library\.clients\.my_litellm'\)\."
      ),
  ):
    LlmAgentConfig(
        name="my_agent",
        instruction="do the thing",
        model="gemini-2.5-flash",
        model_code=CodeConfig(name="my_library.clients.my_litellm"),
    )


def test_llm_agent_config_rejects_misspelled_field():
  """A typo in a YAML key must fail loudly rather than be silently dropped."""
  with pytest.raises(ValueError, match="instructions"):
    LlmAgentConfig(
        name="my_agent",
        instruction="do the thing",
        instructions="do the other thing",
    )


def test_llm_agent_config_minimal_defaults():
  """A config with only the required keys carries the documented defaults."""
  config = LlmAgentConfig(name="my_agent", instruction="do the thing")

  # agent_class must stay the bare built-in name: the discriminator only
  # recognises "LlmAgent", so any other default would route this config to
  # BaseAgentConfig instead.
  assert config.agent_class == "LlmAgent"
  assert config.include_contents == "default"
  assert config.model is None
  assert config.model_code is None
  assert config.tools is None


# --- LoopAgentConfig round trip -----------------------------------------


@pytest.mark.parametrize("max_iterations", [3, 0])
def test_loop_agent_config_max_iterations_reaches_the_agent(
    tmp_path: Path, max_iterations: int
):
  """max_iterations is LoopAgentConfig's only own field; it must round trip."""
  config_file = tmp_path / "loop.yaml"
  config_file.write_text(
      "agent_class: LoopAgent\n"
      "name: looper\n"
      "description: repeats its sub agents\n"
      f"max_iterations: {max_iterations}\n"
      "sub_agents: []\n"
  )

  agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LoopAgent)
  assert agent.max_iterations == max_iterations


# --- resolve_callbacks ---------------------------------------------------


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (
            [
                "google.adk.agents.llm_agent.LlmAgent",
                "google.adk.agents.loop_agent.LoopAgent",
            ],
            [LlmAgent, LoopAgent],
        ),
        (
            [
                "google.adk.agents.loop_agent.LoopAgent",
                "google.adk.agents.llm_agent.LlmAgent",
            ],
            [LoopAgent, LlmAgent],
        ),
    ],
)
def test_resolve_callbacks_preserves_config_order(
    names: list[str], expected: list[type]
):
  """Callback order is the invocation order, so resolution must not reorder."""
  resolved = config_agent_utils.resolve_callbacks(
      [CodeConfig(name=name) for name in names]
  )

  assert resolved == expected


def test_resolve_callbacks_with_no_configs_returns_empty_list():
  """No configured callbacks means no callbacks, not None."""
  assert config_agent_utils.resolve_callbacks([]) == []


# --- Workflow & FunctionNode YAML Configuration Tests ---------------------


def test_from_config_loads_function_node_with_func_code(tmp_path: Path):
  """FunctionNode can be loaded from YAML specifying func_code."""
  from google.adk.workflow import FunctionNode

  config_file = tmp_path / "func_node.yaml"
  config_file.write_text(
      "agent_class: FunctionNode\n"
      "name: my_func_node\n"
      "description: A function node\n"
      "func_code: google.adk.agents.config_agent_utils.from_config\n"
  )

  node = config_agent_utils.from_config(str(config_file))
  assert isinstance(node, FunctionNode)
  assert node.name == "my_func_node"
  assert node._func == config_agent_utils.from_config
  # FunctionNode's constructor takes no `description` (it derives one from the
  # function docstring), so this is the field the mapper has to apply after
  # construction rather than drop -- the YAML sets it, so it must win.
  assert node.description == "A function node"


def test_from_config_invokes_overridden_parse_config(tmp_path: Path):
  """A subclass overriding `_parse_config` still gets the last word on kwargs.

  Reflection replaced the per-class parsers, not this hook, so a subclass that
  overrides it keeps working -- and keeps being handed a parsed config model
  rather than the raw mapping.
  """
  received = {}

  class MyCustomAgent(BaseAgent):
    custom_field: str = ""

    @classmethod
    def _parse_config(cls, config, config_abs_path, kwargs):
      received["config"] = config
      kwargs["custom_field"] = "set by _parse_config"
      return kwargs

  yaml_content = """\
agent_class: mylib.agents.MyCustomAgent
name: custom_agent
custom_field: from yaml
"""
  config_file = tmp_path / "custom_agent.yaml"
  config_file.write_text(yaml_content)

  with mock.patch.object(
      config_agent_utils,
      "resolve_fully_qualified_name",
      return_value=MyCustomAgent,
  ):
    agent = config_agent_utils.from_config(str(config_file))

  assert agent.custom_field == "set by _parse_config"
  assert isinstance(received["config"], BaseAgentConfig)
  assert received["config"].name == "custom_agent"


def test_from_config_warns_when_parse_config_is_overridden(tmp_path: Path):
  """The override is what is deprecated, so the warning has to name it.

  The `@deprecated` on the base implementation cannot do this: an override is
  a different, undecorated function, and the base one is never called.
  """

  class MyCustomAgent(BaseAgent):

    @classmethod
    def _parse_config(cls, config, config_abs_path, kwargs):
      return kwargs

  yaml_content = """\
agent_class: mylib.agents.MyCustomAgent
name: custom_agent
"""
  config_file = tmp_path / "custom_agent.yaml"
  config_file.write_text(yaml_content)

  with mock.patch.object(
      config_agent_utils,
      "resolve_fully_qualified_name",
      return_value=MyCustomAgent,
  ):
    with pytest.warns(
        DeprecationWarning, match=r"MyCustomAgent\._parse_config is deprecated"
    ):
      config_agent_utils.from_config(str(config_file))


def test_from_config_does_not_warn_without_a_parse_config_override(
    tmp_path: Path,
):
  """Inheriting the no-op hook is not deprecated usage and must stay quiet."""
  yaml_content = """\
name: search_agent
model: gemini-2.5-flash
instruction: search
"""
  config_file = tmp_path / "root_agent.yaml"
  config_file.write_text(yaml_content)

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    agent = config_agent_utils.from_config(str(config_file))

  assert isinstance(agent, LlmAgent)
  assert not [
      w for w in caught if "_parse_config is deprecated" in str(w.message)
  ]


def test_from_config_warns_about_a_key_a_node_class_cannot_take(
    tmp_path: Path, caplog
):
  """A node class declares no `config_type`, so reflection is the only gate.

  Without this the misspelled key is skipped in silence and the node comes back
  missing whatever it was meant to set.
  """
  config_file = tmp_path / "root_agent.yaml"
  config_file.write_text("agent_class: Workflow\nname: wf\nmax_concurrncy: 3\n")

  with caplog.at_level(logging.WARNING, logger="google_adk"):
    wf = config_agent_utils.from_config(str(config_file))

  assert wf.max_concurrency is None
  assert "does not accept ['max_concurrncy']" in caplog.text


def test_from_config_stays_quiet_for_keys_the_class_accepts(
    tmp_path: Path, caplog
):
  """Only keys reflection could not place are reported."""
  config_file = tmp_path / "root_agent.yaml"
  config_file.write_text(
      "name: search_agent\nmodel: gemini-2.5-flash\ninstruction: search\n"
  )

  with caplog.at_level(logging.WARNING, logger="google_adk"):
    config_agent_utils.from_config(str(config_file))

  assert "does not accept" not in caplog.text


def test_from_config_keeps_a_raw_json_schema_on_a_node(tmp_path: Path):
  """`BaseNode.input_schema` is a `SchemaUnion`, whose first member is `dict`.

  Treating every mapping as a code reference to import made a valid config die
  with a CodeConfig validation error instead.
  """
  config_file = tmp_path / "root_agent.yaml"
  config_file.write_text(
      "agent_class: Workflow\n"
      "name: wf\n"
      "input_schema:\n"
      "  type: object\n"
      "  properties:\n"
      "    x:\n"
      "      type: string\n"
  )

  wf = config_agent_utils.from_config(str(config_file))

  assert wf.input_schema == {
      "type": "object",
      "properties": {"x": {"type": "string"}},
  }


def test_from_config_delegates_to_a_subclass_that_overrides_from_config(
    tmp_path: Path,
):
  """A subclass owning its construction is handed the config, not built here.

  The mapper path would build the class itself and never reach the override.
  """
  seen = {}

  class OwnBuildAgent(BaseAgent):

    @classmethod
    def from_config(cls, config, config_abs_path):
      seen["config"] = config
      seen["path"] = config_abs_path
      return cls(name="built_by_override")

  config_file = tmp_path / "root_agent.yaml"
  config_file.write_text(
      "agent_class: mylib.OwnBuildAgent\nname: ignored_by_the_override\n"
  )

  with mock.patch.object(
      config_agent_utils,
      "resolve_fully_qualified_name",
      return_value=OwnBuildAgent,
  ):
    agent = config_agent_utils.from_config(str(config_file))

  assert agent.name == "built_by_override"
  assert isinstance(seen["config"], BaseAgentConfig)
  assert seen["path"] == str(config_file)


def test_from_config_resolves_a_leading_dot_against_the_config_directory(
    tmp_path: Path,
):
  """A `*_code` value starting with `.` is relative to the config's own package.

  `.my_model` inside `<...>/mypkg/root_agent.yaml` means `mypkg.my_model`.
  """
  pkg_dir = tmp_path / "mypkg"
  pkg_dir.mkdir()
  config_file = pkg_dir / "root_agent.yaml"
  config_file.write_text(
      "name: dotted\ninstruction: hi\nmodel_code:\n  name: .my_model\n"
  )

  seen = {}

  def fake_resolve(code_config):
    seen["name"] = code_config.name
    return "resolved-model"

  with mock.patch.object(
      config_agent_utils, "resolve_code_reference", fake_resolve
  ):
    agent = config_agent_utils.from_config(str(config_file))

  assert seen["name"] == "mypkg.my_model"
  assert agent.model == "resolved-model"


def test_resolve_code_reference_keeps_a_missing_dependency_as_itself(
    tmp_path: Path, monkeypatch
):
  """A module that imports something missing is not a bad reference.

  Rewriting its ModuleNotFoundError into "invalid fully qualified name" would
  point the reader at the config instead of at the absent package.
  """
  (tmp_path / "userpkg.py").write_text(
      "import a_dependency_that_is_not_installed\nthing = 1\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))

  with pytest.raises(ModuleNotFoundError) as exc_info:
    config_agent_utils.resolve_code_reference(CodeConfig(name="userpkg.thing"))
  assert "a_dependency_that_is_not_installed" in str(exc_info.value)


def test_from_config_validates_a_deferred_field(tmp_path: Path):
  """Fields held back from the constructor are still type-checked.

  They are applied after construction, so a plain `setattr` would be the one
  path into the node that pydantic never sees.
  """
  config_file = tmp_path / "root_agent.yaml"
  config_file.write_text(
      "agent_class: FunctionNode\n"
      "name: fn\n"
      "description: [1, 2, 3]\n"
      "func_code: google.adk.agents.config_agent_utils.from_config\n"
  )

  with pytest.raises(ValidationError, match="description"):
    config_agent_utils.from_config(str(config_file))


def test_from_config_loads_workflow_with_routing_map(tmp_path: Path):
  """Workflow can be loaded from YAML with conditional routing maps."""
  from google.adk.workflow import Workflow

  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: routing_pipeline\n"
      "edges:\n"
      "  - - START\n"
      "    - agent_class: LlmAgent\n"
      "      name: classifier\n"
      "      model: gemini-2.5-flash\n"
      "      instruction: classify\n"
      "    - tech:\n"
      "        agent_class: LlmAgent\n"
      "        name: tech_agent\n"
      "        model: gemini-2.5-flash\n"
      "        instruction: tech\n"
      "      other:\n"
      "        agent_class: LlmAgent\n"
      "        name: other_agent\n"
      "        model: gemini-2.5-flash\n"
      "        instruction: other\n"
  )

  wf = config_agent_utils.from_config(str(wf_file))
  assert isinstance(wf, Workflow)
  assert wf.name == "routing_pipeline"
  assert len(wf.graph.edges) == 3
  routes = {e.route: e.to_node.name for e in wf.graph.edges if e.route}
  assert routes == {"tech": "tech_agent", "other": "other_agent"}


def test_from_config_loads_workflow_with_fan_out_and_join(tmp_path: Path):
  """Workflow can be loaded from YAML with fan-out and join edges."""
  from google.adk.workflow import Workflow

  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: fan_out_pipeline\n"
      "edges:\n"
      "  - - START\n"
      "    - - agent_class: LlmAgent\n"
      "        name: worker_1\n"
      "        model: gemini-2.5-flash\n"
      "        instruction: worker 1\n"
      "      - agent_class: LlmAgent\n"
      "        name: worker_2\n"
      "        model: gemini-2.5-flash\n"
      "        instruction: worker 2\n"
      "    - agent_class: LlmAgent\n"
      "      name: aggregator\n"
      "      model: gemini-2.5-flash\n"
      "      instruction: aggregate\n"
  )

  wf = config_agent_utils.from_config(str(wf_file))
  assert isinstance(wf, Workflow)
  assert wf.name == "fan_out_pipeline"
  assert len(wf.graph.edges) == 4
  to_nodes = [e.to_node.name for e in wf.graph.edges]
  assert to_nodes.count("aggregator") == 2


def test_from_config_loads_workflow_with_linear_chain(tmp_path: Path):
  """Workflow can be loaded from YAML with linear edge chains."""
  from google.adk.workflow import Workflow

  sub_agent_file = tmp_path / "processor.yaml"
  sub_agent_file.write_text(
      "agent_class: LlmAgent\n"
      "name: processor\n"
      "model: gemini-2.5-flash\n"
      "instruction: process input\n"
  )

  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: linear_pipeline\n"
      "edges:\n"
      "  - - START\n"
      "    - processor.yaml\n"
  )

  wf = config_agent_utils.from_config(str(wf_file))
  assert isinstance(wf, Workflow)
  assert wf.name == "linear_pipeline"
  assert wf.graph is not None
  assert len(wf.graph.edges) == 1
  assert wf.graph.edges[0].from_node.name == "__START__"
  assert wf.graph.edges[0].to_node.name == "processor"


def test_from_config_workflow_node_caching_across_edges(tmp_path: Path):
  """Subsequent edges referencing the same node name use the same instance."""
  from google.adk.workflow import Workflow

  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: looping_workflow\n"
      "edges:\n"
      "  - - START\n"
      "    - agent_class: LlmAgent\n"
      "      name: loop_node\n"
      "      model: gemini-2.5-flash\n"
      "      instruction: loop\n"
      "    - agent_class: LlmAgent\n"
      "      name: checker\n"
      "      model: gemini-2.5-flash\n"
      "      instruction: check\n"
      "  - - checker\n"
      "    - repeat: loop_node\n"
  )

  wf = config_agent_utils.from_config(str(wf_file))
  assert isinstance(wf, Workflow)
  # Find loop_node from first chain and from repeat route
  loop_node_from_start = next(
      e.to_node for e in wf.graph.edges if e.from_node.name == "__START__"
  )
  loop_node_from_repeat = next(
      e.to_node for e in wf.graph.edges if e.route == "repeat"
  )
  assert loop_node_from_start is loop_node_from_repeat


def test_from_config_keeps_a_function_node_subclass(tmp_path: Path):
  """`agent_class` naming a FunctionNode subclass builds that class.

  Constructing `FunctionNode` directly downgraded the node, and skipping the
  mapper left the other keys unresolved -- `description` here never landed.
  """

  from google.adk.workflow import FunctionNode

  class MyFuncNode(FunctionNode):
    pass

  def a_func(x):
    return x

  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: wf\n"
      "edges:\n"
      "  - - START\n"
      "    - agent_class: mylib.MyFuncNode\n"
      "      name: fn1\n"
      "      description: from yaml\n"
      "      func_code: mylib.a_func\n"
  )

  def fake_resolve(name):
    return {"mylib.MyFuncNode": MyFuncNode, "mylib.a_func": a_func}[name]

  with mock.patch.object(
      config_agent_utils, "resolve_fully_qualified_name", fake_resolve
  ):
    with mock.patch.object(
        config_agent_utils,
        "resolve_code_reference",
        lambda cc: fake_resolve(cc.name),
    ):
      wf = config_agent_utils.from_config(str(wf_file))

  node = wf.graph.edges[0].to_node
  assert type(node) is MyFuncNode
  assert node.description == "from yaml"


def test_from_config_reads_a_routing_map_whose_route_is_called_name(
    tmp_path: Path,
):
  """`name` is a legal route value, not only an inline-node key."""
  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: wf\n"
      "edges:\n"
      "  - - START\n"
      "    - agent_class: LlmAgent\n"
      "      name: classifier\n"
      "      model: gemini-2.5-flash\n"
      "      instruction: classify\n"
      "    - name:\n"
      "        agent_class: LlmAgent\n"
      "        name: target_a\n"
      "        model: gemini-2.5-flash\n"
      "        instruction: a\n"
      "      other:\n"
      "        agent_class: LlmAgent\n"
      "        name: target_b\n"
      "        model: gemini-2.5-flash\n"
      "        instruction: b\n"
  )

  wf = config_agent_utils.from_config(str(wf_file))

  routes = {e.route: e.to_node.name for e in wf.graph.edges if e.route}
  assert routes == {"name": "target_a", "other": "target_b"}


def test_from_config_names_an_undefined_node_plainly(tmp_path: Path):
  """A bare word is a node name; saying "invalid fully qualified name" misleads."""
  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: wf\n"
      "edges:\n"
      "  - - START\n"
      "    - agent_class: LlmAgent\n"
      "      name: checker\n"
      "      model: gemini-2.5-flash\n"
      "      instruction: check\n"
      "  - - checker\n"
      "    - later_node\n"
  )

  with pytest.raises(ValueError, match="Unknown node 'later_node'"):
    config_agent_utils.from_config(str(wf_file))


def test_from_config_reads_an_inline_node_spelled_with_a_code_key(
    tmp_path: Path,
):
  """`model_code` names the `model` field even though it is not a field itself.

  Testing keys against `model_fields` alone sent this to the routing-map branch,
  where `name` then failed to resolve as a node.
  """
  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: wf\n"
      "edges:\n"
      "  - - START\n"
      "    - name: n1\n"
      "      model_code:\n"
      "        name: mylib.a_model\n"
      "      instruction: i\n"
  )

  with mock.patch.object(
      config_agent_utils,
      "resolve_code_reference",
      lambda cc: "gemini-2.5-flash",
  ):
    wf = config_agent_utils.from_config(str(wf_file))

  assert [(e.from_node.name, e.to_node.name) for e in wf.graph.edges] == [
      ("__START__", "n1")
  ]


def test_from_config_reads_a_single_route_map_keyed_code(tmp_path: Path):
  """`{code: <node>}` is a routing map; `{code: "a.b.c"}` is a reference.

  The two are told apart by the value, since only a reference is a string.
  """
  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: wf\n"
      "edges:\n"
      "  - - START\n"
      "    - agent_class: LlmAgent\n"
      "      name: cls\n"
      "      model: gemini-2.5-flash\n"
      "      instruction: i\n"
      "    - code:\n"
      "        agent_class: LlmAgent\n"
      "        name: tgt\n"
      "        model: gemini-2.5-flash\n"
      "        instruction: i\n"
  )

  wf = config_agent_utils.from_config(str(wf_file))

  assert {e.route: e.to_node.name for e in wf.graph.edges if e.route} == {
      "code": "tgt"
  }


def test_from_config_lets_an_inline_node_reach_a_from_config_override(
    tmp_path: Path,
):
  """The same class inlined or referenced by path has to build the same way.

  An inline node was constructed directly, so a subclass owning its own
  construction was silently skipped in that spelling only.
  """
  ran = []

  class OwnBuild(BaseAgent):

    @classmethod
    def from_config(cls, config, config_abs_path):
      ran.append(True)
      return cls(name="built_by_override")

  wf_file = tmp_path / "root_agent.yaml"
  wf_file.write_text(
      "agent_class: Workflow\n"
      "name: wf\n"
      "edges:\n"
      "  - - START\n"
      "    - agent_class: mylib.OwnBuild\n"
      "      name: ignored_by_the_override\n"
  )

  with mock.patch.object(
      config_agent_utils, "resolve_fully_qualified_name", lambda n: OwnBuild
  ):
    wf = config_agent_utils.from_config(str(wf_file))

  assert ran
  assert wf.graph.edges[0].to_node.name == "built_by_override"
