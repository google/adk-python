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

from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.sessions.session import Session
from google.adk.utils import instructions_utils
import pytest

from .. import testing_utils


class MockArtifactService:

  def __init__(self, artifacts: dict):
    self.artifacts = artifacts

  async def load_artifact(self, app_name, user_id, session_id, filename):
    if filename in self.artifacts:
      return self.artifacts[filename]
    else:
      return None


async def _create_test_readonly_context(
    state: dict = None,
    artifact_service: MockArtifactService = None,
    app_name: str = "test_app",
    user_id: str = "test_user",
    session_id: str = "test_session_id",
) -> ReadonlyContext:
  agent = Agent(
      model="gemini-2.5-flash",
      name="agent",
      instruction="test",
  )
  invocation_context = await testing_utils.create_invocation_context(
      agent=agent
  )
  invocation_context.session = Session(
      state=state if state else {},
      app_name=app_name,
      user_id=user_id,
      id=session_id,
  )

  invocation_context.artifact_service = artifact_service
  return ReadonlyContext(invocation_context)


@pytest.mark.asyncio
async def test_inject_session_state():
  instruction_template = "Hello {user_name}, you are in {app_state} state."
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo", "app_state": "active"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Hello Foo, you are in active state."


@pytest.mark.asyncio
async def test_inject_session_state_with_artifact():
  instruction_template = "The artifact content is: {artifact.my_file}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert (
      populated_instruction
      == "The artifact content is: This is my artifact content."
  )


@pytest.mark.asyncio
async def test_inject_session_state_with_optional_state():
  instruction_template = "Optional value: {optional_value?}"
  invocation_context = await _create_test_readonly_context()

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Optional value: "


@pytest.mark.asyncio
async def test_inject_session_state_with_missing_state_raises_key_error():
  instruction_template = "Hello {missing_key}!"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )

  with pytest.raises(
      KeyError, match="Context variable not found: `missing_key`."
  ):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context
    )


@pytest.mark.asyncio
async def test_inject_session_state_with_missing_artifact_raises_key_error():
  instruction_template = "The artifact content is: {artifact.missing_file}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )

  with pytest.raises(KeyError, match="Artifact missing_file not found."):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context
    )


@pytest.mark.asyncio
async def test_inject_session_state_with_invalid_state_name_returns_original():
  instruction_template = "Hello {invalid-key}!"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Hello {invalid-key}!"


@pytest.mark.asyncio
async def test_inject_session_state_with_invalid_prefix_state_name_returns_original():
  instruction_template = "Hello {invalid:key}!"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Hello {invalid:key}!"


@pytest.mark.asyncio
async def test_inject_session_state_with_valid_prefix_state():
  instruction_template = "Hello {app:user_name}!"
  invocation_context = await _create_test_readonly_context(
      state={"app:user_name": "Foo"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Hello Foo!"


@pytest.mark.asyncio
async def test_inject_session_state_with_multiple_variables_and_artifacts():
  instruction_template = """
    Hello {user_name},
    You are {user_age} years old.
    Your favorite color is {favorite_color?}.
    The artifact says: {artifact.my_file}
    And another optional artifact: {artifact.other_file}
    """
  mock_artifact_service = MockArtifactService({
      "my_file": "This is my artifact content.",
      "other_file": "This is another artifact content.",
  })
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo", "user_age": 30, "favorite_color": "blue"},
      artifact_service=mock_artifact_service,
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  expected_instruction = """
    Hello Foo,
    You are 30 years old.
    Your favorite color is blue.
    The artifact says: This is my artifact content.
    And another optional artifact: This is another artifact content.
    """
  assert populated_instruction == expected_instruction


@pytest.mark.asyncio
async def test_inject_session_state_with_empty_artifact_name_raises_key_error():
  instruction_template = "The artifact content is: {artifact.}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )

  with pytest.raises(KeyError, match="Artifact  not found."):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context
    )


@pytest.mark.asyncio
async def test_inject_session_state_artifact_service_not_initialized_raises_value_error():
  instruction_template = "The artifact content is: {artifact.my_file}"
  invocation_context = await _create_test_readonly_context()
  with pytest.raises(ValueError, match="Artifact service is not initialized."):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context
    )


@pytest.mark.asyncio
async def test_inject_session_state_with_optional_missing_artifact_returns_empty():
  instruction_template = "Optional artifact: {artifact.missing_file?}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Optional artifact: "


@pytest.mark.asyncio
async def test_inject_session_state_with_none_state_value_returns_empty():
  instruction_template = "Value: {test_key}"
  invocation_context = await _create_test_readonly_context(
      state={"test_key": None}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Value: "


@pytest.mark.asyncio
async def test_inject_session_state_with_optional_missing_state_returns_empty():
  instruction_template = "Optional value: {missing_key?}"
  invocation_context = await _create_test_readonly_context()

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Optional value: "


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_basic_substitution():
  instruction_template = "Hello {{ state['user_name'] }}, welcome back."
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert populated_instruction == "Hello Foo, welcome back."


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_conditional():
  instruction_template = (
      "{% if state['is_premium'] %}Premium user.{% else %}Free user.{% endif %}"
  )
  premium_context = await _create_test_readonly_context(
      state={"is_premium": True}
  )
  free_context = await _create_test_readonly_context(
      state={"is_premium": False}
  )

  assert (
      await instructions_utils.inject_session_state(
          instruction_template, premium_context, use_jinja2=True
      )
      == "Premium user."
  )
  assert (
      await instructions_utils.inject_session_state(
          instruction_template, free_context, use_jinja2=True
      )
      == "Free user."
  )


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_loop():
  instruction_template = (
      "Items:{% for item in state['items'] %} {{ item }}{% endfor %}"
  )
  invocation_context = await _create_test_readonly_context(
      state={"items": ["a", "b", "c"]}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert populated_instruction == "Items: a b c"


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_filter():
  instruction_template = "Name: {{ state['name'] | upper }}"
  invocation_context = await _create_test_readonly_context(
      state={"name": "foo"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert populated_instruction == "Name: FOO"


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_artifact():
  # The async Jinja2 environment auto-awaits the artifact coroutine, so the
  # template uses a plain call without an `await` keyword.
  instruction_template = "Artifact: {{ artifact('my_file') }}"
  mock_artifact_service = MockArtifactService(
      {"my_file": "This is my artifact content."}
  )
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert populated_instruction == "Artifact: This is my artifact content."


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_missing_artifact_returns_empty():
  instruction_template = "Artifact: [{{ artifact('missing') }}]"
  mock_artifact_service = MockArtifactService({})
  invocation_context = await _create_test_readonly_context(
      artifact_service=mock_artifact_service
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert populated_instruction == "Artifact: []"


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_artifact_service_not_initialized():
  instruction_template = "Artifact: {{ artifact('my_file') }}"
  invocation_context = await _create_test_readonly_context()

  with pytest.raises(ValueError, match="Artifact service is not initialized."):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context, use_jinja2=True
    )


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_sandbox_blocks_unsafe_access():
  # Instruction templates may carry user/session data, so rendering runs in a
  # SandboxedEnvironment. The classic sandbox-escape vector (reaching an
  # object's class MRO) must be rejected.
  from jinja2.exceptions import SecurityError

  instruction_template = "{{ ''.__class__.__mro__ }}"
  invocation_context = await _create_test_readonly_context()

  with pytest.raises(SecurityError):
    await instructions_utils.inject_session_state(
        instruction_template, invocation_context, use_jinja2=True
    )


@pytest.mark.asyncio
async def test_inject_session_state_jinja2_does_not_touch_brace_var_syntax():
  # The legacy {var} syntax is not interpolated by the Jinja2 path; only
  # Jinja2 delimiters are. This documents the opt-in boundary between engines.
  instruction_template = "Legacy {user_name} and jinja {{ state['user_name'] }}"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context, use_jinja2=True
  )
  assert populated_instruction == "Legacy {user_name} and jinja Foo"


@pytest.mark.asyncio
async def test_inject_session_state_default_flag_uses_regex_path():
  # Explicitly assert the default (use_jinja2=False) keeps the regex behavior:
  # Jinja2 delimiters are left untouched and {var} is substituted.
  instruction_template = "Regex {user_name}, jinja {{ state['user_name'] }}"
  invocation_context = await _create_test_readonly_context(
      state={"user_name": "Foo"}
  )

  populated_instruction = await instructions_utils.inject_session_state(
      instruction_template, invocation_context
  )
  assert populated_instruction == "Regex Foo, jinja {{ state['user_name'] }}"
