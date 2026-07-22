# mypy: ignore-errors
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

"""Tests for domain_specific_publish.py."""

import inspect
from unittest import mock

from google.adk.integrations.eventarc import _config as config
from google.adk.integrations.eventarc import _domain_specific_publish as domain_specific_publish
from google.adk.integrations.eventarc import _eventarc_toolset as eventarc_toolset
import pydantic
import pytest


class DummyPayload(pydantic.BaseModel):
  user_id: str
  action: str


@pytest.fixture
def toolset():
  ts = mock.Mock(spec=eventarc_toolset.EventarcToolset)
  ts.tool_config = config.EventarcToolConfig(project_id="test-project")
  ts.credentials_config = None
  return ts


def test_mandatory_missing_raises_typeerror(toolset):
  with pytest.raises(
      TypeError,
      match="The 'bus' parameter is mandatory and must be provided.",
  ):
    domain_specific_publish.build_domain_specific_tool(
        toolset=toolset,
        name="test_tool",
        description="desc",
        bus=domain_specific_publish.MISSING,
        ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
            type="type",
            source="source",
        ),
    )


def test_mandatory_omit_raises_typeerror(toolset):
  with pytest.raises(
      TypeError,
      match="CloudEvent field 'type' is mandatory and cannot be OMIT.",
  ):
    domain_specific_publish.build_domain_specific_tool(
        toolset=toolset,
        name="test_tool",
        description="desc",
        bus="bus",
        ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
            type=domain_specific_publish.OMIT,
            source="source",
        ),
    )


def test_mandatory_none_raises_typeerror(toolset):
  with pytest.raises(
      TypeError,
      match="The 'bus' parameter is mandatory and cannot be None.",
  ):
    domain_specific_publish.build_domain_specific_tool(
        toolset=toolset,
        name="test_tool",
        description="desc",
        bus=None,
        ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
            type="type",
            source="source",
        ),
    )

  with pytest.raises(
      TypeError,
      match="CloudEvent field 'type' is mandatory and cannot be None.",
  ):
    domain_specific_publish.build_domain_specific_tool(
        toolset=toolset,
        name="test_tool",
        description="desc",
        bus="bus",
        ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
            type=None,
            source="source",
        ),
    )


def test_signature_generation(toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type=domain_specific_publish.AgentProvided("The type"),
          source=domain_specific_publish.AgentProvided(
              "The source", default="default-source"
          ),
          subject=domain_specific_publish.AgentProvided(
              "The subject", default=lambda x: "dyn-subject"
          ),
          time=domain_specific_publish.AgentProvided(
              "The time", default=domain_specific_publish.OMIT
          ),
      ),
      payload_schema=DummyPayload,
  )

  sig = inspect.signature(tool.func)

  # type has no default
  assert sig.parameters["type"].default == inspect.Parameter.empty

  # source has static default
  assert sig.parameters["source"].default == "default-source"

  # subject has dynamic default -> exposes None
  assert sig.parameters["subject"].default is None

  # time has domain_specific_publish.OMIT default -> exposes None
  assert sig.parameters["time"].default is None

  # payload schema
  assert sig.parameters["event_data"].annotation == DummyPayload


@pytest.mark.asyncio
@mock.patch.object(domain_specific_publish, "publish_message", autospec=True)
async def test_runtime_execution_with_payload(mock_publish, toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type=lambda p: f"action.{p.action}",
          source="my-source",
          subject=domain_specific_publish.AgentProvided(
              "Subject", default=lambda p: p.user_id
          ),
          time=domain_specific_publish.OMIT,
      ),
      payload_schema=DummyPayload,
  )

  payload = DummyPayload(user_id="user123", action="login")

  await tool.func(
      event_data=payload,
      credentials=None,
      settings=config.EventarcToolConfig(),
      tool_context=mock.Mock(),
  )

  mock_publish.assert_called_once()
  kwargs = mock_publish.call_args.kwargs

  assert kwargs["bus"] == "my-bus"
  assert kwargs["type"] == "action.login"
  assert kwargs["source"] == "my-source"
  assert kwargs["subject"] == "user123"
  assert "time" not in kwargs
  assert kwargs["data"] == {"user_id": "user123", "action": "login"}


@pytest.mark.asyncio
@mock.patch.object(domain_specific_publish, "publish_message", autospec=True)
async def test_runtime_execution_explicit_null_fallback(mock_publish, toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type="my-type",
          source="my-source",
          subject=domain_specific_publish.AgentProvided(
              "Subject", default="fallback-subject"
          ),
      ),
  )

  # Agent passed explicitly None for 'subject'
  await tool.func(
      subject=None,
      credentials=None,
      settings=config.EventarcToolConfig(),
      tool_context=mock.Mock(),
  )

  kwargs = mock_publish.call_args.kwargs
  assert kwargs["subject"] == "fallback-subject"


@pytest.mark.asyncio
async def test_runtime_mandatory_omit_raises(toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type=lambda p: domain_specific_publish.OMIT,
          source="my-source",
      ),
  )

  with pytest.raises(
      ValueError,
      match="Mandatory CloudEvent attribute 'type' cannot evaluate to OMIT.",
  ):
    await tool.func(
        credentials=None,
        settings=config.EventarcToolConfig(),
        tool_context=mock.Mock(),
    )


@pytest.mark.asyncio
@mock.patch.object(domain_specific_publish, "publish_message", autospec=True)
async def test_runtime_agent_provided_missing_raises(_, toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type=domain_specific_publish.AgentProvided("The type"),
          source="my-source",
      ),
  )

  with pytest.raises(
      ValueError, match="Agent did not provide mandatory attribute 'type'"
  ):
    # We don't pass 'type' in kwargs
    await tool.func(
        credentials=None,
        settings=config.EventarcToolConfig(),
        tool_context=mock.Mock(),
    )


@pytest.mark.asyncio
@mock.patch.object(domain_specific_publish, "publish_message", autospec=True)
async def test_runtime_agent_provided_bus_missing_raises(_, toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus=domain_specific_publish.AgentProvided("The bus"),
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type="my-type",
          source="my-source",
      ),
  )

  with pytest.raises(
      ValueError, match="Agent did not provide mandatory attribute 'bus'"
  ):
    # We don't pass 'bus' in kwargs
    await tool.func(
        credentials=None,
        settings=config.EventarcToolConfig(),
        tool_context=mock.Mock(),
    )


@pytest.mark.asyncio
@mock.patch.object(domain_specific_publish, "publish_message", autospec=True)
async def test_optional_fields_as_none_are_ignored(mock_publish, toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type="my-type",
          source="my-source",
          time=None,
          subject=None,
          id=None,
      ),
  )

  await tool.func(
      credentials=None,
      settings=config.EventarcToolConfig(),
      tool_context=mock.Mock(),
  )

  mock_publish.assert_called_once()
  kwargs = mock_publish.call_args.kwargs
  assert "time" not in kwargs
  assert "subject" not in kwargs
  assert "id" not in kwargs


def test_no_payload_schema_omits_event_data(toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type="my-type",
          source="my-source",
      ),
      payload_schema=None,
  )

  sig = inspect.signature(tool.func)
  assert "event_data" not in sig.parameters


def test_invalid_cloudevent_attributes(toolset):
  invalid_keys = [
      "self_",
      "my-key",
      "MyKey",
      "event_data",
  ]
  for key in invalid_keys:
    with pytest.raises(
        ValueError,
        match=f"Custom attribute '{key}' is invalid",
    ):
      domain_specific_publish.build_domain_specific_tool(
          toolset=toolset,
          name="test_tool",
          description="desc",
          bus="my-bus",
          ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
              type="my-type",
              source="my-source",
              custom_attributes={
                  key: domain_specific_publish.AgentProvided("desc")
              },
          ),
      )


@pytest.mark.asyncio
@mock.patch.object(domain_specific_publish, "publish_message", autospec=True)
async def test_runtime_execution_with_python_keywords(mock_publish, toolset):
  tool = domain_specific_publish.build_domain_specific_tool(
      toolset=toolset,
      name="test_tool",
      description="desc",
      bus="my-bus",
      ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
          type="my-type",
          source="my-source",
          custom_attributes={
              "self": domain_specific_publish.AgentProvided("desc"),
              "cls": domain_specific_publish.AgentProvided("desc"),
              "123foo": domain_specific_publish.AgentProvided("desc"),
          },
      ),
  )

  sig = inspect.signature(tool.func)
  assert "self_" in sig.parameters
  assert "cls_" in sig.parameters
  assert "_123foo" in sig.parameters

  await tool.func(
      self_="self_value",
      cls_="cls_value",
      _123foo="foo_value",
      credentials=None,
      settings=config.EventarcToolConfig(),
      tool_context=mock.Mock(),
  )

  mock_publish.assert_called_once()
  kwargs = mock_publish.call_args.kwargs
  assert "custom_attributes" in kwargs
  assert kwargs["custom_attributes"]["self"] == "self_value"
  assert kwargs["custom_attributes"]["cls"] == "cls_value"
  assert kwargs["custom_attributes"]["123foo"] == "foo_value"


def test_custom_attribute_missing_raises_typeerror(toolset):
  with pytest.raises(
      TypeError,
      match="Custom attribute 'mykey' cannot be MISSING.",
  ):
    domain_specific_publish.build_domain_specific_tool(
        toolset=toolset,
        name="test_tool",
        description="desc",
        bus="bus",
        ce_attributes_binding=domain_specific_publish.CloudEventAttributesBinding(
            type="type",
            source="source",
            custom_attributes={"mykey": domain_specific_publish.MISSING},
        ),
    )
