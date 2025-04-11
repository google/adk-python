# Copyright 2025 Google LLC
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

import json

import pytest

from .fixture import context_variable_agent
from .utils import TestRunner


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": context_variable_agent.context_variable_echo_agent}],
    indirect=True,
)
def test_context_variable_missing(agent_runner: TestRunner):
    with pytest.raises(KeyError) as e_info:
        agent_runner.run("Hi echo my customer id.")
    assert "customerId" in str(e_info.value)


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": context_variable_agent.context_variable_update_agent}],
    indirect=True,
)
def test_context_variable_update(agent_runner: TestRunner):
    _call_function_and_assert(
        agent_runner,
        "update_fc",
        ["RRRR", "3.141529", ["apple", "banana"], [1, 3.14, "hello"]],
        "successfully",
    )


@pytest.mark.parametrize(
    "agent_runner",
    [{"agent": context_variable_agent.context_variable_with_complicated_format_agent}],
    indirect=True,
)
def test_context_variable_with_complicated_format(agent_runner: TestRunner):
    agent_runner.run("Hi echo my customer id.")
    model_response_event = agent_runner.get_events()[-1]
    assert model_response_event.author == "context_variable_with_complicated_format_agent"
    assert model_response_event.content.role == "model"
    assert "1234567890" in model_response_event.content.parts[0].text.strip()
    assert "30" in model_response_event.content.parts[0].text.strip()
    assert "{  non-identifier-float}}" in model_response_event.content.parts[0].text.strip()
    assert "{'key1': 'value1'}" in model_response_event.content.parts[0].text.strip()
    assert "{{'key2': 'value2'}}" in model_response_event.content.parts[0].text.strip()


def _call_function_and_assert(
    agent_runner: TestRunner, function_name: str, params, expected
):
    param_section = (
        " with params"
        f" {params if isinstance(params, str) else json.dumps(params)}"
        if params is not None
        else ""
    )
    agent_runner.run(
        f"Call {function_name}{param_section} and show me the result"
    )

    model_response_event = agent_runner.get_events()[-1]
    assert model_response_event.author == "context_variable_update_agent"
    assert model_response_event.content.role == "model"
    assert expected in model_response_event.content.parts[0].text.strip()
