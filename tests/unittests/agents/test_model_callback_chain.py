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

from enum import Enum
from functools import partial
from typing import Any
from typing import List
from typing import Optional
from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel
import pytest

from .. import testing_utils


class CallbackType(Enum):
  SYNC = 1
  ASYNC = 2


async def mock_async_before_cb_side_effect(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    ret_value=None,
):
  if ret_value:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=ret_value)]
        )
    )
  return None


def mock_sync_before_cb_side_effect(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    ret_value=None,
):
  if ret_value:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=ret_value)]
        )
    )
  return None


async def mock_async_after_cb_side_effect(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
    ret_value=None,
):
  if ret_value:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=ret_value)]
        )
    )
  return None


def mock_sync_after_cb_side_effect(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
    ret_value=None,
):
  if ret_value:
    return LlmResponse(
        content=testing_utils.ModelContent(
            [types.Part.from_text(text=ret_value)]
        )
    )
  return None


CALLBACK_PARAMS = [
    pytest.param(
        [
            (None, CallbackType.SYNC),
            ("callback_2_response", CallbackType.ASYNC),
            ("callback_3_response", CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        "callback_2_response",
        [1, 1, 0, 0],
        id="middle_async_callback_returns",
    ),
    pytest.param(
        [
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
            (None, CallbackType.SYNC),
            (None, CallbackType.ASYNC),
        ],
        "model_response",
        [1, 1, 1, 1],
        id="all_callbacks_return_none",
    ),
    pytest.param(
        [
            ("callback_1_response", CallbackType.SYNC),
            ("callback_2_response", CallbackType.ASYNC),
        ],
        "callback_1_response",
        [1, 0],
        id="first_sync_callback_returns",
    ),
]


@pytest.mark.parametrize(
    "callbacks, expected_response, expected_calls",
    CALLBACK_PARAMS,
)
@pytest.mark.asyncio
async def test_before_model_callbacks_chain(
    callbacks: List[tuple[str, int]],
    expected_response: str,
    expected_calls: List[int],
):
  responses = ["model_response"]
  mock_model = testing_utils.MockModel.create(responses=responses)

  mock_cbs = []
  for response, callback_type in callbacks:

    if callback_type == CallbackType.ASYNC:
      mock_cb = mock.AsyncMock(
          side_effect=partial(
              mock_async_before_cb_side_effect, ret_value=response
          )
      )
    else:
      mock_cb = mock.Mock(
          side_effect=partial(
              mock_sync_before_cb_side_effect, ret_value=response
          )
      )
    mock_cbs.append(mock_cb)
  # Create agent with multiple callbacks
  agent = Agent(
      name="root_agent",
      model=mock_model,
      before_model_callback=[mock_cb for mock_cb in mock_cbs],
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  result = await runner.run_async_with_new_session("test")
  assert testing_utils.simplify_events(result) == [
      ("root_agent", expected_response),
  ]

  # Assert that the callbacks were called the expected number of times
  for i, mock_cb in enumerate(mock_cbs):
    expected_calls_count = expected_calls[i]
    if expected_calls_count == 1:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited_once()
      else:
        mock_cb.assert_called_once()
    elif expected_calls_count == 0:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_not_awaited()
      else:
        mock_cb.assert_not_called()
    else:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited(expected_calls_count)
      else:
        mock_cb.assert_called(expected_calls_count)


@pytest.mark.parametrize(
    "callbacks, expected_response, expected_calls",
    CALLBACK_PARAMS,
)
@pytest.mark.asyncio
async def test_after_model_callbacks_chain(
    callbacks: List[tuple[str, int]],
    expected_response: str,
    expected_calls: List[int],
):
  responses = ["model_response"]
  mock_model = testing_utils.MockModel.create(responses=responses)

  mock_cbs = []
  for response, callback_type in callbacks:

    if callback_type == CallbackType.ASYNC:
      mock_cb = mock.AsyncMock(
          side_effect=partial(
              mock_async_after_cb_side_effect, ret_value=response
          )
      )
    else:
      mock_cb = mock.Mock(
          side_effect=partial(
              mock_sync_after_cb_side_effect, ret_value=response
          )
      )
    mock_cbs.append(mock_cb)
  # Create agent with multiple callbacks
  agent = Agent(
      name="root_agent",
      model=mock_model,
      after_model_callback=[mock_cb for mock_cb in mock_cbs],
  )

  runner = testing_utils.TestInMemoryRunner(agent)
  result = await runner.run_async_with_new_session("test")
  assert testing_utils.simplify_events(result) == [
      ("root_agent", expected_response),
  ]

  # Assert that the callbacks were called the expected number of times
  for i, mock_cb in enumerate(mock_cbs):
    expected_calls_count = expected_calls[i]
    if expected_calls_count == 1:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited_once()
      else:
        mock_cb.assert_called_once()
    elif expected_calls_count == 0:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_not_awaited()
      else:
        mock_cb.assert_not_called()
    else:
      if isinstance(mock_cb, mock.AsyncMock):
        mock_cb.assert_awaited(expected_calls_count)
      else:
        mock_cb.assert_called(expected_calls_count)
