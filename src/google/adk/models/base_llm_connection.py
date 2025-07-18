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

from __future__ import annotations

from abc import abstractmethod
from typing import AsyncGenerator

from google.genai import types

from .llm_response import LlmResponse


class BaseLlmConnection:
  """The base class for a live model connection."""

  @abstractmethod
  async def send_history(self, history: list[types.Content]):
    """Sends the conversation history to the model.

    You call this method right after setting up the model connection.
    The model will respond if the last content is from user, otherwise it will
    wait for new user input before responding.

    Args:
      history: The conversation history to send to the model.
    """
    pass

  @abstractmethod
  async def send_content(self, content: types.Content):
    """Sends a user content to the model.

    The model will respond immediately upon receiving the content.
    If you send function responses, all parts in the content should be function
    responses.

    Args:
      content: The content to send to the model.
    """
    pass

  @abstractmethod
  async def send_realtime(self, blob: types.Blob):
    """Sends a chunk of audio or a frame of video to the model in realtime.

    The model may not respond immediately upon receiving the blob. It will do
    voice activity detection and decide when to respond.

    Args:
      blob: The blob to send to the model.
    """
    pass

  @abstractmethod
  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receives the model response using the llm server connection.

    Args: None.

    Yields:
      LlmResponse: The model response.
    """
    pass

  @abstractmethod
  async def close(self):
    """Closes the llm server connection."""
    pass
