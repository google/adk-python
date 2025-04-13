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

"""Ollama LLM integration."""

from __future__ import annotations

import json
import logging
from functools import cached_property
from typing import AsyncGenerator, Optional, Any, Iterable, Literal

import httpx
from google.genai import types
from pydantic import Field
from typing_extensions import override

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

logger = logging.getLogger(__name__)

# Type definition for Ollama's message format
OllamaMessage = dict[Literal["role", "content"], str]

def _convert_content_to_ollama_message(content: types.Content | str) -> OllamaMessage:
    """Converts ADK Content object or string to Ollama's message format."""
    if isinstance(content, str):
        return {"role": "user", "content": content}

    # Map ADK roles to Ollama roles
    role = "assistant" if content.role == types.Role.MODEL else "user"
    # Assuming simple text parts for Ollama
    text_content = "".join(part.text for part in content.parts if part.text)
    return {"role": role, "content": text_content}

def _build_ollama_request_payload(
    llm_request: LlmRequest, model_name: str, stream: bool
) -> dict[str, Any]:
    """Builds the request payload for the Ollama API."""
    messages = [
        _convert_content_to_ollama_message(content)
        for content in llm_request.contents
    ]

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
    }

    # Add options like temperature if specified in config
    if llm_request.config:
        payload["options"] = {
            "temperature": llm_request.config.temperature or 0.7,
            # Add other potential Ollama options here if needed
        }
        # Add system instruction if present
        if llm_request.config.system_instruction:
            payload["system"] = llm_request.config.system_instruction

    return payload

class OllamaLlm(BaseLlm):
    """Ollama LLM implementation.

    Connects to a running Ollama instance (http://localhost:11434 by default).
    Requires the httpx library (`pip install httpx`).
    """

    api_base: str = Field(default="http://localhost:11434")
    """The base URL for the Ollama API."""
    request_timeout: float = Field(default=600.0)
    """Timeout for the API request in seconds."""

    @override
    @classmethod
    def supported_models(cls) -> list[str]:
        """Returns a list of supported models in regex for LlmRegistry."""
        # Ollama supports any model name that's pulled locally
        return [".*"]

    @cached_property
    def _client(self) -> httpx.AsyncClient:
        """Cached asynchronous HTTP client for Ollama API requests."""
        return httpx.AsyncClient(base_url=self.api_base, timeout=self.request_timeout)

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generates content using the Ollama API.

        Args:
            llm_request: The LlmRequest object containing the prompt, history,
                         and configuration.
            stream: Whether to stream the response.

        Yields:
            LlmResponse objects containing the generated content.

        Raises:
            RuntimeError: If the Ollama API returns an error.
            httpx.HTTPStatusError: If the API request fails.
        """
        endpoint = "/api/chat"
        # Ensure model name from LlmRequest is used if provided, otherwise use self.model
        model_name = llm_request.model or self.model
        payload = _build_ollama_request_payload(llm_request, model_name, stream)

        logger.info(f"Sending request to Ollama model '{model_name}' at {self.api_base}")

        try:
            if stream:
                async with self._client.stream('POST', endpoint, json=payload) as response:
                    response.raise_for_status()  # Raise exception for bad status codes (4xx or 5xx)
                    accumulated_content = ""
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode JSON line from Ollama stream: {line}")
                            continue

                        if "error" in data:
                            raise RuntimeError(f"Ollama API error: {data['error']}")

                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            accumulated_content += chunk
                            yield LlmResponse(
                                content=types.Content(
                                    parts=[types.Part(text=chunk)],
                                    role='model' # Use standard ADK 'model' role
                                ),
                                partial=True # Indicate this is a streaming chunk
                            )

                        # Check if the stream is done
                        if data.get("done", False):
                            # Yield final aggregated content if needed (optional, Gemini does similar)
                            # yield LlmResponse(
                            #    content=types.Content(
                            #        parts=[types.Part(text=accumulated_content)],
                            #        role='model'
                            #    )
                            #)
                            break # Exit loop once done

            else: # Non-streaming
                response = await self._client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    raise RuntimeError(f"Ollama API error: {data['error']}")

                full_content = data.get("message", {}).get("content", "")
                yield LlmResponse(
                    content=types.Content(
                        parts=[types.Part(text=full_content)],
                        role='model' # Use standard ADK 'model' role
                    )
                )

        except httpx.RequestError as e:
            logger.error(f"Error requesting Ollama API: {e}")
            # Re-raise as a more generic exception or handle as needed
            raise RuntimeError(f"Failed to connect to Ollama API at {self.api_base}: {e}") from e

        # Ensure client is closed if we created it here, though cached_property helps
        # Alternative: Use lifespan management if running in an ASGI app context
        # await self._client.aclose() # Only if not using cached_property or lifespan