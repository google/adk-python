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

import json
from typing import AsyncGenerator, Optional
import httpx

from google.genai import types
from pydantic import Field

from .base_llm import BaseLlm
from .llm_request import LlmRequest
from .llm_response import LlmResponse

class OllamaLlm(BaseLlm):
    """Ollama LLM implementation."""

    api_base: str = Field(default="http://localhost:11434")
    """The base URL for the Ollama API."""

    @classmethod
    def supported_models(cls) -> list[str]:
        """Returns a list of supported models in regex for LlmRegistry."""
        return [".*"]  # Ollama supports any model that's pulled locally

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generates content using Ollama API.

        Args:
            llm_request: LlmRequest, the request to send to Ollama.
            stream: bool = False, whether to stream the response.

        Yields:
            LlmResponse objects containing the generated content.
        """
        # Convert ADK request format to Ollama format
        messages = []
        for content in llm_request.contents:
            if isinstance(content, str):
                messages.append({"role": "user", "content": content})
            else:
                role = "assistant" if content.role == types.Role.MODEL else "user"
                messages.append({"role": role, "content": content.parts[0].text})

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": llm_request.config.temperature if llm_request.config else 0.7,
            }
        }

        # Add system instruction if present
        if llm_request.config and llm_request.config.system_instruction:
            payload["system"] = llm_request.config.system_instruction

        async with httpx.AsyncClient() as client:
            endpoint = f"{self.api_base}/api/chat"
            
            if stream:
                async with client.stream('POST', endpoint, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if "error" in data:
                            raise RuntimeError(f"Ollama API error: {data['error']}")
                        
                        content = types.Content(
                            parts=[types.Part(text=data["message"]["content"])],
                            role=types.Role.MODEL
                        )
                        yield LlmResponse(candidates=[content])
                        
                        if data.get("done", False):
                            break
            else:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    raise RuntimeError(f"Ollama API error: {data['error']}")
                
                content = types.Content(
                    parts=[types.Part(text=data["message"]["content"])],
                    role=types.Role.MODEL
                )
                yield LlmResponse(candidates=[content])