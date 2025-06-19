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

from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from ...utils.feature_decorator import working_in_progress


@working_in_progress("agent config is under development")
class SubAgentConfig(BaseModel):
  """The config for a sub-agent."""

  model_config = ConfigDict(extra="forbid")

  config: Optional[str] = None
  """The config file path of the sub-agent defined via agent config.

  This is used to reference a sub-agent defined in another agent config file.

  Example:

    ```
    sub_agents:
      - config: search_agent.yaml
      - config: my_library/my_toy_agent.yaml
    ```
  """

  code_agent: Optional[CodeConfig] = None
  """The agent defined by code.

  When this exists, it override the config.

  Example:

    Below is a sample usage to reference the `toy_agent` defined in below Python
    and Java code:

    ```
    # my_library/my_toy_agent.py
    from google.adk.agents import LlmAgent

    my_toy_agent = LlmAgent(
        name="my_toy_agent",
        instruction="You are a helpful toy agent.",
        model="gemini-2.0-flash",
    )
    ```

    ```
    package com.acme.agents

    class ToyAgent {
      public static final LlmAgent toyAgent = LlmAgent.builder()
        .name("toy_agent")
        .model("gemini-2.0-flash")
        .instruction("You are a helpful assistant.")
        .build();
    }
    ```

    The yaml config should be:

    ```
    sub_agents:
      - code_agent:
          py_name: my_library.my_toy_agent
          java_name: com.acme.agents.ToyAgent.toyAgent
    ```
  """


@working_in_progress("agent config is under development")
class InlineCodeConfig(BaseModel):
  """Config for inline code.

  NOTE: this only applies to interpreted languages like Python.
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  """The name of the symbol in the inline code."""

  code: str
  """The inlined code."""


@working_in_progress("agent config is under development")
class CodeConfig(BaseModel):
  """Reference the code"""

  model_config = ConfigDict(extra="forbid")

  builtin_name: Optional[str] = None
  """The ADK builtin name across all languages.

  Examples:

    When used for agent_class, it could LlmAgent, SequetialAgent, etc.

    When used for tools, it could mean ADK builtin tools, such as google_search,
    load_memory, etc.
  """

  py_inline: Optional[InlineCodeConfig] = None
  """The inline code in Python.

  Examples:

    Below is a sample usage to use inline code to define a callback.

    ```
    before_agent_callbacks:
      - py_inline:
          name: my_callback
          code: |
            def my_callback(callback_context):
              ...
    ```
  """

  py_name: Optional[str] = None
  """The fully qualified name of the Python symbol.

  The symbal can be variable, function, class, or module-level object.

  Example:

    ```
    before_agent_callbacks:
      - py_name: my_library.my_callbacks.before_agent_callback
    ```
  """

  java_name: Optional[str] = None
  """The fully qualified name of the Java symbol.

  The symbal can be variable, function, class, or module-level object.

  Example:

    Below is a sample usage:

    ```
    before_agent_callbacks:
      - java_name: com.acme.security.Callbacks.beforeAgentCallback
    ```
  """
