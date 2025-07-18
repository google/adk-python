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

"""Common configuration classes for agent YAML configs."""
from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from ..utils.feature_decorator import working_in_progress


@working_in_progress("ArgumentConfig is not ready for use.")
class ArgumentConfig(BaseModel):
  """An argument passed to a function or a class's constructor."""

  model_config = ConfigDict(extra="forbid")

  name: Optional[str] = None
  """Optional. The argument name.

  When the argument is for a positional argument, this can be omitted.
  """

  value: Any
  """The argument value."""


@working_in_progress("CodeConfig is not ready for use.")
class CodeConfig(BaseModel):
  """Code reference config for a variable, a function, or a class.

  This config is used for configuring callbacks and tools.
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  """Required. The name of the variable, function, class, etc. in code.

  Examples:

    When used for tools,
      - It can be ADK built-in tools, such as `google_search` and `AgentTool`.
      - It can also be users' custom tools, e.g. my_library.my_tools.my_tool.

    When used for callbacks, it refers to a function, e.g. `my_library.my_callbacks.my_callback`
  """

  args: Optional[List[ArgumentConfig]] = None
  """Optional. The arguments for the code when `name` refers to a function or a
  class's contructor.

  Examples:
    ```
    tools
      - name: AgentTool
        args:
          - name: agent
            value: search_agent.yaml
          - name: skip_summarization
            value: True
    ```
  """
