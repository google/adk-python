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

from typing import FrozenSet
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import Field

from ..utils.feature_decorator import experimental
from .base_agent_config import BaseAgentConfig
from .common_configs import CodeConfig
from ..tools.tool_configs import ToolConfig


# Default set of safe imports for Python code execution
DEFAULT_SAFE_IMPORTS: FrozenSet[str] = frozenset(
    {
        # Standard library - safe modules
        "json",
        "math",
        "re",
        "datetime",
        "collections",
        "collections.*",
        "itertools",
        "functools",
        "operator",
        "string",
        "textwrap",
        "unicodedata",
        "decimal",
        "fractions",
        "random",
        "statistics",
        "typing",
        "typing.*",
        "dataclasses",
        "enum",
        "abc",
        "copy",
        "pprint",
        "reprlib",
        "numbers",
        "cmath",
        "time",
        "calendar",
        "hashlib",
        "hmac",
        "base64",
        "binascii",
        "html",
        "html.*",
        "urllib.parse",
        "uuid",
        "struct",
        "codecs",
        "locale",
        "gettext",
        "bisect",
        "heapq",
        "array",
        "weakref",
        "types",
        "contextlib",
        "warnings",
        "traceback",
        "linecache",
        "difflib",
        "graphlib",
        "zoneinfo",
        # Common data science (can be enabled explicitly)
        "numpy",
        "numpy.*",
        "pandas",
        "pandas.*",
        "scipy",
        "scipy.*",
        "matplotlib",
        "matplotlib.*",
    }
)


@experimental
class CodingAgentConfig(BaseAgentConfig):
    """Configuration for CodingAgent.

    This config extends BaseAgentConfig with fields specific to agents that
    generate and execute Python code to accomplish tasks using tools.
    """

    agent_class: Union[Literal["CodingAgent"], str] = Field(
        default="CodingAgent",
        description="The class of the agent. Must be CodingAgent.",
    )

    model: str = Field(
        default="",
        description=(
            "The model to use for the agent. When not set, the agent will "
            "inherit the model from its ancestor or use the default model."
        ),
    )

    model_code: Optional[CodeConfig] = Field(
        default=None,
        description=(
            "Optional. Code reference to a custom model instance. "
            "Takes precedence over the model field if both are set."
        ),
    )

    instruction: str = Field(
        default="",
        description=(
            "Dynamic instructions for the agent, guiding its behavior. "
            "Can contain placeholders like {variable_name} that will be "
            "resolved at runtime using session state and context."
        ),
    )

    tools: Optional[List[ToolConfig]] = Field(
        default=None,
        description=(
            "Optional. The list of tools available to the agent. "
            "Tools are exposed as Python functions that the agent can call "
            "in the generated code."
        ),
    )

    code_executor: Optional[CodeConfig] = Field(
        default=None,
        description=(
            "Optional. Code reference to a custom code executor instance. "
            "If not set, a default ContainerCodeExecutor will be used."
        ),
    )

    authorized_imports: FrozenSet[str] = Field(
        default=DEFAULT_SAFE_IMPORTS,
        description=(
            "Set of allowed import names/patterns. Supports wildcards "
            '(e.g., "collections.*" allows all collections submodules). '
            "Any imports not in this set will be rejected before execution."
        ),
    )

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Maximum number of ReAct loop iterations. Each iteration "
            "involves generating code, executing it, and processing results."
        ),
    )

    error_retry_attempts: int = Field(
        default=2,
        ge=0,
        le=10,
        description=(
            "Number of times to retry code execution on errors. "
            "Error messages are fed back to the LLM for correction."
        ),
    )

    stateful: bool = Field(
        default=False,
        description=(
            "Whether to maintain state across iterations. If True, "
            "execution history is preserved and re-executed to restore state."
        ),
    )

    tool_server_host: Optional[str] = Field(
        default=None,
        description=(
            "Host address for the tool execution server. If not set, "
            "auto-detection will try host.docker.internal first, "
            "then fall back to 172.17.0.1 for Linux."
        ),
    )

    tool_server_port: int = Field(
        default=8765,
        ge=1024,
        le=65535,
        description="Port for the tool execution server.",
    )

    before_model_callbacks: Optional[List[CodeConfig]] = Field(
        default=None,
        description="Optional. Callbacks to be called before calling the LLM.",
    )

    after_model_callbacks: Optional[List[CodeConfig]] = Field(
        default=None,
        description="Optional. Callbacks to be called after calling the LLM.",
    )

    before_tool_callbacks: Optional[List[CodeConfig]] = Field(
        default=None,
        description="Optional. Callbacks to be called before calling a tool.",
    )

    after_tool_callbacks: Optional[List[CodeConfig]] = Field(
        default=None,
        description="Optional. Callbacks to be called after calling a tool.",
    )
