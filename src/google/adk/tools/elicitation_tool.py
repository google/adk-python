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

"""Tool for triggering elicitation flow."""

from __future__ import annotations

import inspect
from typing import Any, Optional
from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext
from ._automatic_function_calling_util import build_function_declaration

class TriggerElicitationTool(BaseTool):
    """Tool used by the model to trigger the elicitation flow.
    
    The model should call this tool when it detects missing information or
    ambiguity that requires user clarification.
    """

    def __init__(self):
        def trigger_elicitation(
            question: str,
            options: Optional[list[str]] = None,
            missing_entities: Optional[list[str]] = None,
            context_snapshot: Optional[dict[str, Any]] = None,
        ) -> str:
            """Triggers the elicitation flow to ask the user for clarification.
            
            Use this tool when you need to ask a question to resolve ambiguity or
            request missing parameters before you can proceed.
            
            Args:
                question: The clarification question to ask the user.
                options: Optional list of suggested options for the user to choose from.
                missing_entities: Optional list of keys or parameters that are missing.
                context_snapshot: Optional state snapshot to be passed back in hidden_context.
            """
            return "Elicitation triggered."

        self.func = trigger_elicitation
        super().__init__(
            name=self.func.__name__,
            description=self.func.__doc__.strip() if self.func.__doc__ else '',
        )

    @override
    def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
        """Gets the OpenAPI specification of this tool."""
        function_decl = types.FunctionDeclaration.model_validate(
            build_function_declaration(
                func=self.func,
                ignore_params=[],
                variant=self._api_variant,
            )
        )
        return function_decl

    @override
    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> Any:
        """Process the tool call.
        
        This tool is a signal and does not perform any external action.
        It returns the arguments passed by the model.
        """
        return args
