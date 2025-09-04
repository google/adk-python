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


from typing import List
from typing import Optional

from google.genai import types
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..agents.readonly_context import ReadonlyContext
from ..models.llm_request import LlmRequest
from .base_planner import BasePlanner

# 使用与 PlanReActPlanner 兼容的标签
PLANNING_TAG = '/*PLANNING*/'
REASONING_TAG = '/*REASONING*/'
ACTION_TAG = '/*ACTION*/'
REFLECTION_TAG = '/*REFLECTION*/'
REPLANNING_TAG = '/*REPLANNING*/'
FINAL_ANSWER_TAG = '/*FINAL_ANSWER*/'


class PlanReflectionPlanner(BasePlanner):
    """Plan-Reflection planner that enforces reflection in the ReAct cycle.

    Note: this planner does not require the model to support built-in thinking
    features or setting the thinking config.
    """

    @override
    def build_planning_instruction(
            self,
            readonly_context: ReadonlyContext,
            llm_request: LlmRequest,
    ) -> str:
        return self._build_reflection_planner_instruction()

    @override
    def process_planning_response(
            self,
            callback_context: CallbackContext,
            response_parts: List[types.Part],
    ) -> Optional[List[types.Part]]:
        if not response_parts:
            return None

        preserved_parts = []
        first_fc_part_index = -1

        for i in range(len(response_parts)):
            if response_parts[i].function_call:
                if not response_parts[i].function_call.name:
                    continue
                preserved_parts.append(response_parts[i])
                first_fc_part_index = i
                break

            self._handle_non_function_call_parts(response_parts[i], preserved_parts)

        if first_fc_part_index > 0:
            j = first_fc_part_index + 1
            while j < len(response_parts):
                if response_parts[j].function_call:
                    preserved_parts.append(response_parts[j])
                    j += 1
                else:
                    break

        return preserved_parts

    def _split_by_last_pattern(self, text, separator):
        """Splits the text by the last occurrence of the separator."""
        index = text.rfind(separator)
        if index == -1:
            return text, ''
        return text[: index + len(separator)], text[index + len(separator):]

    def _handle_non_function_call_parts(
            self, response_part: types.Part, preserved_parts: list[types.Part]
    ):
        """Handles non-function-call parts of the response."""
        if response_part.text and FINAL_ANSWER_TAG in response_part.text:
            reasoning_text, final_answer_text = self._split_by_last_pattern(
                response_part.text, FINAL_ANSWER_TAG
            )
            if reasoning_text:
                reasoning_part = types.Part(text=reasoning_text)
                self._mark_as_thought(reasoning_part)
                preserved_parts.append(reasoning_part)
            if final_answer_text:
                preserved_parts.append(types.Part(text=final_answer_text))
        else:
            response_text = response_part.text or ''
            # 包含所有标签，包括 REFLECTION_TAG
            if response_text and (
                    any(
                        response_text.startswith(tag)
                        for tag in [
                            PLANNING_TAG,
                            REASONING_TAG,
                            ACTION_TAG,
                            REFLECTION_TAG,
                            REPLANNING_TAG,
                        ]
                    )
            ):
                self._mark_as_thought(response_part)
            preserved_parts.append(response_part)

    def _mark_as_thought(self, response_part: types.Part):
        """Marks the response part as thought."""
        if response_part.text:
            response_part.thought = True
        return

    def _build_reflection_planner_instruction(self) -> str:
        """Builds the reflection planner instruction with stronger enforcement."""

        high_level_preamble = f"""
When answering the question, try to leverage the available tools to gather the information instead of your memorized knowledge.

Follow this enhanced process when answering the question:

(1) **Initial Planning Phase**: First come up with a comprehensive plan in natural language text format
(2) **Execution Phase**: Use tools to execute the plan with reasoning between tool code snippets to summarize current state and determine next steps. Tool code snippets and reasoning should be interleaved with each other
(3) **Reflection Phase**: After each major step or when encountering obstacles, reflect on the progress, effectiveness of the approach, and quality of results obtained
(4) **Replanning Phase**: Based on reflections, adjust the plan if necessary, considering new information, alternative approaches, or course corrections
(5) **Iterative Execution**: Continue with refined execution based on updated plans
(6) **Final Answer**: Return one comprehensive final answer incorporating all insights gained

Follow this structured format when answering the question:

- **(1) {PLANNING_TAG}**: The initial planning part should be under this tag
- **(2) {ACTION_TAG}**: Tool code snippets should be under this tag  
- **(3) {REASONING_TAG}**: Reasoning parts should be under this tag
- **(4) {REFLECTION_TAG}**: Critical analysis of progress, results quality, approach effectiveness, and identification of potential issues or improvements
- **(5) {REPLANNING_TAG}**: Updated plans, alternative strategies, or course corrections based on reflections and new information discovered
- **(6) {FINAL_ANSWER_TAG}**: The comprehensive final answer incorporating all findings and insights

**Process Flow Guidelines:**
- Interleave ACTION_TAG and REASONING_TAG as you execute steps
- Use REFLECTION_TAG after completing significant milestones or when encountering challenges
- Follow REFLECTION_TAG with REPLANNING_TAG when adjustments to the approach are needed
- Continue the ACTION → REASONING → REFLECTION → REPLANNING cycle as necessary
- Multiple reflection-replanning cycles are encouraged for complex problems
- End with FINAL_ANSWER_TAG containing the complete, well-reasoned response
"""

        planning_preamble = f"""  
{PLANNING_TAG} Requirements:  
Create a numbered plan that breaks down the user query into actionable steps. Each step should specify which tools to use.  
"""

        reasoning_preamble = """
Below are the requirements for the reasoning:
The reasoning makes a summary of the current trajectory based on the user query and tool outputs. Based on the tool outputs and plan, the reasoning also comes up with instructions to the next steps, making the trajectory closer to the final answer.
"""

        reflection_preamble = f"""  
{REFLECTION_TAG} Requirements - ABSOLUTELY MANDATORY:  
After completing your actions, you MUST include this section to:  
1. Evaluate if your actions achieved the intended goals  
2. Identify any gaps or issues in your approach  
3. Assess if you have enough information to answer the user's query  
4. Determine if replanning is necessary  

This section is REQUIRED - do not proceed to final answer without reflection.  
"""

        replanning_preamble = f"""  
{REPLANNING_TAG} Requirements (conditional):  
Only if reflection reveals issues, create a revised plan and execute it with new {ACTION_TAG} and {REASONING_TAG} sections.  
"""

        final_answer_preamble = f"""  
{FINAL_ANSWER_TAG} Requirements:  
Provide your final answer only after completing reflection. Base your answer on execution results and reflection insights.  
"""

        # Only contains the requirements for custom tool/libraries.
        tool_code_without_python_libraries_preamble = """
Below are the requirements for the tool code:

**Custom Tools:** The available tools are described in the context and can be directly used.
- Code must be valid self-contained Python snippets with no imports and no references to tools or Python libraries that are not in the context.
- You cannot use any parameters or fields that are not explicitly defined in the APIs in the context.
- The code snippets should be readable, efficient, and directly relevant to the user query and reasoning steps.
- When using the tools, you should use the library name together with the function name, e.g., vertex_search.search().
- If Python libraries are not provided in the context, NEVER write your own code other than the function calls using the provided tools.
"""

        return '\n\n'.join([
            high_level_preamble,
            planning_preamble,
            reasoning_preamble,
            reflection_preamble,
            replanning_preamble,
            final_answer_preamble,
            tool_code_without_python_libraries_preamble,
        ])
