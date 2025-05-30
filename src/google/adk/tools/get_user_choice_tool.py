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

from typing import Optional

from .long_running_tool import LongRunningFunctionTool
from .tool_context import ToolContext


def get_user_choice(
    options: list[str], tool_context: ToolContext
) -> Optional[str]:
  """Provides the options to the user and asks them to choose one."""
  tool_context.actions.skip_summarization = True
  return None


get_user_choice_tool = LongRunningFunctionTool(func=get_user_choice)
