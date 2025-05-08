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

"""Content configuration for LLM Agents."""
from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

class SummarizationConfig(BaseModel):
    """
    Configuration for summarization step in content processing.
    """
    model: Optional[str] = Field(
        default=None, description="Model to use for summarization (if different from main LLM)."
    )
    instruction: Optional[str] = Field(
        default=None, description="Instruction prompt for summarization."
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens for the summary."
    )

class ContentConfig(BaseModel):
    """
    Configuration for how an agent prepares conversation history for the LLM.
    """
    enabled: bool = Field(
        default=True, description="Whether to include conversation history at all."
    )
    include_authors: Optional[List[str]] = Field(
        default=None, description="Only include messages from these authors (if set)."
    )
    exclude_authors: Optional[List[str]] = Field(
        default=None, description="Exclude messages from these authors (if set)."
    )
    max_events: Optional[int] = Field(
        default=None, description="Maximum number of total events/messages to initially consider."
    )
    summarize: bool = Field(
        default=False, description="Whether to summarize older history."
    )
    summary_template: Optional[str] = Field(
        default=None, description="Template for formatting the summary (e.g., 'Summary: {summary}')."
    )
    summarization_config: Optional[SummarizationConfig] = Field(
        default=None, description="Config for how summarization is performed."
    )
    summarization_window: Optional[int] = Field(
        default=None, description="Summarize only events within this window size from the end (excluding always_include_last_n). If None, consider all non-excluded recent events."
    )
    always_include_last_n: Optional[int] = Field(
        default=None, description="Always include the last N messages in full, even if summarizing."
    )
    context_from_state: Optional[List[str]] = Field(
        default=None, description="List of session state keys to inject as context."
    )
    state_template: Optional[str] = Field(
        default=None, description="Template for formatting injected state context."
    )
    convert_foreign_events: bool = Field(
        default=True, description="Whether to convert events from other agents to a 'For context:' format."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
