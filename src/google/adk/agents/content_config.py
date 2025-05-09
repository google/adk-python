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

"""Content configuration for LLM Agents.

This module provides configuration classes for controlling how conversation history
and context are prepared for LLM agents. These configurations enable fine-grained
control over content management, including summarization of older history, selective
inclusion of messages, and injection of session state.
"""
from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

class SummarizationConfig(BaseModel):
    """Configuration for summarization of conversation history.
    
    This class controls how older messages are summarized to save context window
    space while preserving important information. When enabled, summarization 
    condenses prior conversation turns into a concise summary, reducing token 
    usage while maintaining conversational coherence.
    
    Example:
        ```python
        summarization_config = SummarizationConfig(
            model="gemini-1.5-flash",
            instruction="Summarize the key points and decisions in the conversation.",
            max_tokens=200
        )
        ```
    """
    model: Optional[str] = Field(
        default=None, 
        description="Model to use for summarization if different from the main LLM. Example: 'gemini-1.5-flash'."
    )
    instruction: Optional[str] = Field(
        default=None, 
        description="Custom instruction prompt for the summarization model. Example: 'Summarize the conversation focusing on key decisions and questions.'"
    )
    max_tokens: Optional[int] = Field(
        default=None, 
        description="Maximum number of tokens for the generated summary. Lower values create more concise summaries. Example: 300."
    )

class ContentConfig(BaseModel):
    """Configuration for conversation history and context management in LLM agents.
    
    This class provides comprehensive control over what content is included in the
    context window sent to the LLM. It allows filtering by author, limiting message 
    count, summarizing older messages, and injecting custom context from session state.
    
    The ContentConfig helps optimize context window usage, prevent token limits from
    being exceeded, and ensure the LLM has the most relevant information for generating
    responses.
    
    The final context presented to the LLM is structured in this order:
    1. Session state context (if configured)
    2. Summary of older history (if summarize=True)
    3. Older messages within the specified window
    4. Most recent messages (always_include_last_n)
    
    Basic Example:
        ```python
        # Simple configuration that includes all history
        content_config = ContentConfig(enabled=True)
        ```
    
    Advanced Example:
        ```python
        # Advanced configuration with summarization and state injection
        content_config = ContentConfig(
            enabled=True,
            max_events=50,
            summarize=True,
            always_include_last_n=5,
            summarization_config=SummarizationConfig(
                model="gemini-1.5-flash",
                instruction="Summarize the key points of the conversation.",
                max_tokens=300
            ),
            context_from_state=["user_profile", "current_task"],
            state_template="Session Info:\\n{context}",
            exclude_authors=["system_notification"]
        )
        ```
    
    Notes:
        - When `summarize=True`, older messages beyond `always_include_last_n` 
          will be summarized according to the `summarization_config`
        - The `convert_foreign_events` parameter should typically remain True to
          ensure proper handling of events from other agents and tools
    """
    enabled: bool = Field(
        default=True, 
        description="Controls whether to include conversation history in LLM context. Set to False to exclude all history."
    )
    include_authors: Optional[List[str]] = Field(
        default=None, 
        description="Only include messages from these specific authors. Example: ['user', 'system']. If None, includes all non-excluded authors."
    )
    exclude_authors: Optional[List[str]] = Field(
        default=None, 
        description="Exclude messages from these specific authors. Example: ['internal_agent']. Takes precedence over include_authors."
    )
    max_events: Optional[int] = Field(
        default=None, 
        description="Maximum number of total events/messages to initially consider, starting from most recent. Example: 50. If None, no limit is applied."
    )
    summarize: bool = Field(
        default=False, 
        description="When True, older conversation history will be summarized to save context window space. Requires setting summarization-related parameters."
    )
    summary_template: Optional[str] = Field(
        default=None, 
        description="Template string for formatting the summary in context. Use {summary} placeholder. Example: 'Previous conversation summary: {summary}'"
    )
    summarization_config: Optional[SummarizationConfig] = Field(
        default=None, 
        description="Detailed configuration for how summarization is performed. Set model, instruction, and token limits for the summarization process."
    )
    summarization_window: Optional[int] = Field(
        default=None, 
        description="Number of messages (counting from the end, excluding always_include_last_n) to include in summarization. Example: 20. If None, all applicable messages are considered."
    )
    always_include_last_n: Optional[int] = Field(
        default=None, 
        description="The N most recent messages to always include in full, never summarizing them. Example: 5. Critical for maintaining immediate conversation context."
    )
    context_from_state: Optional[List[str]] = Field(
        default=None, 
        description="List of session state keys to inject as additional context. Example: ['user_profile', 'conversation_topic']. Values from these keys are added to the context."
    )
    state_template: Optional[str] = Field(
        default=None, 
        description="Template for formatting injected state context. Uses the {context} placeholder for the entire dictionary of state values. Example: 'Session Info:\\n{context}'. Falls back to key-value pairs if {context} format fails."
    )
    convert_foreign_events: bool = Field(
        default=True, 
        description="Controls conversion of events from other agents to a readable format. WARNING: Setting to False can cause serious issues with tool execution and sub-agent interactions, potentially creating infinite loops. Keep as True unless you understand the implications."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
