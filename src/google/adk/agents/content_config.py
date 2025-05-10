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
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationError, model_validator

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
    
    Options:
        enabled: Include conversation history (default: True)
        include_authors: Only include messages from these authors
        exclude_authors: Exclude messages from these authors
        max_events: Max number of events/messages to consider
        summarize: Summarize older history to save context window space
        summary_template: Template for formatting the summary
        summarization_config: How summarization is performed
        summarization_window: Number of recent messages to summarize
        always_include_last_n: N most recent messages to always include in full
        context_from_state: Session state keys to inject as context
        state_template: Template for formatting injected state context
        convert_foreign_events: Convert events from other agents to readable format
    
    Notes:
        - When `summarize=True`, older messages beyond `always_include_last_n` will be summarized according to `summarization_config`.
        - The `convert_foreign_events` parameter should typically remain True to ensure proper handling of events from other agents and tools.
        - The `include_authors` and `exclude_authors` lists cannot contain the same author - this will raise a validation error.
    """
    enabled: bool = Field(
        default=True, 
        description=(
            "Controls whether to include conversation history in LLM context. Set to False to exclude all history. "
            "If False, only session state context (if configured) will be included."
        )
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
        description=(
            "Maximum number of total events/messages to initially consider, starting from most recent. "
            "Example: 50. If None, no limit is applied. "
            "WARNING: If your agent uses tools or function calls, setting a low max_events value may exclude "
            "required tool call/response pairs from the context, leading to broken tool execution or incomplete agent behavior. "
            "Use with caution for tool-using agents."
        )
    )
    summarize: bool = Field(
        default=False, 
        description=(
            "When True, older conversation history will be summarized to save context window space. "
            "Requires setting summarization-related parameters. "
            "NOTE: Enabling summarization does NOT reduce LLM usage costs, as it requires an additional LLM call to generate the summary. "
            "The main benefit is to reduce the context window size for the main agent LLM, not to save on API costs."
        )
    )
    summary_template: Optional[str] = Field(
        default="Previous conversation summary: {summary}", 
        description="Template string for formatting the summary in context. Use {summary} placeholder. Example: 'Previous conversation summary: {summary}'"
    )
    summarization_config: Optional[SummarizationConfig] = Field(
        default=None, 
        description="Detailed configuration for how summarization is performed. Set model, instruction, and token limits for the summarization process."
    )
    summarization_window: Optional[int] = Field(
        default=None, 
        description=(
            "Number of messages (counting from the end, excluding always_include_last_n) to include in summarization. "
            "Example: 20. If None, all applicable messages are considered. "
            "Only the most recent summarization_window events (excluding always_include_last_n) will be summarized. Older events are discarded."
        )
    )
    always_include_last_n: Optional[int] = Field(
        default=None, 
        description=(
            "The N most recent messages to always include in full, never summarizing them. Example: 5. "
            "Critical for maintaining immediate conversation context. "
            "WARNING: For agents that use tools, always_include_last_n should be set high enough to guarantee that all relevant "
            "tool call and response events are included. Setting this too low may break tool workflows or cause the agent to lose necessary context."
        )
    )
    context_from_state: Optional[List[str]] = Field(
        default=None, 
        description=(
            "List of session state keys to inject as additional context. Example: ['user_profile', 'conversation_topic']. "
            "Values from these keys are added to the context. "
            "WARNING: Injecting session state keys into the LLM context may expose sensitive or private information to the model. "
            "Only include non-sensitive identifiers, and be aware that this data will be visible to the LLM and may affect its outputs. Use with caution."
        )
    )
    state_template: Optional[str] = Field(
        default="Session Information:\n{context}", 
        description="Template for formatting injected state context. Uses the {context} placeholder for the entire dictionary of state values. Example: 'Session Info:\n{context}'. Falls back to key-value pairs if {context} format fails."
    )
    convert_foreign_events: bool = Field(
        default=True, 
        description=(
            "Controls conversion of events from other agents to a readable format. "
            "WARNING: Disabling convert_foreign_events may break tool execution, sub-agent workflows, or cause infinite loops. "
            "Only set to False if you are certain your agent does not rely on tool or sub-agent context."
        )
    )

    @field_validator('exclude_authors')
    def validate_no_author_overlap(cls, exclude_authors, info):
        """Validate that no author is both included and excluded."""
        include_authors = info.data.get('include_authors')
        
        if include_authors and exclude_authors:
            overlap = set(include_authors) & set(exclude_authors)
            if overlap:
                raise ValueError(
                    f"Author(s) {', '.join(overlap)} cannot be in both include_authors and "
                    f"exclude_authors lists. This creates an ambiguous configuration."
                )
        return exclude_authors
        
    @model_validator(mode='after')
    def validate_summarization_config(self):
        """Validate that summarization_config is provided when summarize=True."""
        if self.summarize and not self.summarization_config:
            raise ValueError(
                "When summarize=True, summarization_config must be provided."
            )
        return self
        
    @field_validator('max_events', 'summarization_window')
    def validate_positive_integers(cls, value, info):
        """Validate that numeric fields are positive when provided."""
        if value is not None and value <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be positive when provided.")
        return value
        
    @field_validator('always_include_last_n')
    def validate_always_include_last_n(cls, always_include_last_n, info):
        """Validate that always_include_last_n is non-negative and not greater than max_events."""
        # Verificar se é não-negativo
        if always_include_last_n is not None and always_include_last_n < 0:
            raise ValueError(f"always_include_last_n must be non-negative when provided.")
            
        # Verificar se não é maior que max_events
        max_events = info.data.get('max_events')
        if (always_include_last_n is not None and 
            max_events is not None and 
            always_include_last_n > max_events):
            raise ValueError(
                f"always_include_last_n ({always_include_last_n}) cannot be greater than "
                f"max_events ({max_events})."
            )
        return always_include_last_n
        
    @field_validator('summarization_window')
    def validate_summarization_window(cls, summarization_window, info):
        """Validate that summarization_window is not greater than max_events."""
        max_events = info.data.get('max_events')
        if (summarization_window is not None and 
            max_events is not None and 
            summarization_window > max_events):
            raise ValueError(
                f"summarization_window ({summarization_window}) cannot be greater than "
                f"max_events ({max_events})."
            )
        return summarization_window

    model_config = ConfigDict(arbitrary_types_allowed=True)
