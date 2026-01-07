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

"""Sample demonstrating AgentTool resilience: timeout, retry, and redirect patterns.

This sample shows how to handle failures, timeouts, and partial results
from downstream agents in multi-agent workflows, including:
- Timeout protection for sub-agents
- Automatic retry with ReflectAndRetryToolPlugin
- Dynamic rerouting to alternative agents
- Error handling without leaking complexity to users
"""

import asyncio
from typing import Any

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools import AgentTool
from google.adk.tools.google_search_tool import google_search
from google.adk.tools.tool_context import ToolContext
import time
from google.genai import types
from google.adk.events.event import Event


# ============================================================================
# Custom TimeoutAgentTool Wrapper
# ============================================================================

class TimeoutAgentTool(AgentTool):
  """AgentTool with timeout protection.

  This wrapper adds timeout handling to AgentTool, catching TimeoutError
  and returning a structured error response that ReflectAndRetryToolPlugin
  can process.
  """

  def __init__(
      self,
      agent,
      timeout: float = 30.0,
      timeout_error_message: str = "Sub-agent execution timed out",
      **kwargs
  ):
    """Initialize TimeoutAgentTool.

    Args:
      agent: The agent to wrap.
      timeout: Timeout in seconds for sub-agent execution.
      timeout_error_message: Custom error message for timeout.
      **kwargs: Additional arguments passed to AgentTool.
    """
    super().__init__(agent, **kwargs)
    self.timeout = timeout
    self.timeout_error_message = timeout_error_message

  async def run_async(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    """Run with timeout protection."""
    try:
      return await asyncio.wait_for(
          super().run_async(args=args, tool_context=tool_context),
          timeout=self.timeout
      )
    except asyncio.TimeoutError:
      # Return structured error that ReflectAndRetryToolPlugin can handle
      return {
          "error": "TimeoutError",
          "message": self.timeout_error_message,
          "timeout_seconds": self.timeout,
          "agent_name": self.agent.name,
      }

  async def run_async_with_events(
      self,
      *,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    """Run with timeout protection and event streaming.
    
    Note: Timeout for async generators requires careful handling.
    This implementation uses a task-based approach with timeout monitoring.
    """
    
    start_time = time.time()
    agen = super().run_async_with_events(
        args=args, tool_context=tool_context
    )
    
    try:
      while True:
        # Check overall timeout
        elapsed = time.time() - start_time
        if elapsed >= self.timeout:
          # Timeout exceeded
          yield Event(
              content=types.Content(
                  role='assistant',
                  parts=[
                      types.Part.from_text(
                          text=f"Timeout: {self.timeout_error_message}"
                      )
                  ],
              ),
          )
          return
        
        # Calculate remaining time
        remaining = self.timeout - elapsed
        if remaining <= 0:
          yield Event(
              content=types.Content(
                  role='assistant',
                  parts=[
                      types.Part.from_text(
                          text=f"Timeout: {self.timeout_error_message}"
                      )
                  ],
              ),
          )
          return
        
        # Get next event with timeout check
        try:
          event = await asyncio.wait_for(
              agen.__anext__(),
              timeout=min(remaining, 0.5)  # Check frequently
          )
          yield event
        except StopAsyncIteration:
          # Generator finished normally
          break
        except asyncio.TimeoutError:
          # This iteration timed out, but check overall timeout
          if time.time() - start_time >= self.timeout:
            yield Event(
                content=types.Content(
                    role='assistant',
                    parts=[
                        types.Part.from_text(
                            text=f"Timeout: {self.timeout_error_message}"
                        )
                    ],
                ),
            )
            return
          # Otherwise, continue waiting for next event
          continue
    except Exception:
      # Re-raise other exceptions
      raise


# ============================================================================
# Sub-Agents with Different Characteristics
# ============================================================================

# Primary agent - may be slow or fail
research_agent_primary = Agent(
    name='research_agent_primary',
    model='gemini-2.5-flash',
    description='Primary research agent for complex queries (may be slow)',
    instruction="""
    You are a thorough research assistant. When given a research task:
    1. Acknowledge the task
    2. ALWAYS use the google_search tool to find current information
    3. Break down the information into detailed steps
    4. Provide a comprehensive summary based on the search results
    
    IMPORTANT: You MUST use google_search for every research query. Do not
    respond without searching first. Be thorough and detailed in your responses.
    """,
    tools=[google_search],
)

# Fallback agent - faster, simpler
research_agent_fallback = Agent(
    name='research_agent_fallback',
    model='gemini-2.5-flash',
    description='Fallback research agent for simpler queries or when primary fails',
    instruction="""
    You are a research assistant focused on quick, concise answers.
    When given a research task:
    1. ALWAYS use the google_search tool first to find information
    2. Provide a direct, well-structured response based on the search results
    3. Keep your response concise without excessive detail
    
    IMPORTANT: You MUST use google_search for every research query. Do not
    respond without searching first.
    """,
    tools=[google_search],
)

# Specialized agent for error recovery
error_recovery_agent = Agent(
    name='error_recovery_agent',
    model='gemini-2.5-flash',
    description='Agent that handles error scenarios and provides alternative approaches',
    instruction="""
    You are an error recovery specialist. When you receive an error message
    or failure report, analyze what went wrong and suggest:
    1. What the error means
    2. Why it might have occurred
    3. Alternative approaches to achieve the goal
    4. Recommendations for the user
    
    Be helpful and constructive in your analysis.
    """,
)


# ============================================================================
# Coordinator Agent with Resilience Patterns
# ============================================================================

coordinator_agent = Agent(
    name='coordinator_agent',
    model='gemini-2.5-flash',
    description='Coordinator that manages research tasks with resilience',
    instruction="""
    You are a coordinator agent that manages research tasks by delegating to
    specialized sub-agents. Your role is to ensure tasks complete successfully
    even when individual agents fail or timeout.

    **Tool Selection Strategy:**
    1. **Primary Tool (research_agent_primary)**: Use for complex, detailed
       research tasks. This agent is thorough but may be slower.
    2. **Fallback Tool (research_agent_fallback)**: Use when:
       - The primary agent times out or fails
       - The query is simple and doesn't need deep research
       - You need a quick answer
    3. **Error Recovery Tool (error_recovery_agent)**: Use when:
       - Multiple attempts have failed
       - You need to understand what went wrong
       - You need alternative approaches suggested

    **Error Handling Protocol:**
    - If research_agent_primary returns an error or timeout:
      1. First, try research_agent_fallback with the same query
      2. If that also fails, use error_recovery_agent to analyze the failure
      3. Present the error_recovery_agent's analysis to the user
      4. Suggest next steps based on the analysis

    **User Communication:**
    - Always present results clearly, even if they come from fallback agents
    - If errors occur, explain what happened and what you tried
    - Never expose internal error details or retry counts to users
    - Frame fallbacks as "using a different approach" rather than "fallback"

    **Example Flow:**
    User: "Research quantum computing applications"
    1. Try research_agent_primary
    2. If timeout/error → Try research_agent_fallback
    3. If still fails → Use error_recovery_agent to understand why
    4. Present final result or error analysis to user
    """,
    tools=[
        # Primary agent with timeout protection
        # For testing timeouts, set a very short timeout (e.g., 5.0 seconds)
        # For production, use a longer timeout (e.g., 30.0 seconds)
        TimeoutAgentTool(
            agent=research_agent_primary,
            timeout=10.0,  # Change to 5.0 for timeout testing
            timeout_error_message="Primary research agent timed out after 10 seconds",
            skip_summarization=True,
        ),
        # Fallback agent timeout
        # For testing: Set to 5.0 to test full failure chain (primary → fallback → error recovery)
        # For production: Set to 60.0 to allow fallback to succeed after primary timeout
        TimeoutAgentTool(
            agent=research_agent_fallback,
            timeout=60.0,  # Set to 60.0 to test successful fallback after primary timeout
            timeout_error_message="Fallback research agent timed out",
            skip_summarization=True,
        ),
        # Error recovery agent
        AgentTool(
            agent=error_recovery_agent,
            skip_summarization=True,
        ),
    ],
)

# ============================================================================
# App Configuration with Retry Plugin
# ============================================================================

# Configure retry plugin for automatic retry handling
retry_plugin = ReflectAndRetryToolPlugin(
    max_retries=2,  # Allow 2 retries per tool before giving up
    throw_exception_if_retry_exceeded=False,  # Return guidance instead of raising
    tracking_scope=None,  # Use default (per-invocation)
)

app = App(
    name='agent_tool_resilience',
    root_agent=coordinator_agent,
    plugins=[retry_plugin],
)

root_agent = coordinator_agent

