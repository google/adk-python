"""
RLM Agent implementation using Google ADK BaseAgent.

This is the core agent that implements the Recursive Language Model
pattern using ADK's agent abstractions and streaming events.
"""

import asyncio
import logging
import time
from typing import Any
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

from adk_rlm.callbacks.code_execution import find_code_blocks
from adk_rlm.callbacks.code_execution import find_final_answer
from adk_rlm.callbacks.code_execution import format_iteration
from adk_rlm.code_executor import RLMCodeExecutor
from adk_rlm.events import RLMEventData
from adk_rlm.events import RLMEventType
from adk_rlm.llm import AsyncLLMRateLimiter
from adk_rlm.logging.rlm_logger import RLMLogger
from adk_rlm.logging.verbose import VerbosePrinter
from adk_rlm.prompts import build_rlm_system_prompt
from adk_rlm.prompts import build_user_prompt
from adk_rlm.prompts import RLM_SYSTEM_PROMPT
from adk_rlm.types import CodeBlock
from adk_rlm.types import QueryMetadata
from adk_rlm.types import REPLResult
from adk_rlm.types import RLMIteration
from adk_rlm.types import RLMMetadata
from adk_rlm.usage import UsageTracker
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types
from pydantic import PrivateAttr

from google import genai


class RLMAgent(BaseAgent):
  """
  Recursive Language Model agent using Google ADK BaseAgent.

  This agent implements the RLM pattern where an LLM can execute Python
  code in a REPL environment, including making recursive LLM calls.
  It emits granular streaming events for UI integration.
  """

  # Pydantic model fields (public configuration)
  model: str = "gemini-3-pro-preview"
  sub_model: str | None = None
  max_iterations: int = 30
  max_depth: int = 5
  current_depth: int = 0  # Current recursion depth (0 = root level)
  custom_system_prompt: str | None = None
  persistent: bool = False

  # Private attributes (not part of the model schema)
  _code_executor: RLMCodeExecutor | None = PrivateAttr(default=None)
  _client: genai.Client | None = PrivateAttr(default=None)
  _usage_tracker: UsageTracker = PrivateAttr(default_factory=UsageTracker)
  _logger: RLMLogger | None = PrivateAttr(default=None)
  _parent_agent: str | None = PrivateAttr(default=None)
  _verbose: VerbosePrinter = PrivateAttr(
      default_factory=lambda: VerbosePrinter(enabled=False)
  )
  _persistent_executor: RLMCodeExecutor | None = PrivateAttr(default=None)
  _ancestry: list[dict] = PrivateAttr(default_factory=list)

  def __init__(
      self,
      name: str = "rlm_agent",
      model: str = "gemini-3-pro-preview",
      sub_model: str | None = None,
      max_iterations: int = 30,
      max_depth: int = 5,
      current_depth: int = 0,
      custom_system_prompt: str | None = None,
      logger: RLMLogger | None = None,
      parent_agent: str | None = None,
      verbose: bool = False,
      persistent: bool = False,
      ancestry: list[dict] | None = None,
      **kwargs,
  ):
    """
    Initialize the RLM Agent.

    Args:
        name: Agent name for identification.
        model: The main model to use for RLM reasoning.
        sub_model: The model to use for sub-LLM calls (defaults to model).
        max_iterations: Maximum number of RLM iterations.
        max_depth: Maximum recursion depth for nested llm_query calls.
        current_depth: Current recursion depth (0 = root level).
        custom_system_prompt: Custom system prompt (uses default if None).
        logger: Optional JSONL logger for trajectory logging.
        parent_agent: Name of the parent agent (for nested agents).
        verbose: Whether to print verbose Rich output.
        persistent: Whether to persist REPL state across calls.
        ancestry: List of ancestor agent context dicts for event tagging.
        **kwargs: Additional arguments for BaseAgent.
    """
    super().__init__(
        name=name,
        model=model,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_depth=max_depth,
        current_depth=current_depth,
        custom_system_prompt=custom_system_prompt,
        persistent=persistent,
        **kwargs,
    )

    # Initialize private attributes
    self._client = genai.Client(vertexai=True, location="global")
    self._usage_tracker = UsageTracker()
    self._logger = logger
    self._parent_agent = parent_agent
    self._verbose = VerbosePrinter(enabled=verbose)
    self._persistent_executor = None
    self._ancestry = ancestry.copy() if ancestry else []

    # Log/print metadata
    if self._logger or verbose:
      metadata = self._build_metadata()
      if self._logger:
        self._logger.log_metadata(metadata)
      self._verbose.print_metadata(metadata)

  @property
  def _effective_sub_model(self) -> str:
    """Return the effective sub-model (defaults to main model)."""
    return self.sub_model or self.model

  @property
  def _system_prompt(self) -> str:
    """Return the effective system prompt."""
    return self.custom_system_prompt or RLM_SYSTEM_PROMPT

  def _build_metadata(self) -> RLMMetadata:
    """Build metadata about this RLM configuration."""
    return RLMMetadata(
        root_model=self.model,
        max_depth=self.max_depth,
        max_iterations=self.max_iterations,
        backend="gemini",
        backend_kwargs={"model_name": self.model},
        environment_type="local",
        environment_kwargs={},
        other_backends=[self._effective_sub_model]
        if self._effective_sub_model != self.model
        else None,
    )

  def _prepare_contents(
      self, prompt: list[dict[str, Any]]
  ) -> tuple[list[types.Content], str | None]:
    """Convert message history to Gemini format."""
    system_instruction = None
    contents = []

    for msg in prompt:
      role = msg.get("role")
      content = msg.get("content", "")

      if role == "system":
        system_instruction = content
      elif role == "user":
        contents.append(
            types.Content(role="user", parts=[types.Part(text=content)])
        )
      elif role == "assistant":
        contents.append(
            types.Content(role="model", parts=[types.Part(text=content)])
        )

    return contents, system_instruction

  async def _call_llm_async(self, message_history: list[dict[str, Any]]) -> str:
    """Call the main LLM asynchronously."""
    contents, system_instruction = self._prepare_contents(message_history)

    # Build config with function calling disabled to prevent MALFORMED_FUNCTION_CALL errors
    # when the model tries to use tools that aren't configured
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        ),
    )

    async with AsyncLLMRateLimiter():
      response = await self._client.aio.models.generate_content(
          model=self.model,
          contents=contents,
          config=config,
      )

    self._usage_tracker.add_from_response(self.model, response.usage_metadata)

    # Handle None/empty responses with detailed logging
    if response.text is None or response.text == "":
      # Extract debugging info from response
      finish_reason = None
      safety_ratings = None
      block_reason = None

      if response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        safety_ratings = getattr(candidate, "safety_ratings", None)
      if hasattr(response, "prompt_feedback"):
        block_reason = getattr(response.prompt_feedback, "block_reason", None)

      logger.warning(
          "LLM returned empty response: model=%s, finish_reason=%s, "
          "block_reason=%s, safety_ratings=%s, usage=%s",
          self.model,
          finish_reason,
          block_reason,
          safety_ratings,
          response.usage_metadata,
      )

      # Return informative message instead of empty string
      reason_parts = []
      if finish_reason:
        reason_parts.append(f"finish_reason={finish_reason}")
      if block_reason:
        reason_parts.append(f"block_reason={block_reason}")
      reason_str = ", ".join(reason_parts) if reason_parts else "unknown reason"

      return f"[LLM returned empty response: {reason_str}]"

    return response.text

  def _create_rlm_event(
      self,
      ctx: InvocationContext,
      event_type: RLMEventType,
      **data,
  ) -> Event:
    """Create an ADK Event with RLM-specific metadata."""
    event_data = RLMEventData(event_type=event_type, **data)
    metadata = event_data.to_dict()

    # Add agent identification for proper UI rendering
    metadata["agent_name"] = self.name
    metadata["agent_depth"] = self.current_depth
    metadata["ancestry"] = self._ancestry

    return Event(
        invocation_id=ctx.invocation_id,
        author=self.name,
        custom_metadata=metadata,
    )

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    """
    Core RLM iteration loop with granular event streaming.

    This is the main entry point called by the ADK Runner.
    """
    start_time = time.perf_counter()

    # Emit run start
    yield self._create_rlm_event(
        ctx,
        RLMEventType.RUN_START,
        model=self.model,
        metadata={
            "sub_model": self._effective_sub_model,
            "max_iterations": self.max_iterations,
        },
    )

    try:
      # Get context from session state
      context_payload = (
          ctx.session.state.get("rlm_context") if ctx.session else None
      )
      root_prompt = ctx.session.state.get("rlm_prompt") if ctx.session else None

      if context_payload is None:
        yield self._create_rlm_event(
            ctx,
            RLMEventType.RUN_ERROR,
            error="No context provided in session state",
        )
        return

      # Get conversation history if present (list of {role, content} messages)
      conversation_history = (
          ctx.session.state.get("rlm_conversation_history")
          if ctx.session
          else None
      )

      # Create or reuse code executor
      if self.persistent and self._persistent_executor is not None:
        executor = self._persistent_executor
        executor.add_context(context_payload)
      else:
        executor = RLMCodeExecutor(
            sub_model=self._effective_sub_model,
            current_depth=self.current_depth,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            usage_tracker=self._usage_tracker,
            logger=self._logger,
            parent_agent=self.name,
            ancestry=self._ancestry,
        )
        executor.load_context(context_payload)
        if self.persistent:
          self._persistent_executor = executor

      # Build initial message history
      query_metadata = QueryMetadata(context_payload)
      message_history = build_rlm_system_prompt(
          self._system_prompt, query_metadata
      )

      # Prepend conversation history if present (multi-turn conversation)
      if conversation_history:
        # Insert previous conversation turns after system prompt
        # Format: user messages become "user", assistant messages become "assistant"
        conv_messages = []
        for msg in conversation_history:
          role = msg.get("role", "user")
          content = msg.get("content", "")
          if role == "user":
            conv_messages.append({
                "role": "user",
                "content": f"[Previous question from user]: {content}",
            })
          elif role == "assistant":
            conv_messages.append({
                "role": "assistant",
                "content": f"[Your previous answer]: {content}",
            })
        # Add a separator to indicate the new question
        if conv_messages:
          conv_messages.append({
              "role": "user",
              "content": (
                  "[End of conversation history. The user is now asking a"
                  " follow-up question below. Use the context from your"
                  " previous answers to provide a coherent response.]"
              ),
          })
          conv_messages.append({
              "role": "assistant",
              "content": (
                  "I understand. I'll consider my previous answers when"
                  " responding to the follow-up question."
              ),
          })
          # Insert after system prompt (first 2 messages)
          message_history = (
              message_history[:2] + conv_messages + message_history[2:]
          )

      final_answer = None

      # Main iteration loop
      for i in range(self.max_iterations):
        # Emit iteration start
        yield self._create_rlm_event(
            ctx,
            RLMEventType.ITERATION_START,
            iteration=i + 1,
        )

        # Build current prompt
        context_count = executor.get_context_count()
        history_count = executor.get_history_count()
        current_prompt = message_history + [
            build_user_prompt(root_prompt, i, context_count, history_count)
        ]

        # Emit LLM call start
        yield self._create_rlm_event(
            ctx,
            RLMEventType.LLM_CALL_START,
            iteration=i + 1,
            model=self.model,
        )

        # Call LLM
        response_text = await self._call_llm_async(current_prompt)

        # Emit LLM call end
        yield self._create_rlm_event(
            ctx,
            RLMEventType.LLM_CALL_END,
            iteration=i + 1,
            response_preview=response_text[:500] if response_text else None,
            response_full=response_text,
        )

        # Find and execute code blocks
        code_block_strs = find_code_blocks(response_text)
        code_blocks = []

        for j, code_str in enumerate(code_block_strs):
          # Emit code found
          yield self._create_rlm_event(
              ctx,
              RLMEventType.CODE_FOUND,
              iteration=i + 1,
              block_index=j,
              code=code_str[:200] if code_str else None,
              code_full=code_str,
          )

          # Emit code execution start
          yield self._create_rlm_event(
              ctx,
              RLMEventType.CODE_EXEC_START,
              iteration=i + 1,
              block_index=j,
          )

          # Set iteration context so child events can reference parent iteration
          executor.set_iteration_context(i + 1, j)

          # Execute code asynchronously while streaming child events in real-time
          from google.adk.code_executors.code_execution_utils import CodeExecutionInput

          # Reset queue state BEFORE starting the task to avoid race conditions
          executor.reset_event_state()

          # Start code execution as a background task
          exec_task = asyncio.create_task(
              executor.execute_code_async(
                  ctx, CodeExecutionInput(code=code_str)
              )
          )

          # Poll for child events while execution runs
          async for child_event in executor.poll_child_events():
            yield child_event

          # Wait for execution to complete
          exec_result = await exec_task

          # Yield any remaining events that arrived after polling stopped
          for remaining_event in executor.pop_child_events():
            yield remaining_event

          # Create REPLResult for compatibility
          repl_result = REPLResult(
              stdout=exec_result.stdout,
              stderr=exec_result.stderr,
              locals=executor.locals.copy(),
              execution_time=0.0,
          )
          code_blocks.append(CodeBlock(code=code_str, result=repl_result))

          # Emit code execution end
          yield self._create_rlm_event(
              ctx,
              RLMEventType.CODE_EXEC_END,
              iteration=i + 1,
              block_index=j,
              output=exec_result.stdout[:1000] if exec_result.stdout else None,
              output_full=exec_result.stdout,
              error=exec_result.stderr[:500] if exec_result.stderr else None,
              error_full=exec_result.stderr,
              has_error=bool(exec_result.stderr),
          )

        # Create iteration object for logging
        iteration = RLMIteration(
            prompt=current_prompt,
            response=response_text,
            code_blocks=code_blocks,
        )

        # Check for final answer in response text
        final_answer = find_final_answer(response_text, None)

        # Also check for FINAL_ANSWER variable
        if final_answer is None and executor.final_answer:
          final_answer = executor.final_answer

        # Also check REPL locals for FINAL_VAR pattern
        if final_answer is None:
          final_answer = find_final_answer(response_text, executor._repl)

        iteration.final_answer = final_answer

        # Log iteration
        if self._logger:
          self._logger.log(
              iteration,
              depth=self.current_depth,
              agent_name=self.name,
              parent_agent=self._parent_agent,
          )

        # Verbose output
        self._verbose.print_iteration(iteration, i + 1)

        if final_answer is not None:
          # Emit final detected
          yield self._create_rlm_event(
              ctx,
              RLMEventType.FINAL_DETECTED,
              iteration=i + 1,
              source="text"
              if find_final_answer(response_text, None)
              else "variable",
          )

          # Emit final answer
          yield self._create_rlm_event(
              ctx,
              RLMEventType.FINAL_ANSWER,
              answer=final_answer,
              total_iterations=i + 1,
              execution_time_ms=(time.perf_counter() - start_time) * 1000,
          )

          # Store history if persistent
          if self.persistent:
            executor.add_history(message_history)

          # Emit run end
          yield self._create_rlm_event(
              ctx,
              RLMEventType.RUN_END,
              success=True,
              total_iterations=i + 1,
          )

          self._verbose.print_final_answer(final_answer)
          self._verbose.print_summary(
              i + 1,
              time.perf_counter() - start_time,
              self._usage_tracker.get_summary().to_dict(),
          )
          return

        # Emit iteration end
        yield self._create_rlm_event(
            ctx,
            RLMEventType.ITERATION_END,
            iteration=i + 1,
        )

        # Format iteration for next prompt
        new_messages = format_iteration(iteration)
        message_history.extend(new_messages)

      # Max iterations reached - generate fallback answer
      fallback = await self._generate_fallback_answer_async(message_history)

      yield self._create_rlm_event(
          ctx,
          RLMEventType.FINAL_ANSWER,
          answer=fallback,
          total_iterations=self.max_iterations,
          execution_time_ms=(time.perf_counter() - start_time) * 1000,
      )

      yield self._create_rlm_event(
          ctx,
          RLMEventType.RUN_END,
          success=True,
          total_iterations=self.max_iterations,
          fallback=True,
      )

      if self.persistent:
        executor.add_history(message_history)

      self._verbose.print_final_answer(fallback)
      self._verbose.print_summary(
          self.max_iterations,
          time.perf_counter() - start_time,
          self._usage_tracker.get_summary().to_dict(),
      )

    except Exception as e:
      yield self._create_rlm_event(
          ctx,
          RLMEventType.RUN_ERROR,
          error=str(e),
          metadata={"error_type": type(e).__name__},
      )
      raise

  async def _generate_fallback_answer_async(
      self, message_history: list[dict[str, Any]]
  ) -> str:
    """Generate a default answer when max iterations is reached."""
    fallback_prompt = message_history + [{
        "role": "user",
        "content": (
            "Please provide a final answer to the user's question based on the"
            " information gathered so far."
        ),
    }]
    response = await self._call_llm_async(fallback_prompt)

    if self._logger:
      self._logger.log(
          RLMIteration(
              prompt=fallback_prompt,
              response=response,
              final_answer=response,
              code_blocks=[],
          ),
          depth=self.current_depth,
          agent_name=self.name,
          parent_agent=self._parent_agent,
      )

    return response

  def close(self) -> None:
    """Clean up persistent environment."""
    if self._persistent_executor is not None:
      self._persistent_executor.cleanup()
      self._persistent_executor = None

  def __enter__(self) -> "RLMAgent":
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
    self.close()
    return False
