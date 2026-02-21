"""Shared utilities for GraphAgent examples.

Provides:
- USE_LLM flag to toggle between deterministic agents and real LLM endpoints
- Helper to create LLM-powered agents with consistent configuration
"""

import os
import sys


def use_llm_mode() -> bool:
  """Check if examples should use real LLM endpoints instead of deterministic agents.

  Returns True if:
  - Environment variable USE_LLM=1 or USE_LLM=true
  - Command-line flag --use-llm is present

  Default: False (use deterministic BaseAgent implementations)
  """
  # Check environment variable
  env_use_llm = os.getenv("USE_LLM", "").lower() in ("1", "true", "yes")

  # Check command-line args
  arg_use_llm = "--use-llm" in sys.argv

  return env_use_llm or arg_use_llm


def create_llm_agent(
    name: str,
    instruction=None,
    model: str = "gemini-2.5-flash",
    tools: list = None,
    **kwargs,
):
  """Create an LLM-powered agent.

  Args:
      name: Agent name
      instruction: System instruction (str or callable for dynamic instructions)
      model: Model to use (default: gemini-2.5-flash)
      tools: Optional list of tools
      **kwargs: Additional Agent configuration (e.g., include_contents='none')

  Returns:
      Agent instance configured with LLM

  Note:
      Requires valid API credentials configured via:
      - GOOGLE_GENAI_API_KEY environment variable, or
      - gcloud auth application-default login
  """
  from google.adk import Agent

  return Agent(
      name=name,
      model=model,
      instruction=instruction,
      tools=tools or [],
      **kwargs,
  )
