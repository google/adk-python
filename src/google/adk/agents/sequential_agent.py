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

"""Sequential agent implementation."""

from __future__ import annotations

from typing import AsyncGenerator

from typing_extensions import override

from ..agents.invocation_context import InvocationContext
from ..events.event import Event
from .base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)


class SequentialAgent(BaseAgent):
  """A shell agent that run its sub-agents in sequence."""

  @override
  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    # Log that we're executing the sequential agent with multiple sub-agents
    logger.debug(f"SequentialAgent running with {len(self.sub_agents)} sub-agents")
    
    # Ensure context session state is using EnhancedStateDict 
    if hasattr(ctx, 'session') and hasattr(ctx.session, 'state'):
      try:
        from ..sessions.in_memory_session_service import EnhancedStateDict
        if not isinstance(ctx.session.state, dict) and not type(ctx.session.state).__name__ == 'EnhancedStateDict':
          # Convert existing state to EnhancedStateDict to ensure persistence
          existing_state = ctx.session.state
          ctx.session.state = EnhancedStateDict(existing_state)
          logger.debug(f"SequentialAgent: Upgraded session state to EnhancedStateDict")
      except (ImportError, AttributeError) as e:
        logger.warning(f"SequentialAgent: Could not upgrade session state: {e}")
    
    # Run each sub-agent with the SAME context object, preserving state
    for idx, sub_agent in enumerate(self.sub_agents):
      logger.debug(f"SequentialAgent running sub-agent {idx+1}/{len(self.sub_agents)}: {sub_agent.name}")
      
      # Log state keys to help debug
      if hasattr(ctx, 'session') and hasattr(ctx.session, 'state'):
        logger.debug(f"Context state keys before agent {sub_agent.name}: {list(ctx.session.state.keys())}")
      
      # Run the sub-agent with the SAME context object
      async for event in sub_agent.run_async(ctx):
        yield event
      
      # Log state keys after agent ran
      if hasattr(ctx, 'session') and hasattr(ctx.session, 'state'):
        logger.debug(f"Context state keys after agent {sub_agent.name}: {list(ctx.session.state.keys())}")
        
        # Print global cache status if in debug mode
        try:
          from ..sessions.in_memory_session_service import _print_global_cache
          _print_global_cache()
        except ImportError:
          pass

  @override
  async def _run_live_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for sub_agent in self.sub_agents:
      async for event in sub_agent.run_live(ctx):
        yield event
