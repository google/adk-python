from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from app.api.custom_adk_server import CustomAdkWebServer

import json
import logging
import time
from typing import AsyncGenerator

from app.core.dependencies import get_sse_event_mapper
from app.core.mapping.sse_event_mapper import SSEEventMapper
from app.models.streaming_request import RunAgentRequestOptimized
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.utils.context_utils import Aclosing
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)


class AgentRouter:
  """Agent-related endpoints router."""

  def __init__(self, web_server_instance: "CustomAdkWebServer"):
    self.web_server = web_server_instance
    self.router = APIRouter()
    self._setup_routes()

  def _setup_routes(self):
    """Setup all agent-related routes."""

    @self.router.post("/run_sse")
    async def custom_run_agent_sse(
        req: RunAgentRequestOptimized,
        sse_event_mapper: SSEEventMapper = Depends(get_sse_event_mapper),
    ) -> StreamingResponse:
      """
      Custom implementation of the run_agent_sse endpoint with enhanced logging.
      """
      session = await self.web_server.session_service.get_session(
          app_name=req.app_name, user_id=req.user_id, session_id=req.session_id
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")

      logger.info(
          f"Starting CUSTOM SSE for app: {req.app_name} and user: {req.user_id}"
      )

      return StreamingResponse(
          self._generate_events(req, sse_event_mapper),
          media_type="text/event-stream",
      )

  async def _generate_events(
      self,
      req: RunAgentRequestOptimized,
      sse_event_mapper: SSEEventMapper,
  ) -> AsyncGenerator[str, None]:
    """Generate SSE events for the agent run."""
    try:
      yield (
          "data:"
          f" {json.dumps({'status': 'Starting custom SSE process.', 'timestamp': time.time()})}\n\n"
      )

      stream_mode = StreamingMode.SSE if req.streaming else StreamingMode.NONE

      runner = await self.web_server.get_runner_async(req.app_name)

      async with Aclosing(
          runner.run_async(
              user_id=req.user_id,
              session_id=req.session_id,
              new_message=req.new_message,
              state_delta=req.state_delta,
              run_config=RunConfig(streaming_mode=stream_mode),
          )
      ) as agen:
        async for event in agen:
          logger.debug(f"Received event: {event}")
          sse_message = sse_event_mapper.map_event_to_sse_message(
              event, req.optimization_level
          )
          if sse_message:
            yield sse_message

    except Exception as e:
      logger.error(f"Error in SSE handler: {str(e)}", exc_info=True)
      yield f'data: {json.dumps({"error": f"An error occurred: {str(e)}"})}\n\n'

  def get_router(self) -> APIRouter:
    """Returns the configured FastAPI router."""
    return self.router
