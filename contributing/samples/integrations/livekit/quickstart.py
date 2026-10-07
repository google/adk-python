# Copyright 2026 Google LLC
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

"""Talk to the dice agent over LiveKit, in one process.

Mints a token for the browser and joins the ADK agent into the same room. See
`livekit_worker.py` to scale calls independently of this process.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid

from fastapi import FastAPI
from google.adk.integrations.livekit import LiveKitRunner
from google.adk.runners import InMemoryRunner
from livekit import api
from livekit import rtc
import uvicorn

from ._common import APP_NAME
from ._common import livekit_credentials
from ._web import make_app
from .agent import root_agent

logger = logging.getLogger("google_adk." + __name__)

runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

# Referenced so a live session is not garbage collected mid-call.
_sessions: set[asyncio.Task[None]] = set()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
  """Ends in-flight calls when the process goes down."""
  yield
  for task in list(_sessions):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await task


app = make_app(lifespan=_lifespan)


async def _run_agent(room_name: str, user_id: str) -> None:
  """Joins the agent into `room_name` and bridges it until the call ends."""
  livekit_url, api_key, api_secret = livekit_credentials()

  # Both grants are required: without them LiveKit's voice-assistant
  # components ignore the agent and silently drop its `lk.agent.state`.
  agent_token = (
      api.AccessToken(api_key, api_secret)
      .with_identity(f"adk-agent-{room_name}")
      .with_kind("agent")
      .with_grants(
          api.VideoGrants(
              room_join=True, room=room_name, can_update_own_metadata=True
          )
      )
      .to_jwt()
  )

  room = rtc.Room()
  await room.connect(livekit_url, agent_token)
  logger.info("agent joined room %s", room_name)
  try:
    await LiveKitRunner(
        runner=runner,
        room=room,
        user_id=user_id,
        session_id=room_name,  # One room is one conversation.
    ).start()
  finally:
    await room.disconnect()
    logger.info("agent left room %s", room_name)


@app.get("/token")
async def token() -> dict:
  """Mints a join token for the browser and puts the agent in the room."""
  livekit_url, api_key, api_secret = livekit_credentials()

  room_name = f"roll-dice-{uuid.uuid4().hex[:8]}"
  identity = f"caller-{uuid.uuid4().hex[:8]}"

  task = asyncio.create_task(_run_agent(room_name, identity))
  _sessions.add(task)
  task.add_done_callback(_sessions.discard)

  join_token = (
      api.AccessToken(api_key, api_secret)
      .with_identity(identity)
      .with_grants(api.VideoGrants(room_join=True, room=room_name))
      .to_jwt()
  )
  return {"url": livekit_url, "token": join_token, "room": room_name}


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  uvicorn.run(app, host="127.0.0.1", port=8080)
