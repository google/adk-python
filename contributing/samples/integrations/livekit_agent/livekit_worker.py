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

"""Runs the dice agent as a LiveKit worker, one job per call. See README.md."""

from __future__ import annotations

import json

from google.adk.integrations.livekit import LiveKitRunner
from google.adk.runners import InMemoryRunner
from livekit.agents import AgentServer
from livekit.agents import cli
from livekit.agents import JobContext

from ._common import APP_NAME
from .agent import root_agent

runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)

server = AgentServer()


# Drop `agent_name` to auto-dispatch this agent into every room.
@server.rtc_session(agent_name=APP_NAME)
async def entrypoint(ctx: JobContext) -> None:
  """Bridges one dispatched call into the ADK agent."""
  await ctx.connect()

  # ADK's ids ride in the job metadata; see client/main.py.
  meta = json.loads(ctx.job.metadata or "{}")

  await LiveKitRunner(
      runner=runner,
      room=ctx.room,
      user_id=meta.get("user_id", "live-user"),
      session_id=meta.get("session_id", ctx.room.name),
  ).start()


if __name__ == "__main__":
  cli.run_app(server)
