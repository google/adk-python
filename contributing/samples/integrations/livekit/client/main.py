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

"""Token and dispatch server for the LiveKit browser client.

A browser cannot hold the LiveKit API secret, so this backend mints its join
token and dispatches the worker. Run it alongside `livekit_worker.py`.
"""

from __future__ import annotations

import json
import uuid

from livekit import api
import uvicorn

from .._common import APP_NAME
from .._common import livekit_credentials
from .._web import make_app

app = make_app()


@app.get("/token")
async def token(room: str | None = None, identity: str | None = None) -> dict:
  """Mints a join token and dispatches the ADK worker into the room.

  Args:
    room: The room to join, which also becomes the ADK session_id.
    identity: The participant identity, reused as the ADK user_id.

  Returns:
    The LiveKit server URL and a join token for the browser.
  """
  livekit_url, api_key, api_secret = livekit_credentials()

  room = room or f"roll-dice-{uuid.uuid4().hex[:8]}"
  identity = identity or f"caller-{uuid.uuid4().hex[:8]}"

  # LiveKit has no user_id / session_id; pass them as job metadata.
  metadata = json.dumps({"user_id": identity, "session_id": room})

  async with api.LiveKitAPI(
      url=livekit_url, api_key=api_key, api_secret=api_secret
  ) as lkapi:
    await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name=APP_NAME, room=room, metadata=metadata
        )
    )

  join_token = (
      api.AccessToken(api_key, api_secret)
      .with_identity(identity)
      .with_grants(api.VideoGrants(room_join=True, room=room))
      .to_jwt()
  )

  return {"url": livekit_url, "token": join_token, "room": room}


if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8080)
