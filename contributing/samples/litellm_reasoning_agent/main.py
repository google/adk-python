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

import asyncio

from agent import agent, inspector
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def main():
  session_service = InMemorySessionService()
  runner = Runner(app_name="test", agent=agent, session_service=session_service)

  await session_service.create_session(
      app_name="test",
      user_id="user",
      session_id="session",
      state={},
  )

  message = types.Content(
      role="user",
      parts=[types.Part(text="Explain quantum computing in detail.")],
  )

  async for _ in runner.run_async(
      user_id="user", session_id="session", new_message=message
  ):
    pass

  print(f"finish_reason: {inspector.captured['finish_reason']}")


if __name__ == "__main__":
  asyncio.run(main())
