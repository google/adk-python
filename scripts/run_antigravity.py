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

"""Runner script to execute prompts/commands via the Antigravity SDK."""

import asyncio
import os
import sys

try:
  from google.antigravity import Agent
  from google.antigravity import CapabilitiesConfig
  from google.antigravity import LocalAgentConfig
except ImportError:
  print(
      "Error: google-antigravity package is not installed. Run 'pip install"
      " google-antigravity'",
      file=sys.stderr,
  )
  sys.exit(1)


async def main():
  if len(sys.argv) < 2:
    print("Usage: python run_antigravity.py <prompt>", file=sys.stderr)
    sys.exit(1)

  prompt = " ".join(sys.argv[1:])

  # Ensure GEMINI_API_KEY is set (using GOOGLE_API_KEY as fallback)
  if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

  if "GEMINI_API_KEY" not in os.environ:
    print(
        "Error: GEMINI_API_KEY environment variable is not set.",
        file=sys.stderr,
    )
    sys.exit(1)

  config = LocalAgentConfig(
      capabilities=CapabilitiesConfig(),
  )

  try:
    async with Agent(config) as agent:
      response = await agent.chat(prompt)
      async for token in response:
        sys.stdout.write(token)
        sys.stdout.flush()
      print()
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"\nError running Antigravity Agent: {e}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  asyncio.run(main())
