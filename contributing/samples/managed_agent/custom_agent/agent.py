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

"""Create, use, and delete a custom managed-agent resource.

This sample demonstrates the control-plane lifecycle of a custom managed agent:
creating a persistent, named agent *resource* (its persona and server-side tools
baked in), then driving it and deleting it.

You don't need a custom resource just to set a persona or server-side tools --
``ManagedAgent`` accepts both inline (``instruction=...`` and
``tools=[google_search]``; see the ``system_instruction`` and ``basic``
samples). Create a custom resource when you instead want a reusable,
server-managed agent that other apps and sessions can share by id.

Run this module with ``--create`` once to provision the resource, then drive
``root_agent`` with ``adk web`` / ``adk run
contributing/samples/managed_agent/custom_agent``, then ``--delete`` to remove
it. See the README for the required GEAP/Vertex setup.

    python contributing/samples/managed_agent/custom_agent/agent.py --create
    python contributing/samples/managed_agent/custom_agent/agent.py --delete
"""

import argparse

from dotenv import load_dotenv
from google.adk.agents import ManagedAgent

load_dotenv()

_AGENT_ID = 'adk-custom-search-agent'

_SYSTEM_INSTRUCTION = (
    'You are a concise research assistant. Use Google Search to ground every '
    'answer in current sources, cite the sources you used, and keep answers to '
    'a few sentences.'
)

root_agent = ManagedAgent(
    name='custom_managed_agent',
    agent_id=_AGENT_ID,
    environment={'type': 'remote'},
)


def main() -> None:
  """Create or delete the custom managed-agent resource."""
  parser = argparse.ArgumentParser(
      description='Create or delete the custom managed agent for this sample.'
  )
  parser.add_argument(
      '--create', action='store_true', help='Create the custom managed agent.'
  )
  parser.add_argument(
      '--delete', action='store_true', help='Delete the custom managed agent.'
  )
  args = parser.parse_args()
  if not (args.create or args.delete):
    parser.print_help()
    return

  # ManagedAgent's genai client also exposes agent create/delete.
  client = root_agent.api_client
  if args.create:
    client.agents.create(
        id=_AGENT_ID,
        base_agent='antigravity-preview-05-2026',
        system_instruction=_SYSTEM_INSTRUCTION,
        tools=[{'type': 'google_search'}],
    )
    print(f'Created "{_AGENT_ID}".')
  if args.delete:
    client.agents.delete(id=_AGENT_ID)
    print(f'Deleted "{_AGENT_ID}".')


if __name__ == '__main__':
  main()
