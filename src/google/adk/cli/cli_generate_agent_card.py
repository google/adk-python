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
from __future__ import annotations

import asyncio
import json
import os

import click

from .utils.agent_loader import AgentLoader


@click.command(name="generate_agent_card")
@click.option(
    "--protocol",
    default="https",
    help="Protocol for the agent URL (default: https)",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host for the agent URL (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default="8000",
    help="Port for the agent URL (default: 8000)",
)
@click.option(
    "--create-file",
    is_flag=True,
    default=False,
    help="Create agent.json file in each agent directory",
)
def generate_agent_card(
    protocol: str, host: str, port: str, create_file: bool
) -> None:
  """Generates agent cards for all detected agents."""
  asyncio.run(_generate_agent_card_async(protocol, host, port, create_file))


async def _generate_agent_card_async(
    protocol: str, host: str, port: str, create_file: bool
) -> None:
  try:
    from ..a2a.utils.agent_card_builder import AgentCardBuilder
  except ImportError:
    click.secho(
        "Error: 'a2a' package is required for this command. "
        "Please install it with 'pip install google-adk[a2a]'.",
        fg="red",
        err=True,
    )
    raise click.Abort()

  cwd = os.getcwd()
  loader = AgentLoader(agents_dir=cwd)
  agent_names = loader.list_agents()
  agent_cards = []

  for agent_name in agent_names:
    try:
      agent = loader.load_agent(agent_name)
      # If it's an App, get the root agent
      if hasattr(agent, "root_agent"):
        agent = agent.root_agent
      builder = AgentCardBuilder(
          agent=agent,
          rpc_url=f"{protocol}://{host}:{port}/{agent_name}",
      )
      card = await builder.build()
      card_dict = card.model_dump(exclude_none=True)
      agent_cards.append(card_dict)

      if create_file:
        agent_dir = os.path.join(cwd, agent_name)
        agent_json_path = os.path.join(agent_dir, "agent.json")
        with open(agent_json_path, "w", encoding="utf-8") as f:
          json.dump(card_dict, f, indent=2)
    except Exception as e:
      # Log error but continue with other agents
      # Using click.echo to print to stderr to not mess up JSON output on stdout
      click.echo(f"Error processing agent {agent_name}: {e}", err=True)

  click.echo(json.dumps(agent_cards, indent=2))
