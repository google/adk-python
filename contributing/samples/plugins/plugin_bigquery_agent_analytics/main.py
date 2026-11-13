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


"""Logs agent events to BigQuery with the agent analytics plugin.

Needs the bigquery-analytics extra: pip install "google-adk[bigquery-analytics]"

Set BIGQUERY_ANALYTICS_DATASET to a dataset that already exists. The plugin
creates the table inside it.
"""

import asyncio
import os
import sys

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins import BigQueryAgentAnalyticsPlugin
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

APP_NAME = 'plugin_bigquery_agent_analytics'
TABLE_ID = 'agent_events'

PROMPTS = [
    'What is the status of order A1?',
    'And order B2?',
]


async def lookup_order(tool_context: ToolContext, order_id: str) -> dict:
  """Returns a canned delivery status for an order."""
  return {'order_id': order_id, 'status': 'in transit'}


root_agent = Agent(
    name='order_agent',
    description='Looks up delivery status for orders.',
    instruction='Use the lookup_order tool to answer questions about an order.',
    tools=[lookup_order],
)


async def main():
  project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
  dataset_id = os.environ.get('BIGQUERY_ANALYTICS_DATASET')
  if not project_id or not dataset_id:
    sys.exit(
        'Set GOOGLE_CLOUD_PROJECT and BIGQUERY_ANALYTICS_DATASET to a project'
        ' and an existing BigQuery dataset.'
    )

  plugin = BigQueryAgentAnalyticsPlugin(
      project_id=project_id,
      dataset_id=dataset_id,
      table_id=TABLE_ID,
  )
  # Leaving the block closes the runner, which closes the plugin and its writer.
  async with InMemoryRunner(
      app=App(name=APP_NAME, root_agent=root_agent, plugins=[plugin])
  ) as runner:
    session = await runner.session_service.create_session(
        user_id='user', app_name=APP_NAME
    )
    for prompt in PROMPTS:
      print(f'\nuser: {prompt}')
      async for event in runner.run_async(
          user_id='user',
          session_id=session.id,
          new_message=types.Content(
              role='user', parts=[types.Part.from_text(text=prompt)]
          ),
      ):
        if event.content and event.content.parts and event.author != 'user':
          for part in event.content.parts:
            if part.text:
              print(f'agent: {part.text.strip()}')

  # The plugin fails open: a write or setup failure is logged and counted here
  # rather than raised. Every reason is reported, most of them as zero.
  dropped = {
      reason: count
      for reason, count in plugin.get_drop_stats().items()
      if count
  }
  if dropped:
    print(f'\nThe plugin reported dropped or degraded events: {dropped}')
    return
  print(f'\nEvents written to {project_id}.{dataset_id}.{TABLE_ID}')
  print('Inspect them with:')
  print(
      '  bq query --nouse_legacy_sql "SELECT event_type, COUNT(*) AS events'
      f' FROM \\`{project_id}.{dataset_id}.{TABLE_ID}\\` GROUP BY event_type"'
  )


if __name__ == '__main__':
  asyncio.run(main())
