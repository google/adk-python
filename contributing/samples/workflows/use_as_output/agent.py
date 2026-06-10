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

from google.adk import Agent
from google.adk import Context
from google.adk.workflow import node
from google.adk.workflow import Workflow
from google.adk.workflow._base_node import START

summarizer = Agent(
    name='summarizer',
    instruction='Summarize the following text in one sentence.',
)


@node(rerun_on_resume=True)
async def orchestrate(ctx: Context, node_input: str) -> str:
  return await ctx.run_node(
      summarizer, node_input=node_input, use_as_output=True
  )


def finalize(node_input: str) -> str:
  return f'final: {node_input}'


root_agent = Workflow(
    name='root_agent',
    edges=[(START, orchestrate, finalize)],
)
