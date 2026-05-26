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

import random
from urllib.error import HTTPError

from google.adk import Context
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import node
from google.adk.workflow import RetryConfig


@node(retry_config=RetryConfig(max_attempts=5, initial_delay=1))
def get_weather(ctx: Context) -> str:
  """A mock task that fails randomly."""

  yield Event(message=f"Getting weather... attempt {ctx.attempt_count}")
  if random.random() < 0.7:  # 70% chance of failure
    raise HTTPError(
        url="http://mock-api.example.com",
        code=500,
        msg="Internal Server Error",
        hdrs={},
        fp=None,
    )

  yield "sunny"


def report_weather(node_input: str):
  yield Event(message=f"The weather is {node_input}")


root_agent = Workflow(
    name="root_agent",
    edges=[("START", get_weather, report_weather)],
)
