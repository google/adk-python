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

"""Microbenchmark for building LLM request contents from session history.

Times contents._get_contents over a large history with sizeable function-call
payloads. To run the benchmark:

  uv run python tests/benchmarks/benchmark_contents.py
"""

from google.adk.events.event import Event
from google.adk.flows.llm_flows import contents
from google.genai import types
import google_benchmark

_NUM_EVENTS = 500
_PAYLOAD_SIZE = 200


def _make_events(num_events: int, payload_size: int) -> list[Event]:
  """Builds a session history with sizeable function-call payloads."""
  payload = {f"k{i}": list(range(payload_size)) for i in range(10)}
  events = [
      Event(
          invocation_id="inv0",
          author="user",
          content=types.UserContent("start"),
      )
  ]
  for i in range(num_events):
    call_id = f"adk-call-{i}"
    events.append(
        Event(
            invocation_id=f"inv{i}",
            author="agent",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call_id, name="tool", args=dict(payload)
                        )
                    )
                ],
            ),
        )
    )
    events.append(
        Event(
            invocation_id=f"inv{i}",
            author="agent",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=call_id, name="tool", response=dict(payload)
                        )
                    )
                ],
            ),
        )
    )
  return events


@google_benchmark.register
def get_contents(state):
  events = _make_events(_NUM_EVENTS, _PAYLOAD_SIZE)
  while state:
    contents._get_contents(None, events, "agent")


if __name__ == "__main__":
  google_benchmark.main()
