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

"""Small, deterministic long-history fixtures; real ADK code, no model calls."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from google.adk.apps.compaction import _latest_compaction_event
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.event_actions import EventCompaction
from google.adk.flows.llm_flows.context import _compaction
from google.adk.flows.llm_flows.context._contents import _get_contents
from google.genai import types


def history(n):
  result = []
  for j in range(n):
    start = float(j * 10 + 1)
    for k in range(8):
      result.append(
          Event(
              timestamp=start + k,
              invocation_id=f'turn-{j}',
              author='agent',
              content=types.Content(
                  role='model',
                  parts=[types.Part(text=f'old turn {j}, step {k}')],
              ),
          )
      )
    result.append(
        Event(
            timestamp=start + 8,
            author='user',
            invocation_id=f'summary-{j}',
            actions=EventActions(
                compaction=EventCompaction(
                    start_timestamp=start,
                    end_timestamp=start + 7,
                    compacted_content=types.Content(
                        role='model',
                        parts=[types.Part(text=f'Completed turn {j}.')],
                    ),
                )
            ),
        )
    )
  result.append(
      Event(
          timestamp=float(n * 10 + 1),
          invocation_id='latest',
          author='user',
          content=types.Content(
              role='user',
              parts=[types.Part(text='Continue with the next step.')],
          ),
      )
  )
  return result


def digest(events):
  data = [e.model_dump(mode='json', exclude={'id'}) for e in events]
  return hashlib.sha256(
      json.dumps(data, sort_keys=True, allow_nan=True).encode()
  ).hexdigest()


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--output', required=True)
  p.add_argument('--repeats', type=int, default=5)
  args = p.parse_args()
  records = []
  for size in [64, 256, 1024]:
    events = history(size)
    functions = {
        'process': lambda: _compaction._process_compaction_events(
            events, 'agent'
        ),
        'contents': lambda: _get_contents(None, events, 'agent'),
        'latest': lambda: _latest_compaction_event(events),
    }
    row = {
        'summaries': size,
        'raw_events': 8 * size + 1,
        'input_events': len(events),
    }
    for name, fun in functions.items():
      fun()
      samples = []
      for _ in range(args.repeats):
        before = time.perf_counter_ns()
        value = fun()
        samples.append(time.perf_counter_ns() - before)
      row[name] = {
          'ns_samples': samples,
          'median_ms': statistics.median(samples) / 1e6,
      }
      if name == 'process':
        assert len(value) == size + 1
        row['semantic_digest'] = digest(value)
      if name == 'contents':
        row['contents_digest'] = hashlib.sha256(
            json.dumps(
                [c.model_dump(mode='json') for c in value], sort_keys=True
            ).encode()
        ).hexdigest()
      if name == 'latest':
        assert value.invocation_id == f'summary-{size-1}'
    records.append(row)
    print(json.dumps(row), flush=True)
  Path(args.output).write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
  main()
