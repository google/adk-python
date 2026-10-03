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

"""Real-LLM: SpeculativeRouterNode overlaps a downstream handler with generation.

Contrived to isolate the speculative win, on a whole real arXiv paper -- NO
chunking, no truncation, one streaming call.

The model reads the paper and is told to FIRST emit a directive line:

    TOOL_CALL: {"name": "lookup", "arguments": {"topic": "<subject>"}}

...THEN write a long analysis. A downstream ``lookup`` handler (here: a fixed
2s sleep, standing in for a retrieval/DB call) depends on that directive.

  A. sequential   -- wait for the whole generation, then run the handler.
  B. speculative  -- the instant the (repaired, partial) directive streams in,
                     dispatch the handler so it runs *while* the model is still
                     writing the analysis; verify against the finalized directive.

B must be faster by ~the handler's duration, because that work overlaps the long
tail of generation instead of following it.

    ADK_TEST_MODEL=gemini-3.5-flash-lite \\
      uv run pytest -s -p no:cacheprovider \\
      tests/integration/test_speculative_router_arxiv.py

Requires Vertex (GOOGLE_CLOUD_PROJECT via ADC), ``pypdf`` and network access to
arXiv; skips otherwise.
"""

import asyncio
import os
import pathlib
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from dotenv import load_dotenv
from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import SpeculativeRouterNode
from google.genai import types
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / '.env', override=False)

pytestmark = pytest.mark.skipif(
    not os.environ.get('GOOGLE_CLOUD_PROJECT'),
    reason='Real-LLM speculative test requires Vertex (GOOGLE_CLOUD_PROJECT).',
)

MODEL = os.environ.get('ADK_TEST_MODEL', 'gemini-3.5-flash-lite')

# The relevant paper; whole document, no chunking.
_ARXIV_ID = '1706.03762'  # Attention Is All You Need

# Stand-in for a real downstream dependency (retrieval / DB / API call). Fixed so
# the overlap saving is deterministic and measurable.
_HANDLER_SECONDS = 2.0

# The directive line the model must emit first. Kept out of the f-string so the
# literal JSON braces are not doubled.
_DIRECTIVE = (
    'FIRST output EXACTLY one line, nothing else on it:\n'
    'TOOL_CALL: {"name": "lookup", "arguments": {"topic":'
    ' "<one or two word subject of the paper>"}}\n'
    'THEN write a long, detailed, multi-paragraph summary of the paper (at'
    ' least 300 words).'
)


def _download_paper_text(arxiv_id: str) -> str:
  from pypdf import PdfReader

  cache = pathlib.Path(tempfile.gettempdir()) / 'adk_arxiv_papers'
  cache.mkdir(parents=True, exist_ok=True)
  pdf_path = cache / (arxiv_id.replace('/', '_') + '.pdf')
  if not pdf_path.exists():
    url = 'https://arxiv.org/pdf/' + arxiv_id
    req = urllib.request.Request(url, headers={'User-Agent': 'adk-test/1.0'})
    pdf_path.write_bytes(urllib.request.urlopen(req, timeout=60).read())  # noqa: S310
  reader = PdfReader(str(pdf_path))
  return ''.join((page.extract_text() or '') for page in reader.pages)


def _prompt(doc: str) -> str:
  return (
      f'PAPER:\n{doc}\n\n----------------------------------------\n'
      'You just read the paper above.\n\n' + _DIRECTIVE
  )


def _topic(payload: Any) -> str:
  if isinstance(payload, dict):
    return payload.get('arguments', {}).get('topic', '')
  return ''


def _build(doc: str, *, speculative: bool) -> tuple[Workflow, dict[str, Any]]:
  stats: dict[str, Any] = {'dispatches': 0}

  async def lookup(node_input: Any):
    # The downstream dependency: expensive work keyed on the directive's topic.
    stats['dispatches'] += 1
    try:
      await asyncio.sleep(_HANDLER_SECONDS)
    except asyncio.CancelledError:
      raise
    yield Event(output={'topic': _topic(node_input), 'handled': True})

  agent = Agent(
      name='reader',
      model=MODEL,
      # Callable instruction bypasses {var} templating so raw paper braces are
      # sent verbatim -- no escaping, no truncation.
      instruction=lambda _ctx, _p=_prompt(doc): _p,
  )
  node = SpeculativeRouterNode(
      name='speculative_read',
      agent=agent,
      target=lookup,
      # A: never speculate (sequential). B: speculate (overlap).
      should_speculate=(lambda p: bool(_topic(p))) if speculative else (
          lambda p: False
      ),
      same=lambda a, b: _topic(a) == _topic(b),
      timeout=120,
  )
  return Workflow(name='spec_wf', edges=[('START', node)]), stats


async def _run(wf: Workflow) -> tuple[float, Any]:
  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='go')], role='user')
  output = None
  start = time.perf_counter()
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    if isinstance(event, Event) and event.output is not None:
      output = event.output
  return time.perf_counter() - start, output


@pytest.mark.asyncio
@pytest.mark.parametrize('llm_backend', ['VERTEX'], indirect=True)
async def test_speculative_dispatch_overlaps_generation(llm_backend):
  doc = _download_paper_text(_ARXIV_ID)

  a_wf, _a_stats = _build(doc, speculative=False)
  a_time, a_out = await _run(a_wf)

  b_wf, b_stats = _build(doc, speculative=True)
  b_time, b_out = await _run(b_wf)

  saved = a_time - b_time
  print(
      f'\n[speculative dispatch] model={MODEL} doc_chars={len(doc)} '
      '(whole paper, no chunking)\n'
      f'  handler work (overlappable):   {_HANDLER_SECONDS:6.2f}s\n'
      f'  A  sequential (gen, then run): {a_time:6.2f}s   topic={_topic_out(a_out)!r}\n'
      f'  B  speculative (run overlaps): {b_time:6.2f}s   topic={_topic_out(b_out)!r}\n'
      f'  wall-clock saved by overlap:   {saved:6.2f}s'
      f'  (dispatches={b_stats["dispatches"]})\n'
  )

  # Both strategies produce the handled result.
  assert isinstance(a_out, dict) and a_out.get('handled')
  assert isinstance(b_out, dict) and b_out.get('handled')
  # Speculation dispatched the handler early (at least once).
  assert b_stats['dispatches'] >= 1
  # Overlap hides most of the handler's cost behind generation.
  assert b_time < a_time
  assert saved > _HANDLER_SECONDS * 0.5


def _topic_out(out: Any) -> str:
  return out.get('topic', '') if isinstance(out, dict) else ''
