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

"""Real-LLM demo: StreamingRouterNode reads a whole Tesla 10-K and preempts.

This is the arXiv timing test's big-document sibling. Instead of five small
papers it hands Gemini a single, enormous real-world filing -- the latest Tesla
annual report (Form 10-K), pulled live from SEC EDGAR (~100k+ input tokens) --
in one whole-document streaming call. NO chunking.

The point it makes on a huge doc: the input is paid for once (prefill), so the
two strategies differ only in how much they *generate*:

  A. read + answer                -- stream the model's answer to completion.
  B. read + answer + preemption   -- the StreamingRouterNode monitor watches the
                                     SSE tokens and cancels the stream the moment
                                     the "VERDICT:" line has streamed in, so the
                                     long analysis that follows is never decoded.

B is dramatically faster (no long decode) and, once the static filing is context
-cached, dramatically cheaper -- the output cut becomes the whole bill.

    ADK_TEST_MODEL=gemini-3.5-flash-lite \\
      uv run pytest -s -p no:cacheprovider \\
      tests/integration/test_streaming_router_tsla_10k.py

Requires Vertex (GOOGLE_CLOUD_PROJECT via ADC) and network access to SEC EDGAR;
skips otherwise.
"""

import os
import pathlib
import re
import time
from typing import Any
from typing import Optional
import urllib.error
import urllib.request

from dotenv import load_dotenv
from google import genai
from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import StreamDecision
from google.adk.workflow import StreamingRouterNode
from google.adk.workflow import StreamView
from google.genai import types
import pytest

# Load the repo-root .env (Vertex project/location/model live there).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / '.env', override=False)

# Vertex backend (ADC auth, no API key). Skip unless a project is configured.
pytestmark = pytest.mark.skipif(
    not os.environ.get('GOOGLE_CLOUD_PROJECT'),
    reason='Real-LLM timing test requires Vertex (GOOGLE_CLOUD_PROJECT).',
)

# gemini-3.5-flash-lite by default (verified on Vertex). Override w/ ADK_TEST_MODEL.
MODEL = os.environ.get('ADK_TEST_MODEL', 'gemini-3.5-flash-lite')

# Published gemini-3.5-flash-lite pricing (USD per 1M tokens). Output is 8.3x
# input; the cached-input rate is 10x cheaper than a fresh read.
_PRICE_IN = 0.30 / 1e6
_PRICE_OUT = 2.50 / 1e6
_PRICE_IN_CACHED = 0.03 / 1e6

# Tesla, Inc. central index key on SEC EDGAR.
_TSLA_CIK = '0001318605'
# SEC requires a descriptive User-Agent with contact info or it returns 403.
_SEC_UA = {'User-Agent': 'google-adk integration-test contact@example.com'}

QUERY = (
    'What are the three most significant risk factors Tesla identifies in this'
    ' filing?'
)
_VERDICT_MARKER = 'VERDICT:'

_CLIENT: Optional[genai.Client] = None


def _client() -> genai.Client:
  """A process-wide Vertex genai client, used only for token counting."""
  global _CLIENT
  if _CLIENT is None:
    _CLIENT = genai.Client(
        vertexai=True,
        project=os.environ['GOOGLE_CLOUD_PROJECT'],
        location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'global'),
    )
  return _CLIENT


def _count_tokens(text: str) -> int:
  """Real token count for ``text`` via the model's tokenizer (0 if empty)."""
  if not text.strip():
    return 0
  return _client().models.count_tokens(model=MODEL, contents=text).total_tokens


def _get(url: str) -> bytes:
  req = urllib.request.Request(url, headers=_SEC_UA)
  return urllib.request.urlopen(req, timeout=60).read()  # noqa: S310


def _html_to_text(html: str) -> str:
  """Crudely strips a 10-K .htm down to readable prose (no external deps)."""
  html = re.sub(r'(?is)<(script|style).*?</\1>', ' ', html)
  html = re.sub(r'(?is)<br\s*/?>', '\n', html)
  html = re.sub(r'(?is)</(p|div|tr|h[1-6]|li)>', '\n', html)
  text = re.sub(r'(?is)<[^>]+>', ' ', html)
  import html as _htmllib

  text = _htmllib.unescape(text)
  text = re.sub(r'[ \t]+', ' ', text)
  return re.sub(r'\n\s*\n+', '\n\n', text).strip()


def _load_tsla_10k() -> str:
  """Fetches the latest Tesla 10-K primary document as whole text (no cache)."""
  import json

  try:
    subs = json.loads(_get(f'https://data.sec.gov/submissions/CIK{_TSLA_CIK}.json'))
    recent = subs['filings']['recent']
    idx = next(i for i, f in enumerate(recent['form']) if f == '10-K')
    accession = recent['accessionNumber'][idx].replace('-', '')
    document = recent['primaryDocument'][idx]
    url = (
        f'https://www.sec.gov/Archives/edgar/data/{int(_TSLA_CIK)}/'
        f'{accession}/{document}'
    )
    return _html_to_text(_get(url).decode('utf-8', 'ignore'))
  except (urllib.error.URLError, TimeoutError, StopIteration, KeyError) as e:
    pytest.skip(f'Could not fetch Tesla 10-K from SEC EDGAR: {e}')


def _verdict_after_marker(text: str) -> Optional[str]:
  """Returns the text after ``VERDICT:`` once that line has fully streamed."""
  idx = text.upper().find(_VERDICT_MARKER)
  if idx == -1:
    return None
  line, sep, _ = text[idx + len(_VERDICT_MARKER) :].partition('\n')
  return line.strip() if sep else None


def _reader_prompt(doc: str) -> str:
  """The whole-document, single-call prompt handed to the reader."""
  return (
      # Whole filing first, then the question -- one prompt, one call.
      f'FILING (Tesla annual report / Form 10-K):\n{doc}\n\n'
      '----------------------------------------\n'
      'You just read the filing above. Answer ONLY from it.\n\n'
      f'QUESTION: {QUERY}\n\n'
      'FIRST output a line beginning "VERDICT:" that names the top three risks'
      ' in one sentence.\nTHEN write a detailed multi-paragraph analysis (at'
      ' least 400 words).'
  )


def _build_workflow(
    doc: str,
    *,
    preempt: bool,
    sink: dict[str, Any],
    gen: dict[str, str],
) -> Workflow:
  prompt = _reader_prompt(doc)
  # A callable (provider) instruction bypasses {var} state-injection, so raw
  # braces in the filing are sent verbatim -- no escaping, no truncation.
  reader = Agent(
      name='reader', model=MODEL, instruction=lambda _ctx, _p=prompt: _p
  )

  def monitor(view: StreamView) -> Optional[StreamDecision]:
    # Capture the latest streamed text so we can count generated tokens.
    gen['text'] = view.text
    if not preempt:
      return None
    verdict = _verdict_after_marker(view.text)
    return StreamDecision(output={'answer': verdict}) if verdict else None

  node = StreamingRouterNode(
      name='reader',
      agent=reader,
      monitor=monitor,
      forward_partials=False,
      timeout=300,
  )

  async def collect(node_input: Any):
    sink['out'] = node_input
    yield Event(message='done')

  return Workflow(name='tsla_10k', edges=[('START', node, collect)])


async def _run(wf: Workflow) -> float:
  ss = InMemorySessionService()
  runner = Runner(app_name=wf.name, node=wf, session_service=ss)
  session = await ss.create_session(app_name=wf.name, user_id='u')
  msg = types.Content(parts=[types.Part(text='go')], role='user')
  start = time.perf_counter()
  async for _ in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    pass
  return time.perf_counter() - start


def _generated_text(sink: dict[str, Any], gen: dict[str, str]) -> str:
  """The text the reader actually generated: full answer (A) or up-to-cut (B)."""
  value = sink.get('out')
  if isinstance(value, str) and value.strip():
    return value  # A: the whole streamed answer is the node output
  return gen.get('text', '')  # B: streamed text captured up to preemption


def _verdict_of(sink: dict[str, Any], gen: dict[str, str]) -> str:
  value = sink.get('out')
  if isinstance(value, dict):
    return str(value.get('answer') or '')
  return _verdict_after_marker(_generated_text(sink, gen)) or ''


@pytest.mark.asyncio
@pytest.mark.parametrize('llm_backend', ['VERTEX'], indirect=True)
async def test_sse_preemption_on_tsla_10k(llm_backend):
  doc = _load_tsla_10k()

  # A: read + answer, stream to completion.
  a_sink: dict[str, Any] = {}
  a_gen: dict[str, str] = {}
  a_time = await _run(_build_workflow(doc, preempt=False, sink=a_sink, gen=a_gen))

  # B: read + answer, SSE + preemption (cut once the verdict streams in).
  b_sink: dict[str, Any] = {}
  b_gen: dict[str, str] = {}
  b_time = await _run(_build_workflow(doc, preempt=True, sink=b_sink, gen=b_gen))

  # Real token counts. Input is identical for A and B (same whole-doc prompt);
  # the difference is entirely in generated output tokens.
  input_tokens = _count_tokens(_reader_prompt(doc))
  a_out = _count_tokens(_generated_text(a_sink, a_gen))
  b_out = _count_tokens(_generated_text(b_sink, b_gen))

  speedup = a_time / b_time if b_time else float('inf')
  tok_ratio = a_out / b_out if b_out else float('inf')

  # Cost (gemini-3.5-flash-lite pricing). Input is identical for A and B; only
  # output differs. With context caching the huge filing is 10x cheaper to read,
  # so preemption's output savings dominate the total.
  a_cost = input_tokens * _PRICE_IN + a_out * _PRICE_OUT
  b_cost = input_tokens * _PRICE_IN + b_out * _PRICE_OUT
  a_cost_cached = input_tokens * _PRICE_IN_CACHED + a_out * _PRICE_OUT
  b_cost_cached = input_tokens * _PRICE_IN_CACHED + b_out * _PRICE_OUT
  save = 100 * (1 - b_cost / a_cost) if a_cost else 0.0
  save_cached = 100 * (1 - b_cost_cached / a_cost_cached) if a_cost_cached else 0

  print(
      f'\n[SSE preemption / TSLA 10-K] model={MODEL} doc_chars={len(doc)}'
      ' (whole filing, no chunking)\n'
      f'  input tokens (both):         {input_tokens:6d}\n'
      f'  A  read+answer:              {a_time:6.2f}s   {a_out:6d} out-tok\n'
      f'  B  read+answer+preemption:   {b_time:6.2f}s   {b_out:6d} out-tok\n'
      f'  speedup:                     {speedup:5.2f}x\n'
      f'  output tokens saved:         {a_out - b_out:6d}'
      f'   (B uses {tok_ratio:4.2f}x fewer)\n'
      '  cost @ $0.30/$2.50 per 1M (in/out):\n'
      f'     A ${a_cost:.4f}   B ${b_cost:.4f}   (B saves {save:4.1f}%)\n'
      '  cost w/ context caching (in @ $0.03/1M):\n'
      f'     A ${a_cost_cached:.4f}   B ${b_cost_cached:.4f}'
      f'   (B saves {save_cached:4.1f}%)\n'
  )
  print(f'Q: {QUERY}')
  print(f'A verdict: {_verdict_of(a_sink, a_gen)}')
  print(f'B verdict: {_verdict_of(b_sink, b_gen)}')

  # The model must actually answer from the filing under both strategies.
  assert _verdict_of(a_sink, a_gen)
  assert _verdict_of(b_sink, b_gen)
  # Preemption stops generating the long analysis -> fewer output tokens...
  assert b_out < a_out
  # ...and it is faster.
  assert b_time < a_time
