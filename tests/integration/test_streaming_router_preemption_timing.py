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

"""Real-LLM: an ADK graph reads N whole documents in parallel and answers a
question, comparing plain SSE streaming against SSE + mid-stream preemption.

NO chunking. Each document is handed to Gemini whole, in a single streaming
API call. The only difference between the two strategies is what we do with the
SSE token stream:

  A. read + answer                -- stream the model's answer to completion.
  B. read + answer + preemption   -- the StreamingRouterNode monitor watches the
                                     SSE tokens and cancels the stream the moment
                                     the verdict has streamed in, so we never
                                     generate the long summary that follows.

Five real arXiv papers are read in parallel; only "Attention Is All You Need" is
a CS AI/ML paper. Both strategies must classify all five correctly; B must be
faster because it stops generating once the answer is known.

    ADK_TEST_MODEL=gemini-3.5-flash-lite \\
      uv run pytest -s -p no:cacheprovider \\
      tests/integration/test_streaming_router_preemption_timing.py

Requires Vertex (GOOGLE_CLOUD_PROJECT via ADC), ``pypdf`` and network access to
arXiv; skips otherwise.
"""

import os
import pathlib
import tempfile
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
from google.adk.workflow import JoinNode
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

QUERY = (
    'Is this paper about artificial intelligence, machine learning, or neural'
    ' networks (i.e. a computer-science AI paper)?'
)

# The relevant paper is first; the other four are real but not AI papers.
_PAPERS: list[tuple[str, str, bool]] = [
    ('1706.03762', 'Attention Is All You Need', True),
    ('1602.03837', 'Observation of Gravitational Waves (GW150914)', False),
    ('1207.7214', 'Observation of the Higgs boson (ATLAS)', False),
    ('astro-ph/9805201', 'Accelerating universe / dark energy', False),
    ('math/0211159', 'The entropy formula for the Ricci flow', False),
]

_VERDICT_MARKER = 'VERDICT:'
_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / 'adk_arxiv_papers'

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


def _download_paper_text(arxiv_id: str) -> str:
  """Downloads an arXiv PDF (cached) and returns its full extracted text."""
  from pypdf import PdfReader

  _CACHE_DIR.mkdir(parents=True, exist_ok=True)
  pdf_path = _CACHE_DIR / (arxiv_id.replace('/', '_') + '.pdf')
  if not pdf_path.exists():
    url = 'https://arxiv.org/pdf/' + arxiv_id
    req = urllib.request.Request(url, headers={'User-Agent': 'adk-test/1.0'})
    data = urllib.request.urlopen(req, timeout=60).read()  # noqa: S310
    pdf_path.write_bytes(data)
  reader = PdfReader(str(pdf_path))
  return ''.join((page.extract_text() or '') for page in reader.pages)


def _load_documents() -> list[str]:
  """Fetches every paper as one whole-document string; skips if unavailable."""
  pytest.importorskip('pypdf', reason='PDF extraction needs pypdf.')
  docs: list[str] = []
  for arxiv_id, label, _ in _PAPERS:
    try:
      text = _download_paper_text(arxiv_id)
    except (urllib.error.URLError, TimeoutError) as e:
      pytest.skip(f'Could not fetch arXiv:{arxiv_id} ({label}): {e}')
    docs.append(text)  # whole document, no truncation
  return docs


def _verdict_after_marker(text: str) -> Optional[str]:
  """Returns the text after ``VERDICT:`` once that line has fully streamed."""
  idx = text.upper().find(_VERDICT_MARKER)
  if idx == -1:
    return None
  after = text[idx + len(_VERDICT_MARKER) :]
  line, sep, _ = after.partition('\n')
  if not sep:
    return None  # verdict line still streaming
  return line.strip()


def _decide(view: StreamView) -> Optional[StreamDecision]:
  """Preemption verdict: fires once the VERDICT line has streamed in."""
  verdict = _verdict_after_marker(view.text)
  if verdict is None:
    return None
  upper = verdict.upper()
  if upper.startswith('IRRELEVANT'):
    return StreamDecision(output={'relevant': False, 'verdict': 'IRRELEVANT'})
  if upper.startswith('RELEVANT'):
    return StreamDecision(output={'relevant': True, 'verdict': verdict})
  return None


def _reader_prompt(doc: str) -> str:
  """The whole-document, single-call prompt handed to one reader."""
  return (
      # Whole document first, then the question -- one prompt, one call.
      f'PAPER:\n{doc}\n\n----------------------------------------\nYou'
      ' just read the paper above. Judge ONLY from it.\n\nQUESTION:'
      f' {QUERY}\n\nFIRST output a line beginning "VERDICT:" that is'
      ' exactly one of:\n  VERDICT: IRRELEVANT           (NOT a'
      ' computer-science AI/ML/neural-networks paper -- e.g. physics,'
      ' astronomy, or pure mathematics, even if it uses the word'
      ' "model"), or\n  VERDICT: RELEVANT - <topic>   (a'
      ' computer-science paper about AI, machine learning, or neural'
      ' networks -- name its subject).\nTHEN write a long, detailed,'
      ' multi-paragraph summary of the paper (at least 300 words).'
  )


def _build_workflow(
    docs: list[str],
    *,
    preempt: bool,
    sink: dict[str, Any],
    gen: dict[int, str],
) -> Workflow:
  readers = []
  for i, doc in enumerate(docs):
    prompt = _reader_prompt(doc)
    reader = Agent(
        name=f'reader_{i}',
        model=MODEL,
        # A callable (provider) instruction bypasses {var} state-injection, so
        # raw LaTeX/math braces in the papers are sent verbatim -- no escaping,
        # no truncation.
        instruction=lambda _ctx, _p=prompt: _p,
    )

    def monitor(view: StreamView, _i: int = i) -> Optional[StreamDecision]:
      # Capture the latest streamed output text so we can count the tokens the
      # model actually generated under each strategy.
      gen[_i] = view.text
      return _decide(view) if preempt else None

    readers.append(
        StreamingRouterNode(
            name=f'reader_{i}',
            agent=reader,
            monitor=monitor,
            forward_partials=False,
            timeout=180,
        )
    )

  join = JoinNode(name='join_sources')

  async def collect(node_input: dict[str, Any]):
    sink['fan_in'] = node_input
    yield Event(message='done')

  return Workflow(
      name='parallel_read_answer',
      edges=[('START', tuple(readers), join, collect)],
  )


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


def _status_and_verdict(value: Any) -> tuple[str, str]:
  """Normalises a reader's output (dict from B, raw text from A) to a verdict."""
  if isinstance(value, dict):
    return (
        'RELEVANT' if value.get('relevant') else 'IRRELEVANT',
        str(value.get('verdict', '')),
    )
  verdict = _verdict_after_marker(str(value)) or ''
  status = (
      'RELEVANT' if verdict.upper().startswith('RELEVANT') else 'IRRELEVANT'
  )
  return status, verdict


def _value_for_index(fan_in: dict[str, Any], index: int) -> Any:
  for key, value in fan_in.items():
    digits = ''.join(ch for ch in key if ch.isdigit())
    if digits and int(digits) == index:
      return value
  raise KeyError(f'No fan-in entry for reader index {index}: {list(fan_in)}')


def _generated_text(
    index: int, fan_in: dict[str, Any], gen: dict[int, str]
) -> str:
  """The text a reader actually generated: full answer (A) or up-to-cut (B)."""
  value = _value_for_index(fan_in, index)
  if isinstance(value, str) and value.strip():
    return value  # A: the whole streamed answer is the node output
  return gen.get(index, '')  # B: streamed text captured up to preemption


def _output_tokens(fan_in: dict[str, Any], gen: dict[int, str]) -> list[int]:
  return [
      _count_tokens(_generated_text(i, fan_in, gen))
      for i in range(len(_PAPERS))
  ]


def _print_qa(
    title: str, fan_in: dict[str, Any], out_tokens: list[int]
) -> None:
  print(f'\n===== {title} =====')
  print(f'Q: {QUERY}\n')
  for i, (arxiv_id, label, _relevant) in enumerate(_PAPERS):
    status, verdict = _status_and_verdict(_value_for_index(fan_in, i))
    mark = 'RELEVANT  ' if status == 'RELEVANT' else 'irrelevant'
    print(f'  [{mark}] {out_tokens[i]:5d} out-tok  {label} (arXiv:{arxiv_id})')
    print(f'            -> {verdict or status}')


@pytest.mark.asyncio
@pytest.mark.parametrize('llm_backend', ['VERTEX'], indirect=True)
async def test_sse_preemption_beats_full_generation(llm_backend):
  docs = _load_documents()

  # A: read + answer, stream to completion.
  a_sink: dict[str, Any] = {}
  a_gen: dict[int, str] = {}
  a_time = await _run(
      _build_workflow(docs, preempt=False, sink=a_sink, gen=a_gen)
  )

  # B: read + answer, SSE + preemption (cut once the verdict streams in).
  b_sink: dict[str, Any] = {}
  b_gen: dict[int, str] = {}
  b_time = await _run(
      _build_workflow(docs, preempt=True, sink=b_sink, gen=b_gen)
  )

  # Real token counts. Input is identical for A and B (same whole-doc prompts);
  # the difference is entirely in generated output tokens.
  input_tokens = sum(_count_tokens(_reader_prompt(doc)) for doc in docs)
  a_out = _output_tokens(a_sink['fan_in'], a_gen)
  b_out = _output_tokens(b_sink['fan_in'], b_gen)
  a_out_total, b_out_total = sum(a_out), sum(b_out)

  speedup = a_time / b_time if b_time else float('inf')
  tok_ratio = a_out_total / b_out_total if b_out_total else float('inf')

  # Cost (gemini-3.5-flash-lite pricing). Input is identical for A and B; the
  # only difference is output tokens. With context caching the input read is 10x
  # cheaper, so preemption's output savings dominate the total.
  a_cost = input_tokens * _PRICE_IN + a_out_total * _PRICE_OUT
  b_cost = input_tokens * _PRICE_IN + b_out_total * _PRICE_OUT
  a_cost_cached = input_tokens * _PRICE_IN_CACHED + a_out_total * _PRICE_OUT
  b_cost_cached = input_tokens * _PRICE_IN_CACHED + b_out_total * _PRICE_OUT
  save = 100 * (1 - b_cost / a_cost) if a_cost else 0.0
  save_cached = (
      100 * (1 - b_cost_cached / a_cost_cached) if a_cost_cached else 0
  )

  print(
      f'\n[SSE preemption] model={MODEL} docs={len(_PAPERS)} (whole docs, no'
      ' chunking)\n'
      f'  input tokens (both):         {input_tokens:6d}\n'
      f'  A  read+answer:              {a_time:6.2f}s   '
      f'{a_out_total:6d} out-tok\n'
      f'  B  read+answer+preemption:   {b_time:6.2f}s   '
      f'{b_out_total:6d} out-tok\n'
      f'  speedup:                     {speedup:5.2f}x\n'
      f'  output tokens saved:         {a_out_total - b_out_total:6d}'
      f'   (B uses {tok_ratio:4.2f}x fewer)\n'
      '  cost @ $0.30/$2.50 per 1M (in/out):\n'
      f'     A ${a_cost:.4f}   B ${b_cost:.4f}   (B saves {save:4.1f}%)\n'
      '  cost w/ context caching (in @ $0.03/1M):\n'
      f'     A ${a_cost_cached:.4f}   B ${b_cost_cached:.4f}'
      f'   (B saves {save_cached:4.1f}%)\n'
  )
  _print_qa('A: read + answer (stream to completion)', a_sink['fan_in'], a_out)
  _print_qa('B: read + answer + SSE preemption', b_sink['fan_in'], b_out)

  # Both strategies must classify all five papers correctly.
  for fan_in in (a_sink['fan_in'], b_sink['fan_in']):
    assert _status_and_verdict(_value_for_index(fan_in, 0))[0] == 'RELEVANT'
    for idx in range(1, len(_PAPERS)):
      assert _status_and_verdict(_value_for_index(fan_in, idx))[0] == (
          'IRRELEVANT'
      )

  # Preemption stops generating the long summaries -> fewer output tokens...
  assert b_out_total < a_out_total
  # ...and it is faster.
  assert b_time < a_time
