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

"""Real-LLM: two chained model calls, where speculation overlaps the second.

This is the *real* use case for ``SpeculativeRouterNode`` -- no sleeps, no
simulated work. A DAG of two LLM calls:

  planner (LLM)  --topic-->  worker (LLM)

The ``planner`` reads a whole arXiv paper (NO chunking) and is told to FIRST emit
a routing directive naming the topic to dig into:

    TOOL_CALL: {"name": "explain", "arguments": {"topic": "<subject>"}}

...THEN write a long rationale. The ``worker`` is a second, real LLM call that
writes the detailed answer for that topic. It only depends on the *directive*,
which streams out at the very start -- long before the planner finishes its
rationale.

Honest framing (no "just plan" escape)
--------------------------------------
The planner's rationale is a **required deliverable**: the workflow returns
``{"plan": <full planner rationale>, "answer": <worker explanation>}`` via the
node's ``combine`` hook, and the test asserts BOTH are present and substantial
for *both* strategies. So a baseline cannot cheat by telling the planner to emit
only the directive and stop -- it is contractually obligated to produce the whole
rationale, exactly like the speculative path. The only difference between A and B
is *when* the worker runs:

  A. sequential   -- produce the whole rationale, THEN call the worker.
  B. speculative  -- the instant the directive streams in, start the worker so
                     its generation overlaps the (still-required) rationale tail;
                     verify against the finalized directive.

Both emit identical deliverables; B just hides the worker's genuine multi-second
generation behind work the caller already demanded. The assertion ties the saving
to the worker's *measured* solo runtime -- not a magic constant -- so it reflects
real overlap.

    ADK_TEST_MODEL=gemini-3.5-flash-lite \\
      uv run pytest -s -p no:cacheprovider \\
      tests/integration/test_speculative_router_chained_llm.py

Requires Vertex (GOOGLE_CLOUD_PROJECT via ADC), ``pypdf`` and arXiv; skips
otherwise.
"""

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
from google.adk.workflow import make_marker_extractor
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
_ARXIV_ID = '1706.03762'  # Attention Is All You Need

_DIRECTIVE = (
    'FIRST output EXACTLY one line, nothing else on it:\n'
    'TOOL_CALL: {"name": "explain", "arguments": {"topic":'
    ' "<the single core technical method of this paper, 2-4 words>"}}\n'
    'THEN write a long, detailed rationale for your choice (at least 300'
    ' words).'
)

_WORKER_INSTRUCTION = (
    'You are given the name of a technical method from a research paper. Write'
    ' a detailed, ~200 word technical explanation of that method: what it is,'
    ' how it works, and why it matters.'
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


def _planner_prompt(doc: str) -> str:
  return (
      f'PAPER:\n{doc}\n\n----------------------------------------\n'
      'You just read the paper above.\n\n' + _DIRECTIVE
  )


def _extract_topic(text: str) -> Any:
  """Returns just the directive's ``topic`` string (or None), repairing partials."""
  obj = make_marker_extractor('TOOL_CALL:')(text)
  if not isinstance(obj, dict):
    return None
  topic = obj.get('arguments', {}).get('topic')
  return topic or None


def _worker_agent() -> Agent:
  return Agent(name='worker', model=MODEL, instruction=_WORKER_INSTRUCTION)


def _build_chain(doc: str, *, speculative: bool) -> Workflow:
  planner = Agent(
      name='planner',
      model=MODEL,
      instruction=lambda _ctx, _p=_planner_prompt(doc): _p,
  )
  node = SpeculativeRouterNode(
      name='plan_then_explain',
      agent=planner,
      target=_worker_agent(),
      extract=_extract_topic,
      should_speculate=(
          (lambda topic: bool(topic) and len(topic) >= 4)
          if speculative
          else (lambda _topic: False)
      ),
      same=lambda a, b: (a or '').strip().lower() == (b or '').strip().lower(),
      # The planner's full rationale is a required deliverable: both strategies
      # must return it alongside the worker's answer, so B has no "just plan and
      # stop" shortcut -- it produces the exact same artifact, only faster.
      combine=lambda plan, answer: {'plan': plan, 'answer': _text(answer)},
      timeout=120,
  )
  return Workflow(name='chain', edges=[('START', node)])


def _worker_only(topic: str) -> Workflow:
  def give_topic(node_input: Any):
    yield Event(output=topic)

  return Workflow(name='worker_only', edges=[('START', give_topic, _worker_agent())])


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


def _text(out: Any) -> str:
  if isinstance(out, str):
    return out
  if isinstance(out, types.Content) and out.parts:
    return ''.join(p.text or '' for p in out.parts)
  return str(out) if out is not None else ''


def _plan(out: Any) -> str:
  """The planner's full rationale from the ``{plan, answer}`` deliverable."""
  return out.get('plan', '') if isinstance(out, dict) else ''


def _answer(out: Any) -> str:
  """The worker's explanation from the ``{plan, answer}`` deliverable."""
  return out.get('answer', '') if isinstance(out, dict) else _text(out)


@pytest.mark.asyncio
@pytest.mark.parametrize('llm_backend', ['VERTEX'], indirect=True)
async def test_speculative_chain_overlaps_worker_llm(llm_backend):
  doc = _download_paper_text(_ARXIV_ID)

  # Measure the worker's solo runtime so the saving is tied to real work.
  w_time, _w_out = await _run(_worker_only('the self-attention mechanism'))

  a_time, a_out = await _run(_build_chain(doc, speculative=False))
  b_time, b_out = await _run(_build_chain(doc, speculative=True))

  saved = a_time - b_time
  print(
      f'\n[speculative chain] model={MODEL} doc_chars={len(doc)} '
      '(two chained LLM calls, whole paper, no chunking)\n'
      f'  worker LLM alone:              {w_time:6.2f}s\n'
      f'  A  sequential (plan -> work):  {a_time:6.2f}s'
      f'   plan={len(_plan(a_out)):5d} chars, answer={len(_answer(a_out)):4d}'
      ' chars\n'
      f'  B  speculative (work overlaps):{b_time:6.2f}s'
      f'   plan={len(_plan(b_out)):5d} chars, answer={len(_answer(b_out)):4d}'
      ' chars\n'
      f'  wall-clock saved by overlap:   {saved:6.2f}s'
      f'   ({100 * saved / w_time:4.0f}% of the worker call hidden)\n'
  )

  # Honest deliverable: BOTH strategies must return the planner's full rationale
  # AND the worker's answer. The rationale is required, so a baseline can't cheat
  # by "just planning" and skipping the tail -- both produce the same artifact.
  for out in (a_out, b_out):
    assert len(_plan(out)) > 200, 'planner rationale is a required deliverable'
    assert len(_answer(out)) > 80, 'worker answer is a required deliverable'
  # Speculation overlaps the worker with the (still-required) planner tail ->
  # faster while producing the identical two-part deliverable...
  assert b_time < a_time
  # ...by a meaningful fraction of the worker's *measured* real runtime (not a
  # magic constant). Overlap can't hide more than the worker takes; requiring
  # >=30% keeps this robust to run-to-run LLM variance.
  assert saved > 0.30 * w_time
