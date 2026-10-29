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

"""A built-in argument matcher that asks a model whether values match.

`SemanticArgMatcher` is registered under the `semantic` type, so a
`ToolArgsMatcher` config can select it with `{"type": "semantic"}`. It lives
in its own module so that `_tool_args_matcher` stays free of model-layer
imports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import ClassVar
from typing import Literal

from google.genai import types as genai_types
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ...models.base_llm import BaseLlm
from ...models.llm_request import LlmRequest
from ...models.registry import LLMRegistry
from ...utils.context_utils import Aclosing
from ._tool_args_matcher import ArgMatchResult
from ._tool_args_matcher import BaseArgMatcher
from ._tool_args_matcher import register_arg_matcher

logger = logging.getLogger('google_adk.' + __name__)


@register_arg_matcher
class SemanticArgMatcher(BaseArgMatcher):
  """Matches values that mean the same thing, as judged by a model.

  Equal values match with score 1.0 without a model call. Otherwise the
  matcher asks `model` whether the two values mean the same thing under
  `criteria`. The model returns a verdict, which sets `matched`, a score from
  0.0 to 1.0 for how closely the values match, which sets `score`, and a short
  reason. The result is remembered for the same tool, argument and values. A
  response with neither a verdict nor a score counts as a mismatch and is not
  remembered. An error from the model propagates to the caller.
  """

  # Constants are ClassVars: Pydantic turns any other class attribute whose
  # name starts with an underscore into a private instance attribute.

  _DEFAULT_MODEL: ClassVar[str] = 'gemini-2.5-flash'

  # Comparing two values is a judgement, not a generation task. The
  # temperature is pinned to 0 so the same comparison gives the same answer
  # from run to run, and the thinking budget is kept small because every
  # compared argument costs a model call. The budget is small rather than 0
  # because some models reject a zero budget.
  _THINKING_BUDGET: ClassVar[int] = 512

  _DEFAULT_CRITERIA: ClassVar[str] = (
      'The values match if they mean the same thing or express the same'
      ' intent, even if they are phrased differently, use different casing or'
      ' word order, or one uses a synonym or abbreviation of the other. They'
      ' do not match if they refer to different entities, quantities, dates,'
      ' or intents.'
  )

  # Bounds the results each matcher remembers, so a long evaluation run cannot
  # grow the cache without limit.
  _CACHE_MAX_ENTRIES: ClassVar[int] = 4096

  # A score at or above this counts as a match when the model gives a score
  # but no verdict.
  _SCORE_MATCH_THRESHOLD: ClassVar[float] = 0.5

  # Long enough for a short judgement with retries, short enough that a hung
  # call does not stall the tool call that is waiting on it.
  _DEFAULT_TIMEOUT_SECONDS: ClassVar[float] = 60.0

  type: Literal['semantic'] = 'semantic'
  is_deterministic: ClassVar[bool] = False

  model: str | None = None
  """The name of the model that compares the values.

  When not set, Environment Simulation uses its `simulation_model`, and other
  callers use gemini-2.5-flash.
  """

  criteria: str | None = None
  """What counts as a match, in plain language.

  Replaces the default guidance, which accepts rephrasings, synonyms and
  differences in casing or word order that keep the same meaning.
  """

  generate_content_config: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: (
          SemanticArgMatcher._default_generate_content_config()
      )
  )
  """The content generation configuration for the model call.

  Defaults to temperature 0, JSON output and a small thinking budget.
  """

  timeout_seconds: float | None = Field(default=_DEFAULT_TIMEOUT_SECONDS, gt=0)
  """How long to wait for the model before giving up, in seconds.

  When the model takes longer, `match` raises `TimeoutError` and the result is
  not remembered. None waits indefinitely.
  """

  _llm: BaseLlm | None = PrivateAttr(default=None)
  _verdicts: dict[tuple[str, str, str, str], ArgMatchResult] = PrivateAttr(
      default_factory=dict
  )

  @override
  async def match(
      self,
      *,
      tool_name: str,
      arg_name: str,
      expected: object,
      actual: object,
  ) -> ArgMatchResult:
    if expected == actual:
      return ArgMatchResult(matched=True, reason='The values are equal.')

    expected_json = self._to_prompt_json(expected)
    actual_json = self._to_prompt_json(actual)
    cache_key = (tool_name, arg_name, expected_json, actual_json)
    cached = self._verdicts.get(cache_key)
    if cached is not None:
      return cached

    response_text = await self._ask_model(
        self._build_prompt(tool_name, arg_name, expected_json, actual_json)
    )
    result = self._parse_judgement(response_text)
    if result is None:
      logger.warning(
          'Semantic matcher got no verdict or score for argument %s of tool'
          ' %s.',
          arg_name,
          tool_name,
      )
      return ArgMatchResult(
          matched=False,
          score=0.0,
          reason=(
              'The model response had no verdict or score:'
              f' {response_text.strip()!r}.'
          ),
      )

    if len(self._verdicts) < self._CACHE_MAX_ENTRIES:
      self._verdicts[cache_key] = result
    return result

  @classmethod
  def _default_generate_content_config(
      cls,
  ) -> genai_types.GenerateContentConfig:
    return genai_types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type='application/json',
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=cls._THINKING_BUDGET,
        ),
    )

  @staticmethod
  def _to_prompt_json(value: object) -> str:
    """Renders a value for the prompt and the cache key, stable across runs."""
    try:
      return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
      return repr(value)

  @staticmethod
  def _parse_verdict(text: str) -> bool | None:
    """Reads a YES or NO verdict from free text.

    Returns:
      True for YES, False for NO, or None when the text says neither or both.
    """
    normalized = text.strip().upper()
    has_yes = re.search(r'\bYES\b', normalized) is not None
    has_no = re.search(r'\bNO\b', normalized) is not None
    if has_yes != has_no:
      return has_yes
    return None

  @staticmethod
  def _parse_score(value: object) -> float | None:
    """Reads a score in [0, 1] from a JSON value, clamping it into range."""
    if isinstance(value, bool):
      return None
    if isinstance(value, str):
      try:
        value = float(value)
      except ValueError:
        return None
    if not isinstance(value, (int, float)) or value != value:  # Rejects NaN.
      return None
    return min(max(float(value), 0.0), 1.0)

  @classmethod
  def _parse_judgement(cls, text: str) -> ArgMatchResult | None:
    """Reads the model's verdict, score and reason from its response.

    The model is asked for a JSON object with `verdict`, `score` and `reason`.
    A missing score falls back to 1.0 or 0.0 from the verdict, and a missing
    verdict falls back to whether the score reaches the match threshold. A
    response that is not JSON is read as a bare YES or NO.

    Returns:
      The result, or None when the response has neither a verdict nor a score.
    """
    decoder = json.JSONDecoder()
    found_dict = False
    for match in re.finditer(r'\{', text):
      try:
        candidate, _ = decoder.raw_decode(text, match.start())
      except ValueError:
        continue
      if isinstance(candidate, dict):
        found_dict = True
        result = cls._parse_payload(candidate)
        if result is not None:
          return result

    if found_dict:
      return None
    verdict = cls._parse_verdict(text)
    if verdict is None:
      return None
    return ArgMatchResult(
        matched=verdict,
        reason=f'The model answered {"YES" if verdict else "NO"}.',
    )

  @classmethod
  def _parse_payload(cls, payload: dict[str, object]) -> ArgMatchResult | None:
    """Reads a verdict, score and reason from a decoded JSON object."""
    raw_verdict = payload.get('verdict')
    verdict: bool | None
    if isinstance(raw_verdict, bool):
      verdict = raw_verdict
    elif isinstance(raw_verdict, str):
      verdict = cls._parse_verdict(raw_verdict)
    else:
      verdict = None
    score = cls._parse_score(payload.get('score'))
    if verdict is None:
      if score is None:
        return None
      verdict = score >= cls._SCORE_MATCH_THRESHOLD
    if score is None:
      score = 1.0 if verdict else 0.0

    raw_reason = payload.get('reason')
    reason = (
        raw_reason.strip()
        if isinstance(raw_reason, str) and raw_reason.strip()
        else f'The model answered {"YES" if verdict else "NO"}.'
    )
    return ArgMatchResult(matched=verdict, score=score, reason=reason)

  def _build_prompt(
      self,
      tool_name: str,
      arg_name: str,
      expected_json: str,
      actual_json: str,
  ) -> str:
    criteria = self.criteria or self._DEFAULT_CRITERIA
    return (
        'You compare one argument of a tool call with the value that was'
        ' expected for it.\n\n'
        f'Tool: {tool_name}\n'
        f'Argument: {arg_name}\n'
        f'Expected value: {expected_json}\n'
        f'Actual value: {actual_json}\n\n'
        f'{criteria}\n\n'
        'Does the actual value match the expected value? Respond with only a'
        ' JSON object with these fields:\n'
        '- "verdict": "YES" if the values match, otherwise "NO".\n'
        '- "score": a number from 0.0 to 1.0 for how closely the actual value'
        ' matches the expected value, where 1.0 means the same meaning and 0.0'
        ' means unrelated. Give YES a score of at least 0.5 and NO a score'
        ' below 0.5.\n'
        '- "reason": one short sentence explaining the verdict.\n'
    )

  async def _ask_model(self, prompt: str) -> str:
    model_name = self.model or self._DEFAULT_MODEL
    if self._llm is None:
      self._llm = LLMRegistry.new_llm(model_name)
    llm_request = LlmRequest(
        model=model_name,
        contents=[
            genai_types.Content(
                role='user', parts=[genai_types.Part(text=prompt)]
            )
        ],
        config=self.generate_content_config.model_copy(deep=True),
    )
    try:
      return await asyncio.wait_for(
          self._collect_response_text(self._llm, llm_request),
          timeout=self.timeout_seconds,
      )
    except asyncio.TimeoutError as e:
      raise TimeoutError(
          f'Semantic matcher model {model_name!r} did not respond within'
          f' {self.timeout_seconds} seconds.'
      ) from e

  @staticmethod
  async def _collect_response_text(
      llm: BaseLlm, llm_request: LlmRequest
  ) -> str:
    response_text = ''
    async with Aclosing(llm.generate_content_async(llm_request)) as agen:
      async for llm_response in agen:
        if llm_response.content is None or not llm_response.content.parts:
          continue
        for part in llm_response.content.parts:
          if part.text and not part.thought:
            response_text += part.text
    return response_text
