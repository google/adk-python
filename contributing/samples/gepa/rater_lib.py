# Copyright 2025 Google LLC
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

"""Library for rating agent trajectories."""

from __future__ import annotations

import re
from typing import Any

from absl import logging
from google.genai import types
import jinja2
from retry import retry

from google import genai


def parse_rubric_validation_response(
    rubric_val_response: str,
) -> dict[str, str]:
  """Parses rubric validation response text into a dictionary.

  Args:
    rubric_val_response: The text response from rubric validation.

  Returns:
    A dictionary containing parsed property, evidence, rationale, and verdict.
  """
  PROPERTY_PATTERN = (
      r'Property:\s*([\s\S]*?)(?=(?:Evidence:|Rationale:|Verdict:|$))'
  )
  EVIDENCE_PATTERN = r'Evidence:\s*([\s\S]*?)(?=(?:Rationale:|Verdict:|$))'
  RATIONALE_PATTERN = r'Rationale:\s*([\s\S]*?)(?=(?:Evidence:|Verdict:|$))'
  VERDICT_PATTERN = r'Verdict:\s*([\s\S]*?)(?=(?:Evidence:|Rationale:|$))'

  property_list = []
  evidence_list = []
  rationale_list = []
  fulfillment_list = []
  property_blocks = rubric_val_response.split('Property: ')[1:]
  for property_block in property_blocks:
    property_name = re.search(PROPERTY_PATTERN, 'Property: ' + property_block)
    if property_name is None:
      continue
    property_name = property_name.group(1).strip()
    property_list.append(property_name)

    evidence_match = re.search(EVIDENCE_PATTERN, property_block, re.DOTALL)
    evidence = evidence_match.group(1).strip() if evidence_match else ''
    evidence_list.append(evidence)

    rationale_match = re.search(RATIONALE_PATTERN, property_block, re.DOTALL)
    rationale = rationale_match.group(1).strip() if rationale_match else ''
    rationale_list.append(rationale)

    verdict = re.search(VERDICT_PATTERN, property_block)
    if verdict is None:
      verdict_str = 'not_found'
    else:
      verdict_str = verdict.group(1).strip().lower()
    if 'yes' in verdict_str:
      verdict_str = 'yes'
    elif 'no' in verdict_str:
      verdict_str = 'no'
    elif 'unknown' in verdict_str:
      verdict_str = 'unknown'
    else:
      verdict_str = 'not_found'
    fulfillment_list.append(verdict_str)
  return dict(
      property=property_list[0],
      evidence=evidence_list[0],
      rationale=rationale_list[0],
      verdict=fulfillment_list[0],
  )


def format_user_agent_conversation(conv: list[dict[str, Any]]) -> str:
  """Formats a conversation between user and agent into a string.

  Args:
    conv: A list of conversation turns.

  Returns:
    A formatted string representing the conversation.
  """
  # conv is a list in this eval data
  # if not, manually convert to list to re-use these logics
  # if not isinstance(conv, list):
  #   conv = [conv]
  res = ''
  turn_idx = 1
  for turn in conv:
    # if 'request' in conv[turn]:\
    role = turn['role']
    for part in turn['parts']:
      if role == 'user' and (txt := part.get('text')):
        res = res + f'USER TURN {turn_idx}:\n' + txt + '\n'
        turn_idx += 1
      elif role == 'model' and (txt := part.get('text')):
        res = res + f'The agent response is: {txt}' + '\n'
      elif fc := part.get('function_call'):
        res = (
            res
            + f'The agent called the function {fc["name"]} with the following'
            f' function arguments: {fc["args"]}.\n'
        )
      elif fc := part.get('function_response'):
        res = (
            res
            + 'The execution result from the agent of function'
            f' {fc["name"]} is: \n{fc["response"]}\n'
        )
  return res


_COMPLETION_RUBRIC_CRITERIA = """The agent fulfilled the user's primary request. Description: It measures if the agent successfully completed the action the user initiated the contact for (e.g., processed a return, provided a tracking number, answered a policy question). A "yes" requires confirmed completion within the transcript."""


class Rater:
  """Rates agent trajectories using an LLM based on rubrics."""

  def __init__(
      self,
      tool_declarations: str,
      developer_instructions: str = '',
      rubric: str = _COMPLETION_RUBRIC_CRITERIA,
      validation_template_path: str = 'rubric_validation_template.txt',
  ):
    """Initializes the Rater.

    Args:
      tool_declarations: JSON string of tool declarations for the agent.
      developer_instructions: Developer instructions.
      rubric: rubric.
      validation_template_path: Path to rubric validation template.
    """
    self._client = genai.Client()
    self._tool_declarations = tool_declarations
    self._developer_instructions = developer_instructions
    with open(validation_template_path) as f:
      self._rubric_validation_template = f.read().strip()
    logging.info(
        'Loaded rubric validate template from path=%s', validation_template_path
    )
    self._rubric = rubric

  @retry(tries=3, delay=2, backoff=2)
  def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Rates a conversation based on rubric criteria.

    Args:
      messages: A list of conversation messages between user and agent.

    Returns:
      A dictionary containing rating information including score.
    """
    env = jinja2.Environment()
    env.globals['user_input'] = (
        messages[0].get('parts', [{}])[0].get('text', '') if messages else ''
    )
    env.globals['developer_instructions'] = self._developer_instructions
    env.globals['tool_declarations'] = self._tool_declarations
    env.globals['model_response'] = format_user_agent_conversation(messages)
    env.globals['decomposed_rubric'] = '* ' + self._rubric
    contents = env.from_string(self._rubric_validation_template).render()
    resp = self._client.models.generate_content(
        model='gemini-2.5-pro',
        contents=contents,
        config=types.GenerateContentConfig(
            candidate_count=1,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_budget=-1
            ),
        ),
    )
    got = parse_rubric_validation_response(resp.text)
    got = dict(got)
    got['score'] = float(got['verdict'] == 'yes')
    got['rating_criteria'] = got.pop('property')
    return got
