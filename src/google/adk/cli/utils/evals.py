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

from __future__ import annotations

import dataclasses
import os
from typing import Any
from typing import Tuple

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from typing_extensions import deprecated

from ...evaluation.eval_case import Invocation
from ...evaluation.evaluation_generator import EvaluationGenerator
from ...evaluation.gcs_eval_set_results_manager import GcsEvalSetResultsManager
from ...evaluation.gcs_eval_sets_manager import GcsEvalSetsManager
from ...sessions.session import Session


class GcsEvalManagers(BaseModel):
  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      arbitrary_types_allowed=True,
  )

  eval_sets_manager: GcsEvalSetsManager

  eval_set_results_manager: GcsEvalSetResultsManager


@deprecated('Use convert_session_to_eval_invocations instead.')
def convert_session_to_eval_format(session: Session) -> list[dict[str, Any]]:
  """Converts a session data into eval format.

  Args:
      session: The session that should be converted.

  Returns:
      list: A single evaluation dataset in the required format.
  """
  eval_case = []
  events = session.events if session and session.events else []

  for event in events:
    if event.author == 'user':
      if not event.content or not event.content.parts:
        continue

      # Extract user query
      content = event.content
      parts = content.parts

      query = parts[0].text or ''

      # Find the corresponding tool usage or response for the query
      expected_tool_use = []
      intermediate_agent_responses = []

      # Check subsequent events to extract tool uses or responses for this turn.
      for subsequent_event in events[events.index(event) + 1 :]:
        event_author = subsequent_event.author or 'agent'
        if event_author == 'user':
          # We found an event where the author was the user. This means that a
          # new turn has started. So close this turn here.
          break

        if not subsequent_event.content or not subsequent_event.content.parts:
          continue

        for subsequent_part in subsequent_event.content.parts:
          # Some events have both function call and reference

          if subsequent_part.function_call:
            tool_name = subsequent_part.function_call.name or ''
            tool_input = subsequent_part.function_call.args or {}
            expected_tool_use.append({
                'tool_name': tool_name,
                'tool_input': tool_input,
            })
          elif subsequent_part.text:
            # Also keep track of all the natural language responses that
            # agent (or sub agents) generated.
            intermediate_agent_responses.append(
                {'author': event_author, 'text': subsequent_part.text}
            )

      # If we are here then either we are done reading all the events or we
      # encountered an event that had content authored by the end-user.
      # This, basically means an end of turn.
      # We assume that the last natural language intermediate response is the
      # final response from the agent/model. We treat that as a reference.
      eval_case.append({
          'query': query,
          'expected_tool_use': expected_tool_use,
          'expected_intermediate_agent_responses': intermediate_agent_responses[
              :-1
          ],
          'reference': (
              intermediate_agent_responses[-1]['text']
              if intermediate_agent_responses
              else ''
          ),
      })

  return eval_case


def convert_session_to_eval_invocations(session: Session) -> list[Invocation]:
  """Converts a session data into a list of Invocation.

  Args:
      session: The session that should be converted.

  Returns:
      list: A list of invocation.
  """
  events = session.events if session and session.events else []
  return EvaluationGenerator.convert_events_to_eval_invocations(events)


def create_gcs_eval_managers_from_uri(
    eval_storage_uri: str,
) -> GcsEvalManagers:
  """Creates GcsEvalManagers from eval_storage_uri.

  Args:
      eval_storage_uri: The evals storage URI to use. Supported URIs:
        gs://<bucket name>. If a path is provided, the bucket will be extracted.

  Returns:
      GcsEvalManagers: The GcsEvalManagers object.

  Raises:
      ValueError: If the eval_storage_uri is not supported.
  """
  if eval_storage_uri.startswith('gs://'):
    gcs_bucket = eval_storage_uri.split('://')[1]
    eval_sets_manager = GcsEvalSetsManager(
        bucket_name=gcs_bucket, project=os.environ['GOOGLE_CLOUD_PROJECT']
    )
    eval_set_results_manager = GcsEvalSetResultsManager(
        bucket_name=gcs_bucket, project=os.environ['GOOGLE_CLOUD_PROJECT']
    )
    return GcsEvalManagers(
        eval_sets_manager=eval_sets_manager,
        eval_set_results_manager=eval_set_results_manager,
    )
  else:
    raise ValueError(
        f'Unsupported evals storage URI: {eval_storage_uri}. Supported URIs:'
        ' gs://<bucket name>'
    )
