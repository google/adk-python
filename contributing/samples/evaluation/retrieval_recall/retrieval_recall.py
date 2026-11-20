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

"""A custom eval metric: did the agent retrieve the expected documents?

Scores each invocation by retrieval recall: the share of expected document ids
that appear in the documents the agent actually retrieved. The expected ids
come from the tool responses recorded in the eval case's expected invocation.

Wire it into `adk eval` through `custom_metrics` in the eval config, the same
way contributing/samples/evaluation/custom_metric does.
"""

from __future__ import annotations

from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import get_all_tool_responses
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult

_RETRIEVAL_TOOL = "search_docs"
_PASS_THRESHOLD = 1.0


def _retrieved_ids(invocation: Invocation) -> list[str]:
  """Returns the document ids in every search_docs response, in order.

  A response that does not have the expected shape (no `documents` list, or
  entries that are not dicts with an `id`) contributes no ids, so a failed or
  malformed retrieval scores as "nothing retrieved" instead of raising.
  """
  ids: list[str] = []
  for response in get_all_tool_responses(invocation.intermediate_data):
    if response.name != _RETRIEVAL_TOOL:
      continue
    documents = (response.response or {}).get("documents")
    if not isinstance(documents, list):
      continue
    for document in documents:
      if not isinstance(document, dict):
        continue
      doc_id = document.get("id")
      if isinstance(doc_id, str) and doc_id not in ids:
        ids.append(doc_id)
  return ids


def retrieval_recall(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None,
) -> EvaluationResult:
  """Scores the share of expected documents that were actually retrieved."""
  if expected_invocations is None:
    raise ValueError("retrieval_recall needs expected invocations.")
  if len(actual_invocations) != len(expected_invocations):
    raise ValueError(
        "retrieval_recall needs the same number of actual and expected"
        f" invocations; got {len(actual_invocations)} and"
        f" {len(expected_invocations)}."
    )

  per_invocation_results = []
  scores = []
  for actual, expected in zip(
      actual_invocations, expected_invocations, strict=True
  ):
    relevant = _retrieved_ids(expected)
    if not relevant:
      # Nothing was expected to be retrieved on this turn, so there is
      # nothing to score; leave it out of the average.
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual, expected_invocation=expected
          )
      )
      continue
    retrieved = set(_retrieved_ids(actual))
    score = sum(doc_id in retrieved for doc_id in relevant) / len(relevant)
    scores.append(score)
    per_invocation_results.append(
        PerInvocationResult(
            actual_invocation=actual,
            expected_invocation=expected,
            score=score,
            eval_status=(
                EvalStatus.PASSED
                if score >= _PASS_THRESHOLD
                else EvalStatus.FAILED
            ),
        )
    )

  if not scores:
    return EvaluationResult(per_invocation_results=per_invocation_results)

  overall_score = sum(scores) / len(scores)
  return EvaluationResult(
      overall_score=overall_score,
      overall_eval_status=(
          EvalStatus.PASSED
          if overall_score >= _PASS_THRESHOLD
          else EvalStatus.FAILED
      ),
      per_invocation_results=per_invocation_results,
  )
