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
import json
import logging
import re
import statistics
from typing import ClassVar
from typing import Optional

from google.genai import types as genai_types
from pydantic import ValidationError
from typing_extensions import override

from ..models.base_llm import BaseLlm
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from .app_details import AppDetails
from .eval_case import Invocation
from .eval_case import InvocationEvent
from .eval_case import InvocationEvents
from .eval_metrics import EvalMetric
from .eval_metrics import HallucinationsCriterion
from .eval_metrics import Interval
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult
from .llm_as_judge_utils import get_eval_status
from .llm_as_judge_utils import get_text_from_content
from .llm_as_judge_utils import get_tool_declarations_as_json_str

logger = logging.getLogger("google_adk." + __name__)

_HALLUCINATIONS_V1_SEGMENTER_PROMPT = """
You are a helpful and harmless AI assistant. You will be provided with a model-generated response.
Your task is to segment the provided response sentence by sentence so that we could analyze each sentence in the future.

**Instructions:**
1. Overall, you should decompose the whole provided response into individual sentences. You should make sure the output covers ALL the sentences in the provided response block.
2. You should COPY each sentence as it is, WORD BY WORD. DO NOT modify the sentence or the surrounding punctuation.
3. If there are bullet points in the response, you should segment each bullet point into DIFFERENT sentences. If one bullet point has sub bullet points, you should further decompose sub bullet points into DIFFERENT sentences.
For example, if there are responses like "it has three criteria: * aaa. * bbb. * ccc", you should segment them into FOUR sentences: "it has three criteria", "aaa", "bbb", "ccc". Bullet points could start with numbers (1/2/3/etc) or symbols like "*", "-" etc.
4. When encountering tables, you should include the whole table in ONE sentence output.
5. Each sentence should be meaningful to further analyze on. DO NOT ONLY put symbols themselves into a sentence.
6. You should ONLY output segmented sentences in the provided response. DO NOT make up any new sentences.

**Input Format:**

The input will be the model-generated response:
* **Response:** The model-generated response to be analyzed.

**Output Format:**

For each decomposed sentence, wrap them with <sentence> and </sentence> like the following:
<sentence>...</sentence>
<sentence>...</sentence>

**Example:**

**Input:**

**Response Begin**
There are three kinds of fruits:
1. Apples are red.
2. Bananas are green.
3. Pears are purple.

For prices:
* Bananas are cheaper than apples.

Enjoy your fruit!
**Response End**

**Output:**
<sentence>There are three kinds of fruits:</sentence>
<sentence>1. Apples are red.</sentence>
<sentence>2. Bananas are green.</sentence>
<sentence>3. Pears are purple.</sentence>
<sentence>For prices:</sentence>
<sentence>* Bananas are cheaper than apples.</sentence>
<sentence>Enjoy your fruit!</sentence>

**Now, given the following response, please segment the response into sentences:**

**Input:**

**Response Begin**
{response}
**Response End**

**Your Sentence Segmentation Output:**
""".strip()

_HALLUCINATIONS_V1_VALIDATOR_PROMPT = """
You are a helpful and harmless AI assistant. You will be provided with a textual context and sentences from a model-generated response.
Your task is to analyze sentence by sentence and classify each sentence according to its relationship with the provided context.

**Instructions:**

1. **Read the textual context carefully.**
2. **For each sentence, assign one of the following labels:**
    * **`supported`**: The sentence is entailed by the given context. Provide a supporting excerpt from the context. The supporting except must *fully* entail the sentence.
    * **`unsupported`**: The sentence is not entailed by the given context. No excerpt is needed for this label.
    * **`contradictory`**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.
    * **`disputed`**: The given context contains both supporting and contradicting information. Provide both supporting and contradicting excerpt from the context.
    * **`not_applicable`**: The sentence does not require factual attribution (e.g., opinions, planning steps, greetings, questions, disclaimers, mathematical calculation).
3. **For each label, provide a short rationale explaining your decision.** The rationale should be separate from the excerpt.
4. **Be very strict with your `supported`, `contradictory` and `disputed` decisions.** Unless you can find straightforward, indisputable evidence excepts *in the context* that a sentence is `supported`, `contradictory` or `disputed`, consider it `unsupported`.  You should not employ world knowledge unless it is truly trivial.
5. "tool_outputs" blocks contain code execution results of the "tool_code" blocks immediately above them. If any sentence is based on "tool_outputs" results, first analyze if the corresponding "tool_code" is supported and if the results are error-free. Only if the "tool_code" block is supported, you can treat code execution results as correct.
6. If you need to cite multiple supporting excerpts, simply concatenate them. Excerpt could be summary from the context if it is too long.

**Input Format:**

The input will consist of two parts, clearly separated:

* **Context:**  The textual context used to generate the response.
* **Sentences:** The sentences from the model-generated response to be analyzed. Each sentence will be wrapped in <sentence>...</sentence>.

**Output Format:**

For each sentence, output a block of text with the following fields:

* sentence: The sentence being analyzed. Please directly copy the sentence which is provided.
* label: One of `supported`, `unsupported`, `contradictory`, `disputed` or `not_applicable`.
* rationale: A brief explanation for the assessment
* supporting_excerpt: A relevant excerpt from the context that supports the sentence. Only required for `supported` and `disputed` labels.
* contradicting_excerpt: A relevant excerpt from the context that contradicts with the sentence. Only required for `contradictory` and `disputed` labels.

**Example:**

**Input:**

**Context Begin**
Apples are red fruits. Bananas are yellow fruits. Pears are purple fruits. Pears are blue fruits.
**Context End**

**Sentences Begin**
<sentence>Apples are red.</sentence>
<sentence>Bananas are green.</sentence>
<sentence>Pears are purple.</sentence>
<sentence>Bananas are cheaper than apples.</sentence>
<sentence>Enjoy your fruit!</sentence>
**Sentences End**

**Output:**
sentence: Apples are red.
label: supported
rationale: The context explicitly states that apples are red.
supporting_excerpt: Apples are red fruits.
contradicting_excerpt: null

sentence: Bananas are green.
label: contradictory
rationale: The context states that bananas are yellow, not green.
supporting_excerpt: null
contradicting_excerpt: Bananas are yellow fruits.

sentence: Pears are purple.
label: disputed
rationale: The context states that pears are purple but it also states that pears are blue.
supporting_excerpt: Pears are purple fruits
contradicting_excerpt: Pears are blue fruits

sentence: Bananas are cheaper than apples.
label: unsupported
rationale: The context does not mention the price of bananas or apples.
supporting_excerpt: null
contradicting_excerpt: null

sentence: Enjoy your fruit!
label: not_applicable
rationale: This is a general expression and does not require factual attribution.
supporting_excerpt: null
contradicting_excerpt: null

**Now, please analyze the following context and sentences:**

**Input:**

**Context Begin**
{context}
**Context End**

**Sentences Begin**
{sentences}
**Sentences End**

**Output:**
""".strip()

_POSITIVE_LABELS = frozenset(["supported", "not_applicable"])

_NEGATIVE_LABELS = frozenset(["unsupported", "contradictory", "disputed"])


@dataclasses.dataclass(frozen=True)
class EvaluationStep:
  """The context and natural language response to be evaluated at a step."""

  context: str
  nl_response: str


def _parse_sentences(response_text: str) -> list[str]:
  """Parses sentences from LLM response."""
  return re.findall(r"<sentence>(.*?)</sentence>", response_text, re.DOTALL)


def _parse_validation_results(
    response_text: str,
) -> list[dict[str, Optional[str]]]:
  """Parses sentence validation results from LLM response."""
  results = []
  pattern = re.compile(
      r"sentence:(.*?)\nlabel:(.*?)\nrationale:(.*?)\nsupporting_excerpt:(.*?)\ncontradicting_excerpt:(.*?)(?=\nsentence:|\Z)",
      re.DOTALL | re.IGNORECASE,
  )
  for match in pattern.finditer(response_text.strip()):
    try:
      sentence, label, rationale, sup_exc, con_exc = match.groups()
      results.append({
          "sentence": sentence.strip(),
          "label": label.strip(),
          "rationale": rationale.strip(),
          "supporting_excerpt": (
              sup_exc.strip() if sup_exc.strip().lower() != "null" else None
          ),
          "contradicting_excerpt": (
              con_exc.strip() if con_exc.strip().lower() != "null" else None
          ),
      })
    except Exception:  # pylint: disable=broad-except
      logger.warning(
          "Failed to parse sentence validation block: %s", match.group(0)
      )
  return results


@experimental
class HallucinationsV1Evaluator(Evaluator):
  """Evaluates whether a model response contains any false, contradictory, or unsupported claims.

  The metric follows a two-step process:
  1. Segmenter: Segments the agent response into individual sentences.
  2. Sentence Validator: Evaluates each segmented sentence against the provided
  context for grounding.

  The metric computes the Accuracy Score (AS): the percentage of sentences that
  are supported or not_applicable.
  """

  criterion_type: ClassVar[type[HallucinationsCriterion]] = (
      HallucinationsCriterion
  )

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

    expected_criterion_type_error = ValueError(
        f"`{eval_metric.metric_name}` metric expects a criterion of type"
        f" `{HallucinationsV1Evaluator.criterion_type}`."
    )

    try:
      if self._eval_metric.criterion is None:
        raise expected_criterion_type_error

      self._criterion = HallucinationsV1Evaluator.criterion_type.model_validate(
          self._eval_metric.criterion.model_dump()
      )
    except ValidationError as e:
      raise expected_criterion_type_error from e

    self._judge_model_options = self._criterion.judge_model_options
    self._judge_model = self._setup_auto_rater()
    self.segmenter_prompt = _HALLUCINATIONS_V1_SEGMENTER_PROMPT
    self.sentence_validator_prompt = _HALLUCINATIONS_V1_VALIDATOR_PROMPT
    self._model = self._judge_model_options.judge_model
    self._model_config = self._judge_model_options.judge_model_config

  def _setup_auto_rater(self) -> BaseLlm:
    model_id = self._judge_model_options.judge_model
    llm_registry = LLMRegistry()
    llm_class = llm_registry.resolve(model_id)
    return llm_class(model=model_id)

  @staticmethod
  def get_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.HALLUCINATIONS_V1.value,
        description=(
            "This metric assesses whether a model response contains any false,"
            " contradictory, or unsupported claims using a LLM as judge. Value"
            " range for this metric is [0,1], with values closer to 1 more"
            " desirable."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  def _create_context_for_step(
      self,
      app_details: Optional[AppDetails],
      invocation: Invocation,
      events: list[InvocationEvent],
  ) -> str:
    """Creates context string for sentence validation based on a list of events.

    Given an invocation and a list of events, this method creates a context
    string that is used to evaluate the natural language responses (NL
    responses) generated by the agent. The context is constructed by combining
    the developer instructions, user query, tool definitions, and tool
    invocations and their results.

    The general format for the context has two parts. First, the header block:
    ```
    Developer instructions:
    <Agent_1 name>:
    <Agent_1 instructions>
    ...
    <Agent_n name>:
    <Agent_n instructions>

    User prompt:
    <User prompt>

    Tool definitions:
    <Tool definitions>
    ```

    Second, is the step-block, which occurs once for each previous step. Recall
    that in the list of all invocation events, a step is the sequence of
    events that occurs between NL responses.
    ```
    tool_calls:
    <Tool calls>

    tool_outputs:
    <Tool responses>

    <NL_response, only for previous steps, not the current step being evaluated>
    ```

    The following is an example of a context string:
    ```
    Developer instructions:
    You are a helpful agent that can tell the time and get the weather.

    User prompt:
    Get the current time and weather of San Francisco.

    Tool definitions:
      [
        {
          "name": "get_current_time",
          "description": '''Gets the current time in the timezone.

    Args:
      timezone: The timezone to get the time of.

    Returns:
      The time in the timezone.
    ''',
          "parameters": {
            "type": "object",
            "properties": {
              "timezone": {
                "description": "The timezone to get the time of.",
                "type": "string"
              }
            }
          }
        },
        {
          "name": "get_weather",
          "description": '''Gets the weather of the given place at the given
          time.

    Args:
      location: The location for which to retrieve weather information.
      time: The specific time point for the weather data.

    Returns:
      The weather at the given time and place.
    ''',
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "description": "The location for which to retrieve weather
                information.",
                "type": "string"
              },
              "time": {
                "description": "The specific time point for the weather data.",
                "type": "string"
              }
            }
          }
        },
      ]

    tool_calls:
      [
        {
          "name": "get_current_time",
          "args": {"timezone": "PST"},
        },
      ]

    tool_outputs:
    "10:30 AM PST Sep 12, 2025"
    ```

    Args:
      app_details: App details to get developer instructions and tool
        definitions.
      invocation: Invocation to get user prompt.
      events: The list of events that occurred before the current step.

    Returns:
      The context string to include in the sentence validation prompt.
    """
    developer_instructions = ""
    tool_declarations = "Agent has no tools."
    if app_details:
      instructions = []
      for agent_name in app_details.agent_details:
        agent_instructions = app_details.get_developer_instructions(agent_name)
        if agent_instructions:
          instructions.append(agent_name + ":\n" + agent_instructions)
      developer_instructions = "\n\n".join(instructions)
      tool_declarations = get_tool_declarations_as_json_str(app_details)

    context_parts = []
    context_parts.append(f"Developer instructions:\n{developer_instructions}\n")
    context_parts.append(
        f"User prompt:\n{get_text_from_content(invocation.user_content)}\n"
    )
    context_parts.append("Tool definitions:")
    context_parts.append(f"{tool_declarations}\n")

    for event in events:
      if not event.content or not event.content.parts:
        continue
      tool_calls = [
          part.function_call
          for part in event.content.parts
          if part.function_call
      ]
      tool_responses = [
          part.function_response
          for part in event.content.parts
          if part.function_response
      ]
      nl_responses = [part.text for part in event.content.parts if part.text]

      if nl_responses:
        context_parts.append("\n".join(nl_responses) + "\n")

      if tool_calls:
        context_parts.append("tool_calls:")
        context_parts.append(
            json.dumps(
                [
                    tool_call.model_dump(exclude_none=True)
                    for tool_call in tool_calls
                ],
                indent=2,
            )
            + "\n"
        )
      if tool_responses:
        context_parts.append("tool_outputs:")
        context_parts.append(
            json.dumps(
                [
                    tool_response.model_dump(exclude_none=True)
                    for tool_response in tool_responses
                ],
                indent=2,
            )
            + "\n"
        )

    return "\n".join(context_parts)

  async def _evaluate_nl_response(
      self, nl_response: str, context: str
  ) -> tuple[Optional[float], str]:
    """Runs segmentation and validation for a single NL response."""
    # Segmentation step: split the NL response into sentences.
    segmenter_llm_request = LlmRequest(
        model=self._model,
        contents=[
            genai_types.Content(
                parts=[
                    genai_types.Part(
                        text=self.segmenter_prompt.format(response=nl_response)
                    )
                ],
                role="user",
            )
        ],
        config=self._model_config,
    )
    try:
      async with Aclosing(
          self._judge_model.generate_content_async(segmenter_llm_request)
      ) as agen:
        segmenter_response = await agen.__anext__()
        sentences = _parse_sentences(
            get_text_from_content(segmenter_response.content)
        )
    except Exception as e:
      return None, f"Error during sentence segmentation: {e}"

    if not sentences:
      return None, "No sentences produced by segmenter."

    sentences_str = "\n".join([f"<sentence>{s}</sentence>" for s in sentences])

    # Evaluation step: evaluate each sentence against the context.
    validator_llm_request = LlmRequest(
        model=self._model,
        contents=[
            genai_types.Content(
                parts=[
                    genai_types.Part(
                        text=self.sentence_validator_prompt.format(
                            context=context, sentences=sentences_str
                        )
                    )
                ],
                role="user",
            )
        ],
        config=self._model_config,
    )
    try:
      async with Aclosing(
          self._judge_model.generate_content_async(validator_llm_request)
      ) as agen:
        validator_response = await agen.__anext__()
        validation_results = _parse_validation_results(
            get_text_from_content(validator_response.content)
        )
    except Exception as e:
      return None, f"Error during sentence validation: {e}"

    scores = []
    for result in validation_results:
      label = result.get("label")
      if label is None:
        logger.debug("No label found for sentence: %s", result)
        continue

      label = label.strip().lower()
      if label in _POSITIVE_LABELS:
        scores.append(1)
      elif label in _NEGATIVE_LABELS:
        scores.append(0)
      else:
        logger.debug("Unexpected label: %s", label)

    accuracy_score = statistics.mean(scores) if scores else None
    return accuracy_score, json.dumps(validation_results, indent=2)

  def _get_steps_to_evaluate(self, actual: Invocation) -> list[EvaluationStep]:
    """Gathers all NL responses and their contexts for evaluation.

    An invocation may look like:
    ```
    {
      "invocation_id": "1234",
      "user_content": {
        "parts": [{"text": "User query."}],
      },
      "final_response": {
        "parts": [{"text": "Final response."}],
      },
      "app_details": {
          "agent_details": {
              "root": {
                  "name": "root",
                  "instructions": "Root agent instructions.",
                  "tool_declarations": []
              }
          }
      },
      "intermediate_data": {
        "invocation_events": [
          {
            "author": "root",
            "content": {
              "parts": [{"text": "Intermediate response 1."}],
            }
          },
          {
            "author": "root",
            "content": {
              "parts": [
                {
                  "function_call": {
                    "name": "tool_1",
                    "args": {
                      "arg_1": "value_1"
                    }
                  }
                },
                {
                  "function_response": {
                    "response": "Tool response"
                  }
                },
                {
                  "text": "Intermediate response 2."
                },
              ]
            }
          }
        ]
      }
    }
    ```

    Args:
      actual: The actual invocation to evaluate.

    Returns:
      EvaluationSteps, one for each NL response to evaluate.
    """
    step_evaluations = []
    events_for_context: list[InvocationEvent] = []
    all_events = []
    if isinstance(actual.intermediate_data, InvocationEvents):
      all_events = actual.intermediate_data.invocation_events or []

    if self._criterion.evaluate_intermediate_nl_responses:
      for event in all_events:
        nl_parts = (
            [p.text for p in event.content.parts if p.text]
            if event.content and event.content.parts
            else []
        )
        if nl_parts:
          context = self._create_context_for_step(
              actual.app_details, actual, events_for_context
          )
          for nl_response in nl_parts:
            step_evaluations.append(
                EvaluationStep(nl_response=nl_response, context=context)
            )
        events_for_context.append(event)
    else:
      events_for_context = all_events

    final_response_text = get_text_from_content(actual.final_response)
    if final_response_text:
      context = self._create_context_for_step(
          actual.app_details, actual, events_for_context
      )
      step_evaluations.append(
          EvaluationStep(nl_response=final_response_text, context=context)
      )
    return step_evaluations

  def _aggregate_invocation_results(
      self,
      per_invocation_results: list[PerInvocationResult],
  ) -> EvaluationResult:
    """Aggregates the per invocation results to get the overall score."""
    valid_results = [r for r in per_invocation_results if r.score is not None]
    if not valid_results:
      return EvaluationResult(
          overall_score=None,
          overall_eval_status=EvalStatus.NOT_EVALUATED,
          per_invocation_results=per_invocation_results,
      )

    overall_fs_score = statistics.mean([r.score for r in valid_results])
    return EvaluationResult(
        overall_score=overall_fs_score,
        overall_eval_status=get_eval_status(
            overall_fs_score, self._eval_metric.threshold
        ),
        per_invocation_results=per_invocation_results,
    )

  @override
  async def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]],
  ) -> EvaluationResult:
    # expected_invocations are not required by the metric and if they are not
    # supplied, we provide a list of None to rest of the code.
    expected_invocations = (
        [None] * len(actual_invocations)
        if expected_invocations is None
        else expected_invocations
    )
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      step_evaluations = self._get_steps_to_evaluate(actual)

      if not step_evaluations:
        per_invocation_results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=None,
                eval_status=EvalStatus.NOT_EVALUATED,
                rubric_scores=[],
            )
        )
        continue

      scores_per_step = []
      for step in step_evaluations:
        fs_score, _ = await self._evaluate_nl_response(
            step.nl_response, step.context
        )
        if fs_score is not None:
          scores_per_step.append(fs_score)

      invocation_score = (
          statistics.mean(scores_per_step) if scores_per_step else None
      )

      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=invocation_score,
              eval_status=get_eval_status(
                  invocation_score, self._eval_metric.threshold
              ),
              rubric_scores=[],
          )
      )

    if per_invocation_results:
      return self._aggregate_invocation_results(per_invocation_results)
    return EvaluationResult()
