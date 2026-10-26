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

"""A workflow node that routes on a classifier's answer."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated
from typing import Any
from typing import TYPE_CHECKING
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from pydantic import SerializeAsAny
from typing_extensions import override

from ...events.event import Event
from ...features import experimental
from ...features import FeatureName
from ...workflow._base_node import BaseNode
from ...workflow._errors import GraphValidationError
from ...workflow._errors import WorkflowDataError
from ._base_classifier import BaseClassifier
from ._types import Answer
from ._types import Binary
from ._types import BinaryAnswer
from ._types import Choice
from ._types import ChoiceAnswer

if TYPE_CHECKING:
  from ...agents.context import Context
  from ...workflow._graph import Edge
  from ...workflow._graph import RouteValue

_QUESTION_NAME = 'route'


@experimental(FeatureName.CLASSIFIER)
class ClassifierNode(BaseNode):
  """Routes a workflow by asking a classifier one question about its input.

  The node classifies its ``node_input`` and emits the answer as the route: the
  chosen label for a :class:`Choice`, or ``True``/``False`` for a
  :class:`Binary`, like any other boolean routing node. Its output is the
  unchanged input, so the routed-to node receives what the classifier saw.

  When the node has routed outgoing edges, graph validation checks that every
  route is a possible answer and that every possible answer has an edge, unless
  a ``DEFAULT_ROUTE`` or unrouted edge catches the rest.
  """

  classifier: SerializeAsAny[BaseClassifier]
  """The classifier that answers :attr:`question`."""

  question: Annotated[Union[Choice, Binary], Field(discriminator='type')]
  """The question whose answer picks the route."""

  threshold: float = Field(default=0.5, ge=0.0, le=1.0)
  """For a :class:`Binary`, the probability at or above which the route is
  ``True``. Setting it for a :class:`Choice` is an error."""

  answer_key: str | None = None
  """If set, the state key the full answer is stored under, including the
  probability of every label."""

  @model_validator(mode='after')
  def _check_threshold_applies(self) -> ClassifierNode:
    if 'threshold' in self.model_fields_set and isinstance(
        self.question, Choice
    ):
      raise ValueError(
          'threshold only applies to Binary questions. A Choice always routes'
          ' to the label the classifier chose.'
      )
    return self

  def _routes(self) -> set[RouteValue]:
    if isinstance(self.question, Choice):
      return set(self.question.criteria)
    return {True, False}

  def _route_for(self, answer: Answer) -> RouteValue:
    if isinstance(self.question, Choice) and isinstance(answer, ChoiceAnswer):
      if answer.choice not in self.question.criteria:
        raise WorkflowDataError(
            f'Classifier for node "{self.name}" chose "{answer.choice}",'
            f' which is not one of its labels: {sorted(self._routes())}.'
        )
      return answer.choice
    if isinstance(self.question, Binary) and isinstance(answer, BinaryAnswer):
      return answer.probability >= self.threshold
    raise WorkflowDataError(
        f'Classifier for node "{self.name}" returned a {answer.type} answer'
        f' to a {self.question.type} question.'
    )

  def _validate_outgoing_edges(self, edges: list[Edge]) -> None:
    from ...workflow._graph import DEFAULT_ROUTE

    edge_routes: set[Any] = set()
    has_unrouted_edge = False
    for edge in edges:
      if edge.route is None:
        has_unrouted_edge = True
        continue
      if isinstance(edge.route, list):
        edge_routes.update(edge.route)
      else:
        edge_routes.add(edge.route)
    if not edge_routes:
      return

    routes = self._routes()
    unknown = edge_routes - routes - {DEFAULT_ROUTE}
    if unknown:
      raise GraphValidationError(
          f'Graph validation failed. ClassifierNode "{self.name}" has edges'
          f' for routes it never emits: {sorted(map(str, unknown))}. Its'
          f' routes are {sorted(routes)}.'
      )
    missing = routes - edge_routes
    # An unrouted edge always fires, so a missing label still continues there.
    if missing and DEFAULT_ROUTE not in edge_routes and not has_unrouted_edge:
      raise GraphValidationError(
          f'Graph validation failed. ClassifierNode "{self.name}" has no edge'
          f' for routes {sorted(missing)}, so the branch would end when the'
          ' classifier picks one. Add an edge for each, or a DEFAULT_ROUTE'
          ' or unrouted edge.'
      )

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if node_input is None:
      raise WorkflowDataError(
          f'ClassifierNode "{self.name}" received no input to classify.'
      )
    classifier_input = node_input
    if isinstance(node_input, BaseModel) and not isinstance(
        node_input, types.Content
    ):
      classifier_input = node_input.model_dump(mode='json')
    response = await self.classifier.classify(
        input=classifier_input, questions={_QUESTION_NAME: self.question}
    )
    answer = response.answers.get(_QUESTION_NAME)
    if answer is None:
      raise WorkflowDataError(
          f'Classifier for node "{self.name}" returned no answer.'
      )
    route = self._route_for(answer)
    if self.answer_key is not None:
      ctx.state[self.answer_key] = answer.model_dump()
    yield Event(
        output=node_input,
        route=route,
        state=dict(ctx.actions.state_delta) or None,
    )
