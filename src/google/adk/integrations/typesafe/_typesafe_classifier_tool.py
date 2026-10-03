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

"""Exposes a TypeSafe judgment as a callable ADK tool."""

from __future__ import annotations

from typing import Optional

from google.genai import types
from typing_extensions import override

from ...tools.function_tool import FunctionTool
from ._typesafe_classifier import Question
from ._typesafe_classifier import TypesafeClassifier

__all__ = ["TypesafeClassifierTool"]


class TypesafeClassifierTool(FunctionTool):
  """An ADK tool that runs one fixed TypeSafe question against agent-supplied text.

  The question (its instructions and criteria) is fixed at construction time,
  matching how TypeSafe expects a judgment to be defined -- the agent supplies
  the text to evaluate as the tool's ``state`` argument, not the question
  itself.

  Example::

      from typesafe_sdk import Choice
      from google.adk.integrations.typesafe import TypesafeClassifier
      from google.adk.integrations.typesafe import TypesafeClassifierTool

      classify_ticket = TypesafeClassifierTool(
          name="classify_ticket_category",
          description="Classifies a support ticket into billing/technical/other.",
          classifier=TypesafeClassifier(),
          question=Choice(
              instructions="What is this support ticket about?",
              criteria={"billing": None, "technical": None, "other": None},
          ),
      )
  """

  def __init__(
      self,
      *,
      name: str,
      description: str,
      classifier: TypesafeClassifier,
      question: Question,
  ):
    """Initializes the tool.

    Args:
      name: The tool name the agent sees.
      description: The tool description the agent sees.
      classifier: The classifier used to reach TypeSafe's Jev model.
      question: The fixed Noul/Choice/Score question to ask about the text the
        agent passes in.
    """
    self._classifier = classifier
    self._question = question
    super().__init__(self._classify)
    self.name = name
    self.description = description

  @override
  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    # FunctionTool builds the declaration from self.func.__name__, which is
    # always "_classify" here. Patch in the name/description set above so the
    # model is shown the same name ADK dispatches on, and the description the
    # caller actually configured.
    declaration = super()._get_declaration()
    if declaration is not None:
      declaration.name = self.name
      declaration.description = self.description
    return declaration

  async def _classify(self, state: str) -> dict:
    """Runs the configured TypeSafe question against ``state``.

    Args:
      state: The text to evaluate.
    """
    response = await self._classifier.classify(
        state=state, questions={"result": self._question}
    )
    return response.answers["result"].model_dump()
