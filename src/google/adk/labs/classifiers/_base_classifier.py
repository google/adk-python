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

"""The interface for models that answer typed questions."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pydantic import ConfigDict

from ...features import experimental
from ...features import FeatureName

if TYPE_CHECKING:
  from ._types import ClassifierInput
  from ._types import ClassifierResponse
  from ._types import Question


@experimental(FeatureName.CLASSIFIER)
class BaseClassifier(BaseModel, abc.ABC):
  """A model that answers typed questions with probability distributions.

  Unlike a ``BaseLlm``, a classifier does not generate text, stream, or call
  tools. It takes an input and a set of :class:`Choice`, :class:`Binary` and
  :class:`Score` questions, and returns a distribution over the possible
  answers to each. Dedicated decision models implement it directly;
  :class:`LlmClassifier` implements it on top of any ``BaseLlm``.

  Classifiers are stateless. Context from earlier turns has to be included in
  the input passed to :meth:`classify`.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  model: str
  """The provider's model name."""

  @abc.abstractmethod
  async def classify(
      self,
      *,
      input: ClassifierInput,
      questions: Mapping[str, Question],
  ) -> ClassifierResponse:
    """Answers every question in `questions` against `input`.

    Args:
      input: What the questions are about. Text and JSON are always accepted;
        an implementation raises ``ValueError`` for input its model does not
        support, such as images.
      questions: Maps a caller-chosen name to each question. The response uses
        the same names.

    Returns:
      One answer per question, keyed by the question's name. A question the
      model declined to answer gets a :class:`RefusalAnswer`.
    """
