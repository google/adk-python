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

"""Question and answer types shared by every classifier."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Annotated
from typing import Literal
from typing import Union

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from typing_extensions import TypeAlias

JSONContent: TypeAlias = Union[
    str, Mapping[str, JsonValue], Sequence[JsonValue]
]
"""Text, or a JSON-serializable mapping or sequence."""

ClassifierInput: TypeAlias = Union[JSONContent, types.Content]
"""The input a classifier answers questions about.

Text and JSON are supported by every classifier. A ``types.Content`` carries
multimodal parts, such as images, for classifiers whose model accepts them.
"""

_Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]


class Choice(BaseModel):
  """Picks one of N labels."""

  model_config = ConfigDict(extra='forbid')

  type: Literal['choice'] = 'choice'

  instructions: JSONContent | None = None
  """What to decide, for example "Which department should handle this?"."""

  criteria: Mapping[str, JSONContent | None] = Field(min_length=2)
  """Maps each label to an optional description of when it applies."""


class ChoiceAnswer(BaseModel):
  """The answer to a :class:`Choice` question."""

  model_config = ConfigDict(extra='forbid')

  type: Literal['choice'] = 'choice'

  choice: str
  """The most likely label."""

  probabilities: dict[str, _Probability]
  """The probability of each label. Sums to 1."""

  confidence: _Probability | None = None
  """The provider's confidence in the answer, from 0 to 1, when it reports one."""


class Binary(BaseModel):
  """Asks for the calibrated probability that a statement is true."""

  model_config = ConfigDict(extra='forbid')

  type: Literal['binary'] = 'binary'

  instructions: JSONContent | None = None
  """The yes/no question, for example "Does this action delete user data?"."""

  criteria: JSONContent | None = None
  """Optional description of what counts as true. Anything else is false."""


class BinaryAnswer(BaseModel):
  """The answer to a :class:`Binary` question."""

  model_config = ConfigDict(extra='forbid')

  type: Literal['binary'] = 'binary'

  probability: _Probability
  """The probability that the answer is true."""

  confidence: _Probability | None = None
  """The provider's confidence in the answer, from 0 to 1, when it reports one."""


class Score(BaseModel):
  """Places the input on an ordered rubric."""

  model_config = ConfigDict(extra='forbid')

  type: Literal['score'] = 'score'

  instructions: JSONContent | None = None
  """What to rate, for example "How urgent is this ticket?"."""

  criteria: Sequence[JSONContent] = Field(min_length=2)
  """One description per level, starting at level 0."""


class ScoreAnswer(BaseModel):
  """The answer to a :class:`Score` question."""

  model_config = ConfigDict(extra='forbid')

  type: Literal['score'] = 'score'

  score: float
  """The expected level, weighted by :attr:`probabilities`."""

  probabilities: dict[int, _Probability]
  """The probability of each level. Sums to 1."""

  confidence: _Probability | None = None
  """The provider's confidence in the answer, from 0 to 1, when it reports one."""


Question: TypeAlias = Annotated[
    Union[Choice, Binary, Score], Field(discriminator='type')
]

Answer: TypeAlias = Annotated[
    Union[ChoiceAnswer, BinaryAnswer, ScoreAnswer],
    Field(discriminator='type'),
]


class ClassifierResponse(BaseModel):
  """The answers to one :meth:`BaseClassifier.classify` call."""

  model_config = ConfigDict(extra='forbid')

  answers: dict[str, Answer]
  """Maps each question name to its answer."""

  model_version: str | None = None
  """The model version that produced the answers, when the provider reports it."""
