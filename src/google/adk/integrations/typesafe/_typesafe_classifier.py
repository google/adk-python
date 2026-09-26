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

"""TypeSafe AI integration for the Jev System One model."""

from __future__ import annotations

from functools import cached_property
from typing import Mapping
from typing import Optional
from typing import Union

try:
  from typesafe_sdk import AsyncTypeSafeClient
  from typesafe_sdk import Choice
  from typesafe_sdk import JSONContent
  from typesafe_sdk import Noul
  from typesafe_sdk import Score
  from typesafe_sdk import SystemOneResponse
except ImportError as e:
  raise ImportError(
      "TypeSafe AI integration requires the typesafe-sdk package. Install it"
      " with: pip install 'google-adk[typesafe]'"
  ) from e

__all__ = [
    "TypesafeClassifier",
    "Choice",
    "Noul",
    "Score",
]

Question = Union[Noul, Choice, Score]


class TypesafeClassifier:
  """Runs typed judgments (Noul/Choice/Score) against TypeSafe's Jev model.

  This wraps ``typesafe_sdk``'s ``system_one`` call: given a piece of state and
  a set of caller-defined questions, it returns typed answers with calibrated
  probabilities. Unlike a chat model, Jev never generates free text -- the
  question's possible answers (its ``criteria``) are fixed by the caller ahead
  of time. See https://docs.typesafe.ai/concepts/system-one for background.

  Attributes:
    model: The TypeSafe model name to use, e.g. "jev-latest". Falls back to
      the SDK's own default when unset.
  """

  def __init__(
      self,
      *,
      client: Optional[AsyncTypeSafeClient] = None,
      api_key: Optional[str] = None,
      model: Optional[str] = None,
  ):
    """Initializes the classifier.

    Args:
      client: An optional pre-configured ``AsyncTypeSafeClient``. Takes
        precedence over ``api_key``/``model`` when given.
      api_key: TypeSafe API key. Defaults to the ``TYPESAFE_API_KEY``
        environment variable, as read by the SDK.
      model: The TypeSafe model name to use, e.g. "jev-latest".
    """
    self._client = client
    self._api_key = api_key
    self.model = model

  @cached_property
  def _typesafe_client(self) -> AsyncTypeSafeClient:
    if self._client is not None:
      return self._client
    return AsyncTypeSafeClient(api_key=self._api_key, model=self.model)

  async def classify(
      self,
      *,
      state: JSONContent,
      questions: Mapping[str, Question],
  ) -> SystemOneResponse:
    """Asks Jev the given questions about ``state``.

    Args:
      state: The context to evaluate -- a string, JSON object, or array.
      questions: Question id to Noul/Choice/Score definition. IDs are for
        this call only and are not sent to the model.

    Returns:
      The response, keyed by question id in ``response.answers``.
    """
    return await self._typesafe_client.system_one(
        state=state, questions=questions
    )
