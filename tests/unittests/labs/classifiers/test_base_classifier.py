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

"""Tests for BaseClassifier."""

from __future__ import annotations

from collections.abc import Mapping

from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.labs.classifiers import BaseClassifier
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import BinaryAnswer
from google.adk.labs.classifiers import ClassifierInput
from google.adk.labs.classifiers import ClassifierResponse
from google.adk.labs.classifiers import Question
import pytest


class _ConstantClassifier(BaseClassifier):
  """Answers every question as a binary with probability 0.5."""

  model: str = 'constant'

  async def classify(
      self,
      *,
      input: ClassifierInput,
      questions: Mapping[str, Question],
  ) -> ClassifierResponse:
    return ClassifierResponse(
        model_version=self.model,
        answers={name: BinaryAnswer(probability=0.5) for name in questions},
    )


async def test_subclass_answers_through_classify():
  response = await _ConstantClassifier().classify(
      input='x', questions={'risky': Binary()}
  )

  assert response == ClassifierResponse(
      model_version='constant', answers={'risky': BinaryAnswer(probability=0.5)}
  )


def test_base_classifier_cannot_be_instantiated():
  with pytest.raises(TypeError):
    BaseClassifier(model='m')


def test_classifier_is_disabled_with_its_feature():
  with temporary_feature_override(FeatureName.CLASSIFIER, False):
    with pytest.raises(RuntimeError, match='CLASSIFIER'):
      _ConstantClassifier()
