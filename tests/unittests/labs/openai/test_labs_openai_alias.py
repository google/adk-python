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

"""The old ``google.adk.labs.openai`` import path still resolves."""

from google.adk.integrations import openai as integrations_openai
from google.adk.labs import openai as labs_openai
import pytest


@pytest.mark.parametrize('name', integrations_openai.__all__)
def test_labs_openai_reexports_integrations_openai(name):
  assert getattr(labs_openai, name) is getattr(integrations_openai, name)


def test_labs_openai_exports_match_integrations_openai():
  assert sorted(labs_openai.__all__) == sorted(integrations_openai.__all__)
