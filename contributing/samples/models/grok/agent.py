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

"""An ADK agent powered by xAI Grok 4.6 served on Vertex AI.

Grok is a partner model on Vertex AI Model Garden. It is reached through the
OpenAI-compatible Chat Completions surface (``endpoints/openapi``), so the
``OpenAILlm`` model talks to it directly:

- ``base_url`` points at the project's ``endpoints/openapi`` surface, and
- ``api_key`` is a Google Cloud access token. It is passed as a *callable* that
  returns a cached Application Default Credentials token, refreshed only when it
  expires, rather than baking a short-lived string into the model.

Enable Grok on its Model Garden card, authenticate with ADC
(``gcloud auth application-default login``), and set ``GOOGLE_CLOUD_PROJECT``.
See README.md for details.
"""

from __future__ import annotations

import functools
import os
import random

from google.adk import Agent
from google.adk.integrations.openai import OpenAILlm
from google.adk.models.base_llm import BaseLlm
from google.adk.utils._mtls_utils import get_api_endpoint
import google.auth
import google.auth.transport.requests


def _required_env(name: str) -> str:
  value = os.getenv(name)
  if not value:
    raise RuntimeError(
        f"Set {name} before starting the sample. See README.md for setup."
    )
  return value


@functools.lru_cache(maxsize=1)
def _adc_credentials():
  """Resolves Application Default Credentials once and caches them."""
  credentials, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  return credentials


def _access_token() -> str:
  """Returns a Google Cloud access token from ADC.

  Passed to ``OpenAILlm(api_key=...)`` as a callable that is invoked on every
  request. Credentials are resolved once and cached, and the token is refreshed
  only when it has expired, so requests do not re-run credential discovery or
  force a network refresh each time.
  """
  credentials = _adc_credentials()
  if not credentials.valid:
    credentials.refresh(google.auth.transport.requests.Request())
  return credentials.token


def _build_model() -> BaseLlm:
  """Builds the Grok-on-Vertex model."""
  project = _required_env("GOOGLE_CLOUD_PROJECT")
  # Grok 4.6 is served on the global endpoint only.
  base_url = get_api_endpoint(
      location="global",
      default_template=(
          f"https://aiplatform.googleapis.com/v1/projects/{project}"
          "/locations/global/endpoints/openapi"
      ),
      mtls_template=(
          f"https://aiplatform.mtls.googleapis.com/v1/projects/{project}"
          "/locations/global/endpoints/openapi"
      ),
  )
  return OpenAILlm(
      model=os.getenv("GROK_MODEL", "xai/grok-4.6"),
      base_url=base_url,
      api_key=_access_token,
  )


def roll_die(sides: int) -> int:
  """Roll a die and return the rolled result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


def check_prime(nums: list[int]) -> str:
  """Check if a given list of numbers are prime.

  Args:
    nums: The list of numbers to check.

  Returns:
    A str indicating which number is prime.
  """
  primes = set()
  for number in nums:
    number = int(number)
    if number <= 1:
      continue
    is_prime = True
    for i in range(2, int(number**0.5) + 1):
      if number % i == 0:
        is_prime = False
        break
    if is_prime:
      primes.add(number)
  return (
      "No prime numbers found."
      if not primes
      else f"{', '.join(str(num) for num in primes)} are prime numbers."
  )


root_agent = Agent(
    name="grok_agent",
    model=_build_model(),
    description=(
        "A hello-world agent powered by xAI Grok 4.6 on Vertex AI that rolls"
        " dice and checks whether numbers are prime."
    ),
    instruction="""
      You are a helpful assistant powered by Grok that can roll dice and check
      whether numbers are prime.
      When asked to roll a die, call the roll_die tool with the integer number
      of sides. Never roll a die yourself.
      When asked to check primes, call the check_prime tool with a list of
      integers. Never decide primality yourself.
      When asked to roll a die and then check the result, first call roll_die,
      wait for its result, then call check_prime with that result. Always report
      the number you rolled.
    """,
    tools=[roll_die, check_prime],
)
