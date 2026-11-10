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

"""Tests for best-effort credential redaction."""

from __future__ import annotations

from google.adk.utils._redaction import redact_secrets
from google.adk.utils._redaction import redact_text
from google.adk.utils._redaction import REDACTED
from pydantic import BaseModel


class _Nested(BaseModel):
  password: str
  label: str


def test_redact_secrets_masks_credential_keys_case_and_punctuation_insensitively():
  payload = {
      "api_key": "k",
      "API-KEY": "k2",
      "clientSecret": "s",
      "Password": "p",
      "access_token": "a",
  }

  redacted = redact_secrets(payload, include_ambiguous=True)

  assert set(redacted.values()) == {REDACTED}


def test_redact_secrets_preserves_ambiguous_keys_by_default():
  """A bare token/apiKey may be a pagination cursor in a tool response."""
  payload = {"token": "next-page-token", "apiKey": "public-id"}

  assert redact_secrets(payload) == payload


def test_redact_secrets_leaves_benign_lookalikes_alone():
  payload = {
      "prompt_token_count": 11,
      "next_page_token": "page-2",
      "author": "user",
  }

  assert redact_secrets(payload) == payload


def test_redact_secrets_recurses_into_lists_and_nested_dicts():
  payload = {"outer": [{"refreshToken": "r", "keep": 1}]}

  assert redact_secrets(payload) == {
      "outer": [{"refreshToken": REDACTED, "keep": 1}]
  }


def test_redact_secrets_dumps_pydantic_models():
  payload = {"account": _Nested(password="p", label="l")}

  assert redact_secrets(payload) == {
      "account": {"password": REDACTED, "label": "l"},
  }


def test_redact_secrets_preserves_tuples():
  redacted = redact_secrets(({"password": "t"}, "plain"))

  assert redacted == ({"password": REDACTED}, "plain")
  assert isinstance(redacted, tuple)


def test_redact_secrets_survives_cycles_without_raising():
  payload: dict[str, object] = {"keep": 1}
  payload["self"] = payload

  redacted = redact_secrets(payload)

  assert redacted["keep"] == 1


def test_redact_text_masks_bearer_tokens():
  assert redact_text("Authorization: Bearer abcdefghijkl") == (
      f"Authorization: Bearer {REDACTED}"
  )


def test_redact_text_masks_key_value_secrets():
  assert redact_text("api_key=supersecret value") == f"api_key={REDACTED} value"


def test_redact_text_leaves_ordinary_prose_unchanged():
  assert redact_text("the token counts are 11 and 7") == (
      "the token counts are 11 and 7"
  )


def test_redact_text_truncates_after_masking():
  result = redact_text("Bearer abcdefghijklmnop tail", max_length=10)

  assert result == f"Bearer {REDACTED}"[:10] + "..."
