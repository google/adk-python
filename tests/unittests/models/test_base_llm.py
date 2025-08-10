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

import pytest
from google.genai import types
from google.adk.models.llm_request import LlmRequest
from google.adk.models.lite_llm import _get_completion_inputs


@pytest.mark.parametrize("content_kwargs", [
    # Case 1: Explicit role provided
    {"role": "user", "parts": [types.Part(text="This is an input text from user.")]},
    # Case 2: Role omitted, should still be treated as 'user'
    {"parts": [types.Part(text="This is an input text from user.")]}
])
def test_user_content_role_defaults_to_user(content_kwargs):
    """
    Verifies that user-provided messages are always passed to the LLM as 'user' role,
    regardless of whether the role is explicitly set in types.Content.
    
    The helper `_get_completion_inputs` should give normalize messages so that
    explicit 'user' and implicit (missing role) are equivalent.
    """
    llm_request = LlmRequest(
        contents=[types.Content(**content_kwargs)],
        config=types.GenerateContentConfig()
    )

    messages, _, _, _ = _get_completion_inputs(llm_request)

    assert all(
        msg.get("role") == "user" for msg in messages
    ), f"Expected role 'user' but got {messages}"
    assert any(
        "This is an input text from user." == (msg.get("content") or "") 
        for msg in messages
    ), f"Expected the user text to be preserved, but got {messages}"