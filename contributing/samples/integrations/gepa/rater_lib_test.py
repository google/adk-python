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
import sys
from unittest import mock

# Mock the retry module before importing rater_lib
mock_retry_module = mock.MagicMock()


def mock_retry_decorator(*args, **kwargs):
  def decorator(func):
    return func

  return decorator


mock_retry_module.retry = mock_retry_decorator
sys.modules["retry"] = mock_retry_module

from gepa import rater_lib


def test_rater_escapes_html_inputs_to_prevent_xss():
  """Rater escapes HTML tags in user and model inputs to prevent XSS.

  Setup: Mock genai.Client to return a dummy rating response.
  Act: Call Rater with messages containing HTML tags.
  Assert: Verify that the rendered prompt template contains escaped HTML.
  """
  # Arrange
  with mock.patch("google.genai.Client") as mock_client_cls:
    mock_client = mock_client_cls.return_value
    mock_generate = mock_client.models.generate_content
    mock_generate.return_value.text = (
        "Property: The agent fulfilled the user's primary request.\n"
        "Evidence: mock evidence\n"
        "Rationale: mock rationale\n"
        "Verdict: yes"
    )

    template_path = (
        "contributing/samples/integrations/gepa/rubric_validation_template.txt"
    )
    rater = rater_lib.Rater(
        tool_declarations="[]", validation_template_path=template_path
    )

    messages = [
        {
            "role": "user",
            "parts": [{"text": "<script>alert('XSS')</script>"}],
        },
        {
            "role": "model",
            "parts": [{"text": "Hello <img src=x onerror=alert(1)>"}],
        },
    ]

    # Act
    rater(messages)

    # Assert
    assert mock_generate.called
    call_args = mock_generate.call_args
    contents = call_args.kwargs["contents"]

    assert "&lt;script&gt;alert(&#39;XSS&#39;)&lt;/script&gt;" in contents
    assert "Hello &lt;img src=x onerror=alert(1)&gt;" in contents
