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

from __future__ import annotations

from google.adk.utils.content_utils import extract_text_from_content
from google.adk.utils.content_utils import filter_audio_parts
from google.adk.utils.content_utils import is_audio_part
from google.adk.utils.content_utils import SKIP_THOUGHT_SIGNATURE_VALIDATOR
from google.adk.utils.content_utils import to_user_content
from google.genai import types
from pydantic import BaseModel


def test_skip_thought_signature_validator_wire_value():
  # The backend recognizes this exact byte string to bypass validation;
  # changing it would break every replayed synthetic part.
  assert SKIP_THOUGHT_SIGNATURE_VALIDATOR == b'skip_thought_signature_validator'


def test_skip_thought_signature_validator_assignable_to_part():
  part = types.Part(
      text='injected',
      thought_signature=SKIP_THOUGHT_SIGNATURE_VALIDATOR,
  )
  assert part.thought_signature == SKIP_THOUGHT_SIGNATURE_VALIDATOR


def test_to_user_content_str_input_becomes_user_text():
  content = to_user_content('hello')
  assert content.role == 'user'
  assert content.parts[0].text == 'hello'


def test_to_user_content_input_is_normalized_to_user_role():
  original = types.Content(role='model', parts=[types.Part(text='hi')])
  content = to_user_content(original)
  assert content.role == 'user'
  assert content.parts[0].text == 'hi'


def test_to_user_content_basemodel_input_is_json():
  class _M(BaseModel):
    a: int

  content = to_user_content(_M(a=1))
  assert content.role == 'user'
  assert '"a":1' in content.parts[0].text.replace(' ', '')


def test_to_user_content_dict_input_is_json():
  content = to_user_content({'a': 1})
  assert content.role == 'user'
  assert content.parts[0].text.replace(' ', '') == '{"a":1}'


def test_to_user_content_other_input_is_str():
  content = to_user_content(42)
  assert content.role == 'user'
  assert content.parts[0].text == '42'


def test_to_user_content_dict_input_preserves_non_ascii():
  """Non-ASCII input must reach the LLM as-is, not as \\uXXXX escapes.

  Escaping (json.dumps' default ensure_ascii=True) turns each non-Latin
  character into a ``\\uXXXX`` sequence, which bloats prompt tokens and
  degrades model responses for non-English inputs.
  """
  content = to_user_content({'query': 'שלום עולם', 'city': '北京'})
  text = content.parts[0].text
  assert 'שלום עולם' in text
  assert '北京' in text
  assert '\\u' not in text


def test_to_user_content_list_input_preserves_non_ascii():
  content = to_user_content(['שלום', '你好'])
  text = content.parts[0].text
  assert 'שלום' in text
  assert '你好' in text
  assert '\\u' not in text


def _audio_blob_part(mime_type: str) -> types.Part:
  return types.Part(
      inline_data=types.Blob(mime_type=mime_type, data=b'\x00\x01')
  )


def _audio_file_part(mime_type: str) -> types.Part:
  return types.Part(
      file_data=types.FileData(file_uri='files/clip', mime_type=mime_type)
  )


def test_is_audio_part_inline_audio_mime_is_audio():
  assert is_audio_part(_audio_blob_part('audio/pcm')) is True


def test_is_audio_part_file_data_audio_mime_is_audio():
  assert is_audio_part(_audio_file_part('audio/wav')) is True


def test_is_audio_part_non_audio_mime_is_not_audio():
  # Only the 'audio/' top-level type counts; video and image blobs must
  # survive so they still reach the model.
  assert is_audio_part(_audio_blob_part('image/png')) is False
  assert is_audio_part(_audio_file_part('video/mp4')) is False


def test_is_audio_part_mime_containing_audio_but_not_prefixed_is_not_audio():
  # The check is a prefix match on the top-level type, not a substring
  # match, so 'application/audio-ish' is not audio.
  assert is_audio_part(_audio_blob_part('application/audio-ish')) is False


def test_is_audio_part_text_part_is_not_audio():
  assert is_audio_part(types.Part(text='hello')) is False


def test_is_audio_part_blob_without_mime_type_is_not_audio():
  # An unlabelled blob cannot be proven to be audio, so it is kept.
  part = types.Part(inline_data=types.Blob(data=b'\x00\x01'))
  assert is_audio_part(part) is False


def test_filter_audio_parts_drops_audio_and_keeps_role_and_order():
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='before'),
          _audio_blob_part('audio/pcm'),
          _audio_file_part('audio/wav'),
          types.Part(text='after'),
      ],
  )

  filtered = filter_audio_parts(content)

  assert filtered is not None
  assert filtered.role == 'user'
  assert [p.text for p in filtered.parts] == ['before', 'after']


def test_filter_audio_parts_all_audio_returns_none():
  # A content whose every part is audio has nothing left to send, so the
  # caller is told to drop the whole content rather than send an empty one.
  content = types.Content(role='user', parts=[_audio_blob_part('audio/pcm')])
  assert filter_audio_parts(content) is None


def test_filter_audio_parts_empty_parts_returns_none():
  assert filter_audio_parts(types.Content(role='user', parts=[])) is None


def test_filter_audio_parts_does_not_mutate_input():
  content = types.Content(
      role='user',
      parts=[types.Part(text='keep'), _audio_blob_part('audio/pcm')],
  )

  filter_audio_parts(content)

  assert len(content.parts) == 2
  assert content.parts[1].inline_data.mime_type == 'audio/pcm'


def test_extract_text_from_content_concatenates_text_parts_verbatim():
  # Parts are joined with no separator: the model emits a single logical
  # string that is chunked arbitrarily across parts.
  content = types.Content(
      role='model',
      parts=[types.Part(text='hello '), types.Part(text='world')],
  )
  assert extract_text_from_content(content) == 'hello world'


def test_extract_text_from_content_omits_thought_parts():
  content = types.Content(
      role='model',
      parts=[
          types.Part(text='reasoning', thought=True),
          types.Part(text='answer'),
      ],
  )
  assert extract_text_from_content(content) == 'answer'


def test_extract_text_from_content_none_returns_empty_string():
  assert extract_text_from_content(None) == ''


def test_extract_text_from_content_without_text_parts_returns_empty_string():
  content = types.Content(role='user', parts=[_audio_blob_part('audio/pcm')])
  assert extract_text_from_content(content) == ''
