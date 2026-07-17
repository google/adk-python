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

from google.adk.evaluation.simulation._cloud_tts_llm import _CloudTTSLlm
from google.adk.models.llm_request import LlmRequest
import google.api_core.exceptions
from google.genai import types as genai_types
import pytest


def _text_request(*texts: str) -> LlmRequest:
  """Builds an LlmRequest whose single Content carries the given text parts."""
  return LlmRequest(
      contents=[
          genai_types.Content(
              role="user",
              parts=[genai_types.Part(text=t) for t in texts],
          )
      ]
  )


# ---------------------------------------------------------------------------
# Model fields / metadata
# ---------------------------------------------------------------------------


def test_default_fields():
  """The TTS-specific fields carry the documented defaults."""
  llm = _CloudTTSLlm(model="cloud_tts")
  assert llm.audio_encoding == "LINEAR16"
  assert llm.speaking_speed == 1.0
  assert llm.pitch == 0.0


def test_supported_models():
  """supported_models advertises the `cloud_tts` registry key."""
  assert _CloudTTSLlm.supported_models() == [r"cloud_tts"]


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------


def test_extract_text_joins_parts():
  """Text parts across the request are joined with spaces."""
  request = _text_request("Hello", "world")
  assert _CloudTTSLlm._extract_text(request) == "Hello world"


def test_extract_text_ignores_non_text_parts():
  """Parts without text are skipped when extracting text."""
  request = LlmRequest(
      contents=[
          genai_types.Content(
              role="user",
              parts=[
                  genai_types.Part(text="say this"),
                  genai_types.Part(
                      inline_data=genai_types.Blob(
                          mime_type="audio/pcm", data=b"x"
                      )
                  ),
              ],
          )
      ]
  )
  assert _CloudTTSLlm._extract_text(request) == "say this"


def test_extract_text_raises_without_text():
  """A request with no text parts raises a ValueError."""
  request = LlmRequest(contents=[genai_types.Content(role="user", parts=[])])
  with pytest.raises(
      ValueError, match="_CloudTTSLlm requires text in LlmRequest.contents"
  ):
    _CloudTTSLlm._extract_text(request)


# ---------------------------------------------------------------------------
# _extract_voice_config
# ---------------------------------------------------------------------------


def test_extract_voice_config_defaults():
  """Without speech_config, the documented defaults are returned."""
  request = _text_request("hi")
  voice_name, language_code = _CloudTTSLlm._extract_voice_config(request)
  assert voice_name == "en-US-Studio-O"
  assert language_code == "en-US"


def test_extract_voice_config_reads_speech_config():
  """voice_name and language_code are read from speech_config when present."""
  request = LlmRequest(
      contents=[
          genai_types.Content(
              role="user", parts=[genai_types.Part(text="bonjour")]
          )
      ],
      config=genai_types.GenerateContentConfig(
          speech_config=genai_types.SpeechConfig(
              language_code="fr-FR",
              voice_config=genai_types.VoiceConfig(
                  prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                      voice_name="fr-FR-Neural2-A"
                  )
              ),
          )
      ),
  )
  voice_name, language_code = _CloudTTSLlm._extract_voice_config(request)
  assert voice_name == "fr-FR-Neural2-A"
  assert language_code == "fr-FR"


# ---------------------------------------------------------------------------
# generate_content_async
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tts_modules(mocker):
  """Stubs the lazily-imported Cloud TTS client library.

  Returns the mock ``cloud_tts`` types module so tests can configure it.
  """
  mock_cloud_tts = mocker.MagicMock()
  tts_module = mocker.MagicMock()
  types_module = mocker.MagicMock()
  types_module.cloud_tts = mock_cloud_tts
  mocker.patch.dict(
      "sys.modules",
      {
          "google.cloud.texttospeech_v1": tts_module,
          "google.cloud.texttospeech_v1.types": types_module,
      },
  )
  return mock_cloud_tts


class TestGenerateContentAsync:
  """Test cases for _CloudTTSLlm.generate_content_async."""

  @pytest.mark.asyncio
  async def test_success_returns_audio(self, mock_tts_modules, mocker):
    """A successful synthesis yields a single LlmResponse with audio bytes."""
    llm = _CloudTTSLlm(model="cloud_tts")
    tts_response = mocker.MagicMock()
    tts_response.audio_content = b"AUDIO_BYTES"
    # Pre-set the client to bypass lazy client construction.
    llm._tts_client = mocker.MagicMock()
    llm._tts_client.synthesize_speech = mocker.AsyncMock(
        return_value=tts_response
    )

    responses = [
        r async for r in llm.generate_content_async(_text_request("hello"))
    ]

    assert len(responses) == 1
    part = responses[0].content.parts[0]
    assert part.inline_data.data == b"AUDIO_BYTES"
    # LINEAR16 (default) maps to audio/l16.
    assert part.inline_data.mime_type == "audio/l16"
    llm._tts_client.synthesize_speech.assert_awaited_once()

  @pytest.mark.asyncio
  async def test_mp3_encoding_mime_type(self, mock_tts_modules, mocker):
    """The MP3 encoding maps to the audio/mpeg MIME type."""
    llm = _CloudTTSLlm(model="cloud_tts", audio_encoding="MP3")
    tts_response = mocker.MagicMock()
    tts_response.audio_content = b"MP3DATA"
    llm._tts_client = mocker.MagicMock()
    llm._tts_client.synthesize_speech = mocker.AsyncMock(
        return_value=tts_response
    )

    responses = [
        r async for r in llm.generate_content_async(_text_request("hi"))
    ]

    assert responses[0].content.parts[0].inline_data.mime_type == "audio/mpeg"

  @pytest.mark.asyncio
  async def test_api_error_yields_error_response(
      self, mock_tts_modules, mocker
  ):
    """A Cloud TTS API error is surfaced as an error LlmResponse (not raised)."""
    llm = _CloudTTSLlm(model="cloud_tts")
    llm._tts_client = mocker.MagicMock()
    llm._tts_client.synthesize_speech = mocker.AsyncMock(
        side_effect=google.api_core.exceptions.GoogleAPICallError("boom")
    )

    responses = [
        r async for r in llm.generate_content_async(_text_request("hi"))
    ]

    assert len(responses) == 1
    assert responses[0].error_code == "TTS_SYNTHESIS_FAILED"
    assert "boom" in responses[0].error_message
    assert responses[0].content is None

  @pytest.mark.asyncio
  async def test_unsupported_encoding_raises(self, mock_tts_modules, mocker):
    """An unknown audio_encoding raises a ValueError before any API call."""
    mock_tts_modules.AudioEncoding.__getitem__.side_effect = KeyError("BADENC")
    llm = _CloudTTSLlm(model="cloud_tts", audio_encoding="BADENC")
    llm._tts_client = mocker.MagicMock()

    with pytest.raises(ValueError, match="Unsupported audio_encoding"):
      _ = [r async for r in llm.generate_content_async(_text_request("hi"))]
