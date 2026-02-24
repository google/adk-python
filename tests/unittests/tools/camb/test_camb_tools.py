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

"""Unit tests for the CAMB AI toolset with mocked CAMB SDK."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from types import SimpleNamespace
from typing import Any
from unittest import mock

from google.adk.tools.camb._helpers import CambHelpers
from google.adk.tools.camb._tools import make_audio_separation_func
from google.adk.tools.camb._tools import make_clone_voice_func
from google.adk.tools.camb._tools import make_list_voices_func
from google.adk.tools.camb._tools import make_text_to_sound_func
from google.adk.tools.camb._tools import make_transcribe_func
from google.adk.tools.camb._tools import make_translate_func
from google.adk.tools.camb._tools import make_translated_tts_func
from google.adk.tools.camb._tools import make_tts_func
from google.adk.tools.camb.camb_toolset import CambAIToolset
from google.adk.tools.function_tool import FunctionTool
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
  """Ensure CAMB_API_KEY is always available for tests."""
  monkeypatch.setenv("CAMB_API_KEY", "test-api-key-123")


def _make_helpers() -> CambHelpers:
  """Create a CambHelpers instance for testing."""
  return CambHelpers(
      api_key="test-api-key",
      timeout=10.0,
      max_poll_attempts=3,
      poll_interval=0.01,
  )


# ---------------------------------------------------------------------------
# CambHelpers tests
# ---------------------------------------------------------------------------


class TestCambHelpers:
  """Tests for the CambHelpers utility class."""

  def test_init_with_explicit_key(self):
    h = CambHelpers(api_key="my-key")
    assert h._api_key == "my-key"

  def test_init_from_env(self):
    h = CambHelpers()
    assert h._api_key == "test-api-key-123"

  def test_init_missing_key_raises(self, monkeypatch):
    monkeypatch.delenv("CAMB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CAMB_API_KEY not set"):
      CambHelpers()

  def test_detect_audio_format_wav(self):
    assert CambHelpers.detect_audio_format(b"RIFF....") == "wav"

  def test_detect_audio_format_mp3_sync(self):
    assert CambHelpers.detect_audio_format(b"\xff\xfb\x90\x00") == "mp3"

  def test_detect_audio_format_mp3_id3(self):
    assert CambHelpers.detect_audio_format(b"ID3....") == "mp3"

  def test_detect_audio_format_flac(self):
    assert CambHelpers.detect_audio_format(b"fLaC....") == "flac"

  def test_detect_audio_format_ogg(self):
    assert CambHelpers.detect_audio_format(b"OggS....") == "ogg"

  def test_detect_audio_format_from_content_type(self):
    assert CambHelpers.detect_audio_format(b"\x00\x00", "audio/mpeg") == "mp3"

  def test_detect_audio_format_pcm_fallback(self):
    assert CambHelpers.detect_audio_format(b"\x00\x00") == "pcm"

  def test_add_wav_header(self):
    pcm = b"\x00" * 100
    wav = CambHelpers.add_wav_header(pcm)
    assert wav.startswith(b"RIFF")
    assert wav[8:12] == b"WAVE"
    # Total size should be 44-byte header + pcm data
    assert len(wav) == 44 + 100

  def test_save_audio(self):
    path = CambHelpers.save_audio(b"fake-audio-data", ".wav")
    try:
      assert path.endswith(".wav")
      with open(path, "rb") as f:
        assert f.read() == b"fake-audio-data"
    finally:
      os.unlink(path)

  def test_gender_str(self):
    assert CambHelpers.gender_str(0) == "not_specified"
    assert CambHelpers.gender_str(1) == "male"
    assert CambHelpers.gender_str(2) == "female"
    assert CambHelpers.gender_str(9) == "not_applicable"
    assert CambHelpers.gender_str(99) == "unknown"

  def test_format_transcription(self):
    transcription = SimpleNamespace(
        text="Hello world",
        segments=[
            SimpleNamespace(start=0.0, end=1.0, text="Hello", speaker="A"),
            SimpleNamespace(start=1.0, end=2.0, text="world", speaker="B"),
        ],
        speakers=["A", "B"],
    )
    result = json.loads(CambHelpers.format_transcription(transcription))
    assert result["text"] == "Hello world"
    assert len(result["segments"]) == 2
    assert result["speakers"] == ["A", "B"]

  def test_format_voices(self):
    h = _make_helpers()
    voices = [
        SimpleNamespace(
            id=1, voice_name="Test Voice", gender=1, age=30, language=1
        ),
        {"id": 2, "voice_name": "Dict Voice", "gender": 2, "age": 25},
    ]
    result = json.loads(h.format_voices(voices))
    assert len(result) == 2
    assert result[0]["name"] == "Test Voice"
    assert result[0]["gender"] == "male"
    assert result[1]["name"] == "Dict Voice"
    assert result[1]["gender"] == "female"

  def test_format_separation(self):
    h = _make_helpers()
    sep = SimpleNamespace(
        vocals_url="https://example.com/vocals.wav",
        background_url="https://example.com/bg.wav",
    )
    result = json.loads(h.format_separation(sep))
    assert result["vocals"] == "https://example.com/vocals.wav"
    assert result["background"] == "https://example.com/bg.wav"
    assert result["status"] == "completed"

  def test_extract_translation_string(self):
    assert CambHelpers.extract_translation("Hello") == "Hello"

  def test_extract_translation_object_with_text(self):
    obj = SimpleNamespace(text="Bonjour")
    assert CambHelpers.extract_translation(obj) == "Bonjour"

  def test_extract_translation_iterable(self):
    chunks = [
        SimpleNamespace(text="Bon"),
        SimpleNamespace(text="jour"),
    ]
    assert CambHelpers.extract_translation(chunks) == "Bonjour"


# ---------------------------------------------------------------------------
# Polling tests
# ---------------------------------------------------------------------------


class TestPolling:
  """Tests for the async polling helper."""

  @pytest.mark.asyncio
  async def test_poll_success(self):
    h = _make_helpers()
    status = SimpleNamespace(status="completed", run_id="run-1")

    async def get_status(task_id, *, run_id=None):
      return status

    result = await h.poll_async(get_status, "task-1")
    assert result.run_id == "run-1"

  @pytest.mark.asyncio
  async def test_poll_failure(self):
    h = _make_helpers()
    status = SimpleNamespace(status="failed", error="boom")

    async def get_status(task_id, *, run_id=None):
      return status

    with pytest.raises(RuntimeError, match="Task failed"):
      await h.poll_async(get_status, "task-1")

  @pytest.mark.asyncio
  async def test_poll_timeout(self):
    h = CambHelpers(api_key="key", max_poll_attempts=2, poll_interval=0.01)
    status = SimpleNamespace(status="pending")

    async def get_status(task_id, *, run_id=None):
      return status

    with pytest.raises(TimeoutError):
      await h.poll_async(get_status, "task-1")


# ---------------------------------------------------------------------------
# Toolset tests
# ---------------------------------------------------------------------------


class TestCambAIToolset:
  """Tests for the CambAIToolset class."""

  @pytest.mark.asyncio
  async def test_default_tools_count(self):
    toolset = CambAIToolset(api_key="test-key")
    tools = await toolset.get_tools()
    assert len(tools) == 8
    assert all(isinstance(t, FunctionTool) for t in tools)

  @pytest.mark.asyncio
  async def test_all_tool_names(self):
    toolset = CambAIToolset(api_key="test-key")
    tools = await toolset.get_tools()
    names = {t.name for t in tools}
    expected = {
        "camb_tts",
        "camb_translate",
        "camb_transcribe",
        "camb_translated_tts",
        "camb_clone_voice",
        "camb_list_voices",
        "camb_text_to_sound",
        "camb_audio_separation",
    }
    assert names == expected

  @pytest.mark.asyncio
  async def test_include_flags_disable_all(self):
    toolset = CambAIToolset(
        api_key="test-key",
        include_tts=False,
        include_translation=False,
        include_transcription=False,
        include_translated_tts=False,
        include_voice_clone=False,
        include_voice_list=False,
        include_text_to_sound=False,
        include_audio_separation=False,
    )
    tools = await toolset.get_tools()
    assert len(tools) == 0

  @pytest.mark.asyncio
  async def test_include_only_tts(self):
    toolset = CambAIToolset(
        api_key="test-key",
        include_tts=True,
        include_translation=False,
        include_transcription=False,
        include_translated_tts=False,
        include_voice_clone=False,
        include_voice_list=False,
        include_text_to_sound=False,
        include_audio_separation=False,
    )
    tools = await toolset.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "camb_tts"

  @pytest.mark.asyncio
  async def test_tool_filter_by_name(self):
    toolset = CambAIToolset(
        api_key="test-key",
        tool_filter=["camb_tts", "camb_translate"],
    )
    tools = await toolset.get_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"camb_tts", "camb_translate"}

  @pytest.mark.asyncio
  async def test_tool_filter_unknown_name(self):
    toolset = CambAIToolset(
        api_key="test-key",
        tool_filter=["nonexistent_tool"],
    )
    tools = await toolset.get_tools()
    assert len(tools) == 0

  @pytest.mark.asyncio
  async def test_tool_filter_mixed(self):
    toolset = CambAIToolset(
        api_key="test-key",
        tool_filter=["camb_tts", "nonexistent"],
    )
    tools = await toolset.get_tools()
    assert len(tools) == 1
    assert tools[0].name == "camb_tts"

  @pytest.mark.asyncio
  async def test_close(self):
    toolset = CambAIToolset(api_key="test-key")
    # Should not raise
    await toolset.close()

  def test_missing_api_key_raises(self, monkeypatch):
    monkeypatch.delenv("CAMB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="CAMB_API_KEY not set"):
      CambAIToolset()


# ---------------------------------------------------------------------------
# Tool function tests (with mocked CAMB SDK)
# ---------------------------------------------------------------------------


class TestToolFunctions:
  """Tests for individual CAMB tool functions with mocked SDK client."""

  def _mock_client(self) -> mock.MagicMock:
    """Create a fully mocked AsyncCambAI client."""
    return mock.MagicMock()

  def _helpers_with_mock_client(self) -> tuple[CambHelpers, mock.MagicMock]:
    """Create helpers with a pre-injected mock client."""
    h = _make_helpers()
    client = self._mock_client()
    h._client = client
    return h, client

  @pytest.mark.asyncio
  async def test_camb_tts(self):
    h, client = self._helpers_with_mock_client()
    audio_data = b"RIFF" + b"\x00" * 100

    async def mock_tts(**kwargs):
      yield audio_data

    client.text_to_speech.tts = mock_tts

    # Mock the StreamTtsOutputConfiguration import
    mock_config = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"camb": mock.MagicMock(StreamTtsOutputConfiguration=mock_config)},
    ):
      func = make_tts_func(h)
      result = await func(text="Hello world")

    assert result.endswith(".wav")
    with open(result, "rb") as f:
      assert f.read() == audio_data
    os.unlink(result)

  @pytest.mark.asyncio
  async def test_camb_translate(self):
    h, client = self._helpers_with_mock_client()

    async def mock_translate(**kwargs):
      return [SimpleNamespace(text="Hola mundo")]

    client.translation.translation_stream = mock_translate

    mock_api_error = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {
            "camb": mock.MagicMock(),
            "camb.core": mock.MagicMock(),
            "camb.core.api_error": mock.MagicMock(ApiError=mock_api_error),
        },
    ):
      func = make_translate_func(h)
      result = await func(
          text="Hello world", source_language=1, target_language=2
      )

    assert result == "Hola mundo"

  @pytest.mark.asyncio
  async def test_camb_transcribe_with_file(self):
    h, client = self._helpers_with_mock_client()

    task_result = SimpleNamespace(task_id="t-1")
    status_result = SimpleNamespace(status="completed", run_id="r-1")
    transcription_result = SimpleNamespace(
        text="Hello",
        segments=[
            SimpleNamespace(start=0.0, end=1.0, text="Hello", speaker="A")
        ],
        speakers=["A"],
    )

    client.transcription.create_transcription = mock.AsyncMock(
        return_value=task_result
    )
    client.transcription.get_transcription_task_status = mock.AsyncMock(
        return_value=status_result
    )
    client.transcription.get_transcription_result = mock.AsyncMock(
        return_value=transcription_result
    )

    # Create a temp audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
      tmp.write(b"fake audio")
      tmp_path = tmp.name

    try:
      func = make_transcribe_func(h)
      result = await func(language=1, audio_file_path=tmp_path)
      parsed = json.loads(result)
      assert parsed["text"] == "Hello"
      assert len(parsed["segments"]) == 1
    finally:
      os.unlink(tmp_path)

  @pytest.mark.asyncio
  async def test_camb_transcribe_no_input(self):
    h, _client = self._helpers_with_mock_client()
    func = make_transcribe_func(h)
    with pytest.raises(
        ValueError, match="Provide either audio_url or audio_file_path"
    ):
      await func(language=1)

  @pytest.mark.asyncio
  async def test_camb_clone_voice(self):
    h, client = self._helpers_with_mock_client()

    clone_result = SimpleNamespace(voice_id=42, message="Voice created")
    client.voice_cloning.create_custom_voice = mock.AsyncMock(
        return_value=clone_result
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
      tmp.write(b"fake audio sample")
      tmp_path = tmp.name

    try:
      func = make_clone_voice_func(h)
      result = await func(
          voice_name="TestVoice",
          audio_file_path=tmp_path,
          gender=1,
          description="A test voice",
      )
      parsed = json.loads(result)
      assert parsed["voice_id"] == 42
      assert parsed["voice_name"] == "TestVoice"
      assert parsed["status"] == "created"
    finally:
      os.unlink(tmp_path)

  @pytest.mark.asyncio
  async def test_camb_list_voices(self):
    h, client = self._helpers_with_mock_client()

    voices = [
        SimpleNamespace(
            id=1, voice_name="Voice1", gender=1, age=30, language=1
        ),
        SimpleNamespace(
            id=2, voice_name="Voice2", gender=2, age=25, language=2
        ),
    ]
    client.voice_cloning.list_voices = mock.AsyncMock(return_value=voices)

    func = make_list_voices_func(h)
    result = await func()
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "Voice1"
    assert parsed[1]["name"] == "Voice2"

  @pytest.mark.asyncio
  async def test_camb_text_to_sound(self):
    h, client = self._helpers_with_mock_client()
    audio_data = b"RIFF" + b"\x00" * 50

    task_result = SimpleNamespace(task_id="t-1")
    status_result = SimpleNamespace(status="completed", run_id="r-1")

    client.text_to_audio.create_text_to_audio = mock.AsyncMock(
        return_value=task_result
    )
    client.text_to_audio.get_text_to_audio_status = mock.AsyncMock(
        return_value=status_result
    )

    async def mock_get_result(run_id):
      yield audio_data

    client.text_to_audio.get_text_to_audio_result = mock_get_result

    func = make_text_to_sound_func(h)
    result = await func(prompt="birds chirping")
    assert result.endswith(".wav")
    with open(result, "rb") as f:
      assert f.read() == audio_data
    os.unlink(result)

  @pytest.mark.asyncio
  async def test_camb_audio_separation_with_file(self):
    h, client = self._helpers_with_mock_client()

    task_result = SimpleNamespace(task_id="t-1")
    status_result = SimpleNamespace(status="completed", run_id="r-1")
    sep_result = SimpleNamespace(
        vocals_url="https://example.com/vocals.wav",
        background_url="https://example.com/bg.wav",
    )

    client.audio_separation.create_audio_separation = mock.AsyncMock(
        return_value=task_result
    )
    client.audio_separation.get_audio_separation_status = mock.AsyncMock(
        return_value=status_result
    )
    client.audio_separation.get_audio_separation_run_info = mock.AsyncMock(
        return_value=sep_result
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
      tmp.write(b"fake audio")
      tmp_path = tmp.name

    try:
      func = make_audio_separation_func(h)
      result = await func(audio_file_path=tmp_path)
      parsed = json.loads(result)
      assert parsed["vocals"] == "https://example.com/vocals.wav"
      assert parsed["background"] == "https://example.com/bg.wav"
      assert parsed["status"] == "completed"
    finally:
      os.unlink(tmp_path)

  @pytest.mark.asyncio
  async def test_camb_audio_separation_no_input(self):
    h, _client = self._helpers_with_mock_client()
    func = make_audio_separation_func(h)
    with pytest.raises(
        ValueError, match="Provide either audio_url or audio_file_path"
    ):
      await func()

  @pytest.mark.asyncio
  async def test_camb_transcribe_url_cleans_up_temp_file(self):
    h, client = self._helpers_with_mock_client()

    task_result = SimpleNamespace(task_id="t-1")
    status_result = SimpleNamespace(status="completed", run_id="r-1")
    transcription_result = SimpleNamespace(
        text="Hello", segments=[], speakers=[]
    )

    client.transcription.create_transcription = mock.AsyncMock(
        return_value=task_result
    )
    client.transcription.get_transcription_task_status = mock.AsyncMock(
        return_value=status_result
    )
    client.transcription.get_transcription_result = mock.AsyncMock(
        return_value=transcription_result
    )

    mock_resp = mock.MagicMock()
    mock_resp.content = b"fake audio data"
    mock_resp.raise_for_status = mock.MagicMock()

    with (
        mock.patch("google.adk.tools.camb._tools.os.unlink") as mock_unlink,
        mock.patch("httpx.AsyncClient") as mock_httpx,
    ):
      mock_http_instance = mock.AsyncMock()
      mock_http_instance.get = mock.AsyncMock(return_value=mock_resp)
      mock_http_instance.__aenter__ = mock.AsyncMock(
          return_value=mock_http_instance
      )
      mock_http_instance.__aexit__ = mock.AsyncMock(return_value=False)
      mock_httpx.return_value = mock_http_instance

      func = make_transcribe_func(h)
      await func(language=1, audio_url="https://example.com/audio.wav")

      mock_unlink.assert_called_once()

  @pytest.mark.asyncio
  async def test_camb_audio_separation_url_cleans_up_temp_file(self):
    h, client = self._helpers_with_mock_client()

    task_result = SimpleNamespace(task_id="t-1")
    status_result = SimpleNamespace(status="completed", run_id="r-1")
    sep_result = SimpleNamespace(
        vocals_url="https://example.com/vocals.wav",
        background_url="https://example.com/bg.wav",
    )

    client.audio_separation.create_audio_separation = mock.AsyncMock(
        return_value=task_result
    )
    client.audio_separation.get_audio_separation_status = mock.AsyncMock(
        return_value=status_result
    )
    client.audio_separation.get_audio_separation_run_info = mock.AsyncMock(
        return_value=sep_result
    )

    mock_resp = mock.MagicMock()
    mock_resp.content = b"fake audio data"
    mock_resp.raise_for_status = mock.MagicMock()

    with (
        mock.patch("google.adk.tools.camb._tools.os.unlink") as mock_unlink,
        mock.patch("httpx.AsyncClient") as mock_httpx,
    ):
      mock_http_instance = mock.AsyncMock()
      mock_http_instance.get = mock.AsyncMock(return_value=mock_resp)
      mock_http_instance.__aenter__ = mock.AsyncMock(
          return_value=mock_http_instance
      )
      mock_http_instance.__aexit__ = mock.AsyncMock(return_value=False)
      mock_httpx.return_value = mock_http_instance

      func = make_audio_separation_func(h)
      await func(audio_url="https://example.com/audio.wav")

      mock_unlink.assert_called_once()


# ---------------------------------------------------------------------------
# FunctionTool wrapping tests
# ---------------------------------------------------------------------------


class TestFunctionToolWrapping:
  """Verify that tool functions are correctly wrapped by FunctionTool."""

  def test_tts_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_tts_func(h))
    assert ft.name == "camb_tts"

  def test_translate_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_translate_func(h))
    assert ft.name == "camb_translate"

  def test_transcribe_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_transcribe_func(h))
    assert ft.name == "camb_transcribe"

  def test_translated_tts_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_translated_tts_func(h))
    assert ft.name == "camb_translated_tts"

  def test_clone_voice_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_clone_voice_func(h))
    assert ft.name == "camb_clone_voice"

  def test_list_voices_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_list_voices_func(h))
    assert ft.name == "camb_list_voices"

  def test_text_to_sound_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_text_to_sound_func(h))
    assert ft.name == "camb_text_to_sound"

  def test_audio_separation_function_tool_name(self):
    h = _make_helpers()
    ft = FunctionTool(make_audio_separation_func(h))
    assert ft.name == "camb_audio_separation"

  def test_tts_has_docstring(self):
    h = _make_helpers()
    ft = FunctionTool(make_tts_func(h))
    assert ft.description
    assert "text to speech" in ft.description.lower()

  def test_function_tool_generates_declaration(self):
    h = _make_helpers()
    ft = FunctionTool(make_tts_func(h))
    decl = ft._get_declaration()
    assert decl is not None
    assert decl.name == "camb_tts"
