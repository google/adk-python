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

"""CAMB AI tool function factories.

Each factory accepts a :class:`CambHelpers` instance and returns an
``async def`` that can be wrapped by :class:`FunctionTool`.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import Optional

from ._helpers import CambHelpers

_CAMB_TTS_RESULT_URL = "https://client.camb.ai/apis/tts-result/{run_id}"


def make_tts_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_tts`` async function."""

  async def camb_tts(
      text: str,
      language: str = "en-us",
      voice_id: int = 147320,
      speech_model: str = "mars-flash",
      user_instructions: Optional[str] = None,
  ) -> str:
    """Convert text to speech using camb.ai.

    Supports 140+ languages and multiple voice models. The audio is
    saved to a temporary file and the file path is returned.

    Args:
      text: Text to convert to speech (3-3000 characters).
      language: BCP-47 language code (e.g. 'en-us', 'fr-fr').
      voice_id: Voice ID. Use camb_list_voices to find voices.
      speech_model: Model: 'mars-flash', 'mars-pro', 'mars-instruct'.
      user_instructions: Instructions for mars-instruct model only.
    """
    from camb import StreamTtsOutputConfiguration

    client = helpers.get_client()
    kwargs: dict[str, Any] = {
        "text": text,
        "language": language,
        "voice_id": voice_id,
        "speech_model": speech_model,
        "output_configuration": StreamTtsOutputConfiguration(format="wav"),
    }
    if user_instructions and speech_model == "mars-instruct":
      kwargs["user_instructions"] = user_instructions

    chunks: list[bytes] = []
    async for chunk in client.text_to_speech.tts(**kwargs):
      chunks.append(chunk)
    return helpers.save_audio(b"".join(chunks), ".wav")

  return camb_tts


def make_translate_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_translate`` async function."""

  async def camb_translate(
      text: str,
      source_language: int,
      target_language: int,
      formality: Optional[int] = None,
  ) -> str:
    """Translate text between 140+ languages using camb.ai.

    Provide integer language codes: 1=English, 2=Spanish, 3=French,
    4=German, 5=Italian, 6=Portuguese, 7=Dutch, 8=Russian, 9=Japanese,
    10=Korean, 11=Chinese.

    Args:
      text: Text to translate.
      source_language: Source language code (integer).
      target_language: Target language code (integer).
      formality: Optional formality level: 1=formal, 2=informal.
    """
    from camb.core.api_error import ApiError

    client = helpers.get_client()
    kwargs: dict[str, Any] = {
        "text": text,
        "source_language": source_language,
        "target_language": target_language,
    }
    if formality:
      kwargs["formality"] = formality

    try:
      result = await client.translation.translation_stream(**kwargs)
      return helpers.extract_translation(result)
    except ApiError as e:
      # The CAMB SDK sometimes wraps a successful (HTTP 200) translation
      # response inside an ApiError. When this happens the body contains
      # the translated text. Re-raise for genuine errors.
      if e.status_code == 200 and e.body:
        return str(e.body)
      raise

  return camb_translate


def make_transcribe_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_transcribe`` async function."""

  async def camb_transcribe(
      language: int,
      audio_url: Optional[str] = None,
      audio_file_path: Optional[str] = None,
  ) -> str:
    """Transcribe audio to text with speaker identification using camb.ai.

    Supports audio URLs or local file paths. Returns JSON with full
    transcription text, timed segments, and speaker labels.

    Args:
      language: Language code (integer). 1=English, 2=Spanish, etc.
      audio_url: URL of the audio file to transcribe.
      audio_file_path: Local file path to the audio file.
    """
    client = helpers.get_client()
    kwargs: dict[str, Any] = {"language": language}

    if audio_url:
      import httpx

      async with httpx.AsyncClient(timeout=helpers._timeout) as http:
        resp = await http.get(audio_url)
        resp.raise_for_status()
      with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
      try:
        with open(tmp_path, "rb") as f:
          kwargs["media_file"] = f
          result = await client.transcription.create_transcription(**kwargs)
      finally:
        os.unlink(tmp_path)
    elif audio_file_path:
      with open(audio_file_path, "rb") as f:
        kwargs["media_file"] = f
        result = await client.transcription.create_transcription(**kwargs)
    else:
      raise ValueError("Provide either audio_url or audio_file_path")

    task_id = result.task_id
    status = await helpers.poll_async(
        client.transcription.get_transcription_task_status, task_id
    )
    transcription = await client.transcription.get_transcription_result(
        status.run_id
    )
    return helpers.format_transcription(transcription)

  return camb_transcribe


def make_translated_tts_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_translated_tts`` async function."""

  async def camb_translated_tts(
      text: str,
      source_language: int,
      target_language: int,
      voice_id: int = 147320,
      formality: Optional[int] = None,
  ) -> str:
    """Translate text and convert to speech in one step using camb.ai.

    Returns the file path to the generated audio file.

    Args:
      text: Text to translate and speak.
      source_language: Source language code (integer).
      target_language: Target language code (integer).
      voice_id: Voice ID for TTS output.
      formality: Optional formality: 1=formal, 2=informal.
    """
    import httpx

    client = helpers.get_client()
    kwargs: dict[str, Any] = {
        "text": text,
        "voice_id": voice_id,
        "source_language": source_language,
        "target_language": target_language,
    }
    if formality:
      kwargs["formality"] = formality

    result = await client.translated_tts.create_translated_tts(**kwargs)
    status = await helpers.poll_async(
        client.translated_tts.get_translated_tts_task_status,
        result.task_id,
    )

    run_id = getattr(status, "run_id", None)
    audio_data = b""
    fmt = "pcm"
    if run_id:
      url = _CAMB_TTS_RESULT_URL.format(run_id=run_id)
      async with httpx.AsyncClient(timeout=helpers._timeout) as http:
        resp = await http.get(
            url, headers={"x-api-key": helpers._api_key or ""}
        )
        if resp.status_code == 200:
          audio_data = resp.content
          fmt = helpers.detect_audio_format(
              audio_data, resp.headers.get("content-type", "")
          )

    if not audio_data:
      raise RuntimeError(
          f"Translated TTS failed: no audio data received (run_id={run_id})"
      )

    if fmt == "pcm" and audio_data:
      audio_data = helpers.add_wav_header(audio_data)
      fmt = "wav"

    ext = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg"}.get(
        fmt, ".wav"
    )
    return helpers.save_audio(audio_data, ext)

  return camb_translated_tts


def make_clone_voice_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_clone_voice`` async function."""

  async def camb_clone_voice(
      voice_name: str,
      audio_file_path: str,
      gender: int,
      description: Optional[str] = None,
      age: Optional[int] = None,
      language: Optional[int] = None,
  ) -> str:
    """Clone a voice from an audio sample using camb.ai.

    Creates a custom voice from a 2+ second audio sample that can be
    used with camb_tts and camb_translated_tts.

    Args:
      voice_name: Name for the new cloned voice.
      audio_file_path: Path to audio file (minimum 2 seconds).
      gender: Gender: 1=Male, 2=Female, 0=Not Specified, 9=Not Applicable.
      description: Optional description of the voice.
      age: Optional age of the voice.
      language: Optional language code for the voice.
    """
    client = helpers.get_client()
    with open(audio_file_path, "rb") as f:
      kwargs: dict[str, Any] = {
          "voice_name": voice_name,
          "gender": gender,
          "file": f,
      }
      if description:
        kwargs["description"] = description
      if age:
        kwargs["age"] = age
      if language:
        kwargs["language"] = language
      result = await client.voice_cloning.create_custom_voice(**kwargs)

    out: dict[str, Any] = {
        "voice_id": getattr(result, "voice_id", getattr(result, "id", None)),
        "voice_name": voice_name,
        "status": "created",
    }
    if hasattr(result, "message"):
      out["message"] = result.message
    return json.dumps(out, indent=2)

  return camb_clone_voice


def make_list_voices_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_list_voices`` async function."""

  async def camb_list_voices() -> str:
    """List all available voices from camb.ai.

    Returns voice IDs, names, genders, ages, and languages. Use the
    voice ID with camb_tts or camb_translated_tts.
    """
    client = helpers.get_client()
    voices = await client.voice_cloning.list_voices()
    return helpers.format_voices(voices)

  return camb_list_voices


def make_text_to_sound_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_text_to_sound`` async function."""

  async def camb_text_to_sound(
      prompt: str,
      duration: Optional[float] = None,
      audio_type: Optional[str] = None,
  ) -> str:
    """Generate sounds, music, or soundscapes from text descriptions using camb.ai.

    Describe the sound or music you want and the tool will generate it.
    Returns the file path to the generated audio file.

    Args:
      prompt: Description of the sound or music to generate.
      duration: Optional duration in seconds.
      audio_type: Optional type: 'music' or 'sound'.
    """
    client = helpers.get_client()
    kwargs: dict[str, Any] = {"prompt": prompt}
    if duration:
      kwargs["duration"] = duration
    if audio_type:
      kwargs["audio_type"] = audio_type

    result = await client.text_to_audio.create_text_to_audio(**kwargs)
    status = await helpers.poll_async(
        client.text_to_audio.get_text_to_audio_status, result.task_id
    )

    chunks: list[bytes] = []
    async for chunk in client.text_to_audio.get_text_to_audio_result(
        status.run_id
    ):
      chunks.append(chunk)
    return helpers.save_audio(b"".join(chunks), ".wav")

  return camb_text_to_sound


def make_audio_separation_func(
    helpers: CambHelpers,
) -> Callable[..., Coroutine[Any, Any, str]]:
  """Create the ``camb_audio_separation`` async function."""

  async def camb_audio_separation(
      audio_url: Optional[str] = None,
      audio_file_path: Optional[str] = None,
  ) -> str:
    """Separate vocals/speech from background audio using camb.ai.

    Provide either an audio URL or a local file path. Returns JSON with
    paths to the separated vocals and background audio files.

    Args:
      audio_url: URL of the audio file to separate.
      audio_file_path: Local file path to the audio file.
    """
    client = helpers.get_client()
    kwargs: dict[str, Any] = {}

    if audio_url:
      import httpx

      async with httpx.AsyncClient(timeout=helpers._timeout) as http:
        resp = await http.get(audio_url)
        resp.raise_for_status()
      with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name
      try:
        with open(tmp_path, "rb") as f:
          kwargs["media_file"] = f
          result = await client.audio_separation.create_audio_separation(
              **kwargs
          )
      finally:
        os.unlink(tmp_path)
    elif audio_file_path:
      with open(audio_file_path, "rb") as f:
        kwargs["media_file"] = f
        result = await client.audio_separation.create_audio_separation(**kwargs)
    else:
      raise ValueError("Provide either audio_url or audio_file_path")

    status = await helpers.poll_async(
        client.audio_separation.get_audio_separation_status, result.task_id
    )
    sep = await client.audio_separation.get_audio_separation_run_info(
        status.run_id
    )
    return helpers.format_separation(sep)

  return camb_audio_separation
