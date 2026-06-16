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

import math
import os
import struct
import tempfile
import wave

from google.adk import Agent
from google.adk import Context
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.genai import types


def generate_wav(filename: str):
  """Generates a simple valid WAV file (a 440Hz sine wave)."""
  sample_rate = 44100.0
  duration = 1.0  # seconds
  frequency = 440.0  # sine wave frequency (A4)
  num_samples = int(sample_rate * duration)

  with wave.open(filename, "w") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(int(sample_rate))
    for i in range(num_samples):
      value = int(
          32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate)
      )
      wav_file.writeframes(struct.pack("<h", value))


def generate_bmp(filename: str):
  """Generates a simple valid BMP file (a red square)."""
  width, height = 100, 100
  file_size = 54 + 3 * width * height
  bmp_header = b"BM" + struct.pack(
      "<III IiiHHIIiiII",
      file_size,
      0,
      54,
      40,
      width,
      height,
      1,
      24,
      0,
      0,
      0,
      0,
      0,
      0,
  )
  pixel_data = b"\x00\x00\xFF" * (width * height)  # Red pixels in BGR
  with open(filename, "wb") as f:
    f.write(bmp_header + pixel_data)


def generate_video(filename: str):
  """Generates a simple valid MP4 video using OpenCV.

  Requires opencv-python.
  """
  try:
    import cv2
    import numpy as np
  except ImportError:
    raise ImportError(
        "opencv-python and numpy are required to generate video. Install"
        " them with: pip install opencv-python numpy"
    )

  width, height = 320, 240
  fourcc = cv2.VideoWriter_fourcc(*"avc1")
  out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
  if not out.isOpened():
    raise RuntimeError(
        "Failed to open VideoWriter. The 'avc1' codec might not be supported on"
        " this system."
    )

  for i in range(60):  # 3 seconds at 20fps
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw a moving white square
    x = (i * 5) % width
    cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 255, 255), -1)
    out.write(frame)

  out.release()


async def generate_report(
    topic: str, ctx: Context, format: str = "text"
) -> dict:
  """Generates a report on a topic and saves it as an artifact.

  Args:
      topic: The topic of the report.
      ctx: The tool context for saving artifacts.
      format: The format of the report ('text' or 'html').
  """
  if format.lower() == "html":
    mime_type = "text/html"
    filename = "report.html"
    content = f"""<html>
<head><title>Report on {topic}</title></head>
<body>
<h1>REPORT: {topic}</h1>
<hr>
<p>This is a detailed report about {topic}.</p>
<p>It contains a lot of useful information that would clutter the conversation history.</p>
<ul>
<li>Key point 1</li>
<li>Key point 2</li>
<li>Key point 3</li>
</ul>
</body>
</html>"""
  else:
    mime_type = "text/plain"
    filename = "report.txt"
    content = f"""REPORT: {topic}
=========================================
This is a detailed report about {topic}.
It contains a lot of useful information that would clutter the conversation history.
- Key point 1
- Key point 2
- Key point 3
"""

  version = await ctx.save_artifact(
      filename,
      types.Part.from_bytes(data=content.encode("utf-8"), mime_type=mime_type),
  )
  return {
      "message": (
          f"Report on {topic} saved as artifact '{filename}' (version"
          f" {version})."
      ),
      "filename": filename,
      "version": version,
  }


async def generate_media_artifact(media_type: str, ctx: Context) -> dict:
  """Generates a valid media artifact of specified type.

  Args:
      media_type: One of 'image', 'audio', 'video'.
      ctx: The tool context for saving artifacts.
  """

  with tempfile.TemporaryDirectory() as tmpdir:
    if media_type == "image":
      mime_type = "image/bmp"
      file_path = os.path.join(tmpdir, "sample.bmp")
      generate_bmp(file_path)
      filename = "sample_image.bmp"
    elif media_type == "audio":
      mime_type = "audio/wav"
      file_path = os.path.join(tmpdir, "sample.wav")
      generate_wav(file_path)
      filename = "sample_audio.wav"
    elif media_type == "video":
      mime_type = "video/mp4"
      file_path = os.path.join(tmpdir, "sample.mp4")
      try:
        generate_video(file_path)
      except (ImportError, RuntimeError) as e:
        return {"error": str(e)}
      filename = "sample_video.mp4"
    else:
      return {"error": f"Unsupported media type: {media_type}"}

    with open(file_path, "rb") as f:
      data = f.read()

    version = await ctx.save_artifact(
        filename,
        types.Part.from_bytes(data=data, mime_type=mime_type),
    )

  return {
      "message": (
          f"Media artifact '{filename}' generated and saved (version"
          f" {version})."
      ),
      "filename": filename,
      "version": version,
  }


root_agent = Agent(
    name="artifacts_agent",
    tools=[generate_report, generate_media_artifact, LoadArtifactsTool()],
    instruction="""You are an agent that can manage artifacts, including different media types.

    - To generate a text report, use `generate_report`.
    - To generate image, audio, or video artifacts, use `generate_media_artifact`.

    When the user asks about an artifact or to load it, use `load_artifacts`.
    """,
)
