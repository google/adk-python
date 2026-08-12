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

import asyncio
import base64
import json
import logging
import os
import re
import sys
import urllib.parse

import httpx
import pyaudio
import websockets

# --- Optional: For Audio Recording ---
# This is used to record audios for debugging purposes.
try:
  import numpy as np  # PyAudio will need NumPy for WAV conversion if not already int16
  import sounddevice as sd  # Sounddevice is for recording in this setup

  AUDIO_RECORDING_ENABLED = True
except ImportError:
  print(
      "WARNING: Sounddevice or numpy not found. Audio RECORDING will be"
      " disabled."
  )
  AUDIO_RECORDING_ENABLED = False

# --- PyAudio Playback Enabled Flag ---
# We assume PyAudio is for playback. If its import failed, this would be an issue.
# For simplicity, we'll try to initialize it and handle errors there.
AUDIO_PLAYBACK_ENABLED = True  # Will be set to False if init fails


# --- Configure Logging ---
LOG_FILE_NAME = "websocket_client.log"
LOG_FILE_PATH = os.path.abspath(LOG_FILE_NAME)
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] (%(funcName)s)"
        " - %(message)s"
    ),
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
print(
    f"INFO: Logging to console and to file. Log file location: {LOG_FILE_PATH}",
    flush=True,
)
logging.info(f"Logging configured. Logs will also be saved to: {LOG_FILE_PATH}")

if not AUDIO_RECORDING_ENABLED:
  logging.warning("Audio RECORDING is disabled due to missing libraries.")

# --- Configuration ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# APP_NAME is the folder name of your agent.
APP_NAME = "hello_world"
# The following default ones also work
USER_ID = "your_user_id_123"
SESSION_ID = "your_session_id_abc"
MODALITIES = ["TEXT", "AUDIO"]

REC_AUDIO_SAMPLE_RATE = 16000  # Matches SEND_SAMPLE_RATE from old code
REC_AUDIO_CHANNELS = 1  # Matches CHANNELS from old code
REC_AUDIO_FORMAT_PYAUDIO = pyaudio.paInt16  # Matches FORMAT from old code

REC_AUDIO_CHUNK_SIZE = 1024  # Matches CHUNK_SIZE from old code
REC_AUDIO_MIME_TYPE = "audio/pcm"  # This remains critical

# Recording parameters
REC_AUDIO_SAMPLE_RATE = 16000
REC_AUDIO_CHANNELS = 1
REC_AUDIO_FORMAT_DTYPE = "int16"  # Sounddevice dtype
# REC_AUDIO_MIME_TYPE = "audio/wav"  # We'll send WAV to server
REC_AUDIO_MIME_TYPE = "audio/pcm"
REC_AUDIO_SOUNDFILE_SUBTYPE = "PCM_16"  # Soundfile subtype for WAV

# PyAudio Playback Stream Parameters
PYAUDIO_PLAY_RATE = 24000
PYAUDIO_PLAY_CHANNELS = 1
PYAUDIO_PLAY_FORMAT = pyaudio.paInt16
PYAUDIO_PLAY_FORMAT_NUMPY = np.int16 if AUDIO_RECORDING_ENABLED else None  # type: ignore
PYAUDIO_FRAMES_PER_BUFFER = 1024

AUDIO_DURATION_SECONDS = 5  # For single "audio" command

# Global PyAudio instances
pya_interface_instance = None
pya_output_stream_instance = None

# --- Globals for Continuous Audio Streaming ---
is_streaming_audio = False
global_input_stream = None  # Holds the sounddevice.InputStream object
audio_stream_task = None  # Holds the asyncio.Task for audio streaming

debug_audio_save_count = 0
MAX_DEBUG_AUDIO_SAMPLES = 3  # Save first 3 chunks


CHUNK = 4200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5
INPUT_RATE = 16000
OUTPUT_RATE = 24000

config = {
    "response_modalities": ["AUDIO"],
    "input_audio_transcription": {},
    "output_audio_transcription": {},
}


# --- PyAudio Initialization and Cleanup ---
def init_pyaudio_playback():
  global pya_interface_instance, pya_output_stream_instance, AUDIO_PLAYBACK_ENABLED
  if (
      not AUDIO_PLAYBACK_ENABLED
  ):  # If already marked as disabled (e.g. previous attempt failed)
    logging.warning("PyAudio playback init skipped as it's marked disabled.")
    return False
  try:
    pya_interface_instance = pyaudio.PyAudio()
    logging.info(
        f"Initializing PyAudio output stream: Rate={PYAUDIO_PLAY_RATE},"
        f" Channels={PYAUDIO_PLAY_CHANNELS}, Format=paInt16"
    )
    pya_output_stream_instance = pya_interface_instance.open(
        format=PYAUDIO_PLAY_FORMAT,
        channels=PYAUDIO_PLAY_CHANNELS,
        rate=PYAUDIO_PLAY_RATE,
        output=True,
        frames_per_buffer=PYAUDIO_FRAMES_PER_BUFFER,
    )
    logging.info("PyAudio output stream initialized successfully.")
    AUDIO_PLAYBACK_ENABLED = True
    return True
  except Exception as e:
    logging.error(
        f"Failed to initialize PyAudio: {e}. Playback will be disabled.",
        exc_info=True,
    )
    print(
        f"ERROR: Failed to initialize PyAudio for playback: {e}. Check"
        " PortAudio installation if on Linux/macOS.",
        flush=True,
    )
    if pya_interface_instance:  # Terminate if open failed mid-way
      try:
        pya_interface_instance.terminate()
      except:
        pass
    pya_interface_instance = None
    pya_output_stream_instance = None
    AUDIO_PLAYBACK_ENABLED = False  # Mark as disabled
    return False


# --- Payload Creation ---
def create_text_request_payload(text: str) -> str:
  live_request_data = {"content": {"parts": [{"text": text}]}}
  logging.debug(
      f"Created LiveRequest text payload: {json.dumps(live_request_data)}"
  )
  return json.dumps(live_request_data)


def create_audio_request_payload(audio_bytes: bytes, mime_type: str) -> str:
  base64_encoded_audio = base64.b64encode(audio_bytes)
  base64_encoded_audio = base64_encoded_audio.decode("utf-8")
  live_request_data = {
      "blob": {
          "mime_type": mime_type,
          "data": base64_encoded_audio,
      }
  }
  return json.dumps(live_request_data)


class AudioStreamingComponent:

  async def stop_audio_streaming(self):
    global is_streaming_audio
    if is_streaming_audio:
      logging.info("Requesting to stop audio streaming (flag set).")
      is_streaming_audio = False
    else:
      logging.info("Audio streaming is not currently active.")

  async def start_audio_streaming(
      self,
      websocket: websockets.WebSocketClientProtocol,
  ):
    print("Starting continuous audio streaming...")
    global is_streaming_audio, global_input_stream, debug_audio_save_count

    # IMPORTANT: Reinstate this check
    if not AUDIO_RECORDING_ENABLED:
      logging.warning("Audio recording disabled. Cannot start stream.")
      is_streaming_audio = (
          False  # Ensure flag is correctly set if we bail early
      )
      return

    is_streaming_audio = True
    debug_audio_save_count = 0  # Reset counter for each stream start
    logging.info("Starting continuous audio streaming...")

    global pya_interface_instance

    try:
      stream = pya_interface_instance.open(
          format=FORMAT,
          channels=CHANNELS,
          rate=INPUT_RATE,
          input=True,
          frames_per_buffer=CHUNK,
      )

      while is_streaming_audio:
        try:
          audio_data_bytes = stream.read(CHUNK)

          if audio_data_bytes:
            payload_str = create_audio_request_payload(
                audio_data_bytes,
                REC_AUDIO_MIME_TYPE,  # REC_AUDIO_MIME_TYPE is likely "audio/wav"
            )

            await websocket.send(payload_str)
            # Make sure we sleep to yield control back to other threads(like audio playing)
            await asyncio.sleep(10**-12)
          else:
            logging.warning("Empty audio data chunk from queue, not sending.")

        except asyncio.TimeoutError:
          continue
        except websockets.exceptions.ConnectionClosed as e:
          logging.warning(
              f"WebSocket connection closed while sending audio stream: {e}"
          )
          is_streaming_audio = False
          break
        except Exception as e:
          logging.error(
              f"Error in audio streaming send loop: {e}", exc_info=True
          )
          is_streaming_audio = False
          break
    except Exception as e:
      logging.error(
          f"Failed to start or run audio InputStream: {e}", exc_info=True
      )
      is_streaming_audio = False  # Ensure flag is reset
    finally:
      logging.info("Cleaning up audio stream...")
      if global_input_stream:
        try:
          if global_input_stream.active:
            global_input_stream.stop()
            global_input_stream.close()
          logging.info("Sounddevice InputStream stopped and closed.")
        except Exception as e_sd_close:
          logging.error(
              f"Error stopping/closing Sounddevice InputStream: {e_sd_close}"
          )
        global_input_stream = None
      is_streaming_audio = False  # Critical to reset this
      logging.info("Continuous audio streaming task finished.")


class AgentResponseAudioPlayer:

  def cleanup_pyaudio_playback(self):
    global pya_interface_instance, pya_output_stream_instance
    logging.info("Attempting PyAudio cleanup...")
    if pya_output_stream_instance:
      try:
        if pya_output_stream_instance.is_active():  # Check if stream is active
          pya_output_stream_instance.stop_stream()
        pya_output_stream_instance.close()
        logging.info("PyAudio output stream stopped and closed.")
      except Exception as e:
        logging.error(f"Error closing PyAudio stream: {e}", exc_info=True)
      finally:
        pya_output_stream_instance = None
    if pya_interface_instance:
      try:
        pya_interface_instance.terminate()
        logging.info("PyAudio interface terminated.")
      except Exception as e:
        logging.error(
            f"Error terminating PyAudio interface: {e}", exc_info=True
        )
      finally:
        pya_interface_instance = None
    logging.info("PyAudio cleanup process finished.")

  # --- Audio Playback Handler (using PyAudio) ---
  def _play_audio_pyaudio_handler(
      self, audio_bytes: bytes, mime_type_full: str
  ):
    if not AUDIO_PLAYBACK_ENABLED or not pya_output_stream_instance:
      logging.warning(
          "PyAudio stream not available or playback disabled. Cannot play"
          " audio."
      )
      return
    try:
      logging.debug(
          f"PyAudio handler: Mime='{mime_type_full}', Size={len(audio_bytes)}"
      )
      playable_data_bytes = None

      mime_type_base = mime_type_full.split(";")[0].strip().lower()

      if mime_type_base == "audio/pcm":
        # Check rate from MIME type like "audio/pcm;rate=24000"
        match = re.search(r"rate=(\d+)", mime_type_full, re.IGNORECASE)
        current_audio_rate = PYAUDIO_PLAY_RATE  # Fallback to stream's rate
        if match:
          try:
            current_audio_rate = int(match.group(1))
          except ValueError:
            logging.warning(
                f"Could not parse rate from '{mime_type_full}', using stream"
                f" default {PYAUDIO_PLAY_RATE}Hz."
            )

        if current_audio_rate != PYAUDIO_PLAY_RATE:
          logging.warning(
              f"Received PCM audio at {current_audio_rate}Hz but PyAudio stream"
              f" is {PYAUDIO_PLAY_RATE}Hz. Playback speed/pitch will be"
              " affected. Resampling would be needed for correct playback."
          )
          # We will play it at PYAUDIO_PLAY_RATE, which will alter speed/pitch if rates differ.

        # We assume the incoming PCM data is 1 channel, 16-bit, matching the stream.
        # If server sent different channel count or bit depth, conversion would be needed.
        playable_data_bytes = audio_bytes
        logging.info(
            "Preparing raw PCM for PyAudio stream (target rate"
            f" {PYAUDIO_PLAY_RATE}Hz)."
        )
      else:
        logging.warning(
            f"Unsupported MIME type for PyAudio playback: {mime_type_full}"
        )
        return

      if playable_data_bytes:
        pya_output_stream_instance.write(playable_data_bytes)
        logging.info(
            "Audio chunk written to PyAudio stream (Size:"
            f" {len(playable_data_bytes)} bytes)."
        )
      else:
        logging.warning("No playable bytes prepared for PyAudio.")

    except Exception as e:
      logging.error(
          f"Error in _blocking_play_audio_pyaudio_handler: {e}", exc_info=True
      )

  async def play_audio_data(self, audio_bytes: bytes, mime_type: str):
    if not AUDIO_PLAYBACK_ENABLED:
      logging.debug(
          "PyAudio Playback is disabled, skipping play_audio_data call."
      )
      return
    print(f"Scheduling PyAudio playback for {mime_type} audio.")
    await asyncio.to_thread(
        self._play_audio_pyaudio_handler, audio_bytes, mime_type
    )


# --- Session Management ---
async def ensure_session_exists(
    app_name: str,
    user_id: str,
    session_id: str,
    server_host: str,
    server_port: int,
) -> bool:
  session_url = f"http://{server_host}:{server_port}/apps/{app_name}/users/{user_id}/sessions/{session_id}"
  try:
    async with httpx.AsyncClient() as client:
      logging.info(f"Checking if session exists via GET: {session_url}")
      response_get = await client.get(session_url, timeout=10)
      if response_get.status_code == 200:
        logging.info(f"Session '{session_id}' already exists.")
        return True
      elif response_get.status_code == 404:
        logging.info(
            f"Session '{session_id}' not found. Attempting to create via POST."
        )
        response_post = await client.post(session_url, json={}, timeout=10)
        if response_post.status_code == 200:
          logging.info(f"Session '{session_id}' created.")
          return True
        else:
          logging.error(
              f"Failed to create session '{session_id}'. POST Status:"
              f" {response_post.status_code}"
          )
          return False
      else:
        logging.warning(
            f"Could not verify session '{session_id}'. GET Status:"
            f" {response_get.status_code}"
        )
        return False
  except Exception as e:
    logging.error(f"Error ensuring session '{session_id}': {e}", exc_info=True)
    return False


async def websocket_client():
  global audio_stream_task
  logging.info("websocket_client function started.")

  # --- ADD THIS SECTION FOR DEVICE DIAGNOSTICS ---
  if AUDIO_RECORDING_ENABLED:
    try:
      print("-" * 30)
      print("Available audio devices:")
      devices = sd.query_devices()
      print(devices)
      print(f"Default input device: {sd.query_devices(kind='input')}")
      print(f"Default output device: {sd.query_devices(kind='output')}")
      print("-" * 30)
    except Exception as e_dev:
      logging.error(f"Could not query audio devices: {e_dev}")
  # --- END DEVICE DIAGNOSTICS ---

  if not init_pyaudio_playback():
    logging.warning("PyAudio playback could not be initialized.")

  agent_response_audio_player = AgentResponseAudioPlayer()
  audio_streaming_component = AudioStreamingComponent()
  if (
      APP_NAME == "hello_world"
      or USER_ID.startswith("your_user_id")
      or SESSION_ID.startswith("your_session_id")
  ):
    logging.warning("Using default/example APP_NAME, USER_ID, or SESSION_ID.")

  session_ok = await ensure_session_exists(
      APP_NAME, USER_ID, SESSION_ID, SERVER_HOST, SERVER_PORT
  )
  if not session_ok:
    logging.error(
        f"Critical: Could not ensure session '{SESSION_ID}'. Aborting."
    )
    return

  params = {
      "app_name": APP_NAME,
      "user_id": USER_ID,
      "session_id": SESSION_ID,
      "modalities": MODALITIES,
  }
  uri = (
      f"ws://{SERVER_HOST}:{SERVER_PORT}/run_live?{urllib.parse.urlencode(params, doseq=True)}"
  )
  logging.info(f"Attempting to connect to WebSocket: {uri}")

  try:
    async with websockets.connect(
        uri, open_timeout=10, close_timeout=10
    ) as websocket:
      logging.info(f"Successfully connected to WebSocket: {uri}.")

      async def receive_messages(websocket: websockets.WebSocketClientProtocol):
        # ... (Logic for parsing event_data and finding audio part is the same) ...
        # ... (When audio part is found, call `await play_audio_data(audio_bytes_decoded, mime_type_full)`) ...
        logging.info("Receiver task started: Listening for server messages...")
        try:
          async for message in websocket:
            # logging.info(f"<<< Raw message from server: {message[:500]}...")
            try:
              event_data = json.loads(message)
              logging.info(
                  "<<< Parsed event from server: (Keys:"
                  f" {list(event_data.keys())})"
              )
              if "content" in event_data and isinstance(
                  event_data["content"], dict
              ):
                content_obj = event_data["content"]
                if "parts" in content_obj and isinstance(
                    content_obj["parts"], list
                ):
                  for part in content_obj["parts"]:
                    if isinstance(part, dict) and "inlineData" in part:
                      inline_data = part["inlineData"]
                      if (
                          isinstance(inline_data, dict)
                          and "mimeType" in inline_data
                          and isinstance(inline_data["mimeType"], str)
                          and inline_data["mimeType"].startswith("audio/")
                          and "data" in inline_data
                          and isinstance(inline_data["data"], str)
                      ):
                        audio_b64 = inline_data["data"]
                        mime_type_full = inline_data["mimeType"]
                        logging.info(
                            f"Audio part found: Mime='{mime_type_full}',"
                            f" Base64Len={len(audio_b64)}"
                        )
                        try:
                          standard_b64_string = audio_b64.replace(
                              "-", "+"
                          ).replace("_", "/")
                          missing_padding = len(standard_b64_string) % 4
                          if missing_padding:
                            standard_b64_string += "=" * (4 - missing_padding)

                          audio_bytes_decoded = base64.b64decode(
                              standard_b64_string
                          )

                          if audio_bytes_decoded:
                            await agent_response_audio_player.play_audio_data(
                                audio_bytes_decoded, mime_type_full
                            )
                          else:
                            logging.warning(
                                "Decoded audio data is empty after sanitization"
                                " and padding."
                            )

                        except base64.binascii.Error as b64e:
                          # Log details if decoding still fails
                          logging.error(
                              "Base64 decode error after sanitization and"
                              " padding."
                              f" Error: {b64e}"
                          )
                        except Exception as e:
                          logging.error(
                              "Error processing audio for playback (original"
                              f" string prefix: '{audio_b64[:50]}...'): {e}",
                              exc_info=True,
                          )
            except json.JSONDecodeError:
              logging.warning(f"Received non-JSON: {message}")
            except Exception as e:
              logging.error(f"Error processing event: {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed as e:
          logging.warning(
              f"Receiver: Connection closed (Code: {e.code}, Reason:"
              f" '{e.reason if e.reason else 'N/A'}')"
          )
        except Exception as e:
          logging.error("Receiver: Unhandled error", exc_info=True)
        finally:
          logging.info("Receiver task finished.")

      async def send_messages_local(ws: websockets.WebSocketClientProtocol):
        global audio_stream_task, is_streaming_audio
        logging.info(
            "Sender task started: Type 'start_stream', 'stop_stream', text,"
            "sendfile, or 'quit'."
        )
        while True:
          await asyncio.sleep(10**-12)
          try:
            user_input = await asyncio.to_thread(input, "Enter command: ")
            if user_input.lower() == "quit":
              logging.info("Sender: 'quit' received.")
              if audio_stream_task and not audio_stream_task.done():
                logging.info(
                    "Sender: Stopping active audio stream due to quit command."
                )
                await audio_streaming_component.stop_audio_streaming()
                await audio_stream_task
                audio_stream_task = None
              break
            elif user_input.lower() == "start_stream":
              if audio_stream_task and not audio_stream_task.done():
                logging.warning("Sender: Audio stream is already running.")
                continue
              audio_stream_task = asyncio.create_task(
                  audio_streaming_component.start_audio_streaming(ws)
              )

              logging.info("Sender: Audio streaming task initiated.")
            elif user_input.lower() == "stop_stream":
              if audio_stream_task and not audio_stream_task.done():
                logging.info("Sender: Requesting to stop audio stream.")
                await audio_streaming_component.stop_audio_streaming()
                await audio_stream_task
                audio_stream_task = None
                logging.info("Sender: Audio streaming task stopped and joined.")
              else:
                logging.warning(
                    "Sender: Audio stream is not currently running or already"
                    " stopped."
                )
            # The 'audio' command for single recording was commented out in your version.
            # If you need it, uncomment the block from my previous response.
            elif user_input.lower().startswith("sendfile "):
              if (
                  audio_stream_task
                  and isinstance(audio_stream_task, asyncio.Task)
                  and not audio_stream_task.done()
              ):
                logging.warning(
                    "Please stop the current audio stream with 'stop_stream'"
                    " before sending a file."
                )
                continue

              filepath = user_input[len("sendfile ") :].strip()
              # fix filepath for testing
              # filepath = "roll_and_check_audio.wav"
              # Remove quotes if user added them around the filepath
              filepath = filepath.strip("\"'")

              if not os.path.exists(filepath):
                logging.error(f"Audio file not found: {filepath}")
                print(
                    f"Error: File not found at '{filepath}'. Please check the"
                    " path."
                )
                continue
              if not filepath.lower().endswith(".wav"):
                logging.warning(
                    f"File {filepath} does not end with .wav. Attempting to"
                    " send anyway."
                )
                print(
                    f"Warning: File '{filepath}' is not a .wav file. Ensure"
                    " it's a compatible WAV."
                )

              try:
                logging.info(f"Reading audio file: {filepath}")
                with open(filepath, "rb") as f:
                  audio_file_bytes = f.read()

                # We assume the file is already in WAV format.
                # REC_AUDIO_MIME_TYPE is "audio/wav"
                payload_str = create_audio_request_payload(
                    audio_file_bytes, REC_AUDIO_MIME_TYPE
                )
                logging.info(
                    ">>> Sending audio file"
                    f" {os.path.basename(filepath)} (Size:"
                    f" {len(audio_file_bytes)} bytes) with MIME type"
                    f" {REC_AUDIO_MIME_TYPE}"
                )
                await ws.send(payload_str)
                logging.info("Audio file sent.")
                print(f"Successfully sent {os.path.basename(filepath)}.")

              except Exception as e_sendfile:
                logging.error(
                    f"Error sending audio file {filepath}: {e_sendfile}",
                    exc_info=True,
                )
                print(f"Error sending file: {e_sendfile}")
            else:  # Text input
              if not user_input.strip():  # Prevent sending empty messages
                logging.info("Sender: Empty input, not sending.")
                continue
              payload_str = create_text_request_payload(user_input)
              logging.info(f">>> Sending text: {user_input[:100]}")
              await ws.send(payload_str)
          except EOFError:  # Handles Ctrl+D
            logging.info("Sender: EOF detected (Ctrl+D).")
            if audio_stream_task and not audio_stream_task.done():
              await audio_streaming_component.stop_audio_streaming()
              await audio_stream_task
              audio_stream_task = None
            break
          except websockets.exceptions.ConnectionClosed as e:
            logging.warning(
                f"Sender: WebSocket connection closed. Code: {e.code}, Reason:"
                f" {e.reason}"
            )
            if audio_stream_task and not audio_stream_task.done():
              is_streaming_audio = False  # Signal loop
              try:
                await asyncio.wait_for(audio_stream_task, timeout=2.0)
              except asyncio.TimeoutError:
                audio_stream_task.cancel()
              except Exception as ex:
                logging.error(f"Error during stream stop on conn close: {ex}")
              audio_stream_task = None
            break
          except Exception as e_send_loop:
            logging.error(
                f"Sender: Unhandled error: {e_send_loop}", exc_info=True
            )
            if audio_stream_task and not audio_stream_task.done():
              await audio_streaming_component.stop_audio_streaming()
              await audio_stream_task
              audio_stream_task = None
            break
        logging.info("Sender task finished.")

      receive_task = asyncio.create_task(
          receive_messages(websocket), name="ReceiverThread"
      )
      send_task = asyncio.create_task(
          send_messages_local(websocket), name="SenderThread"
      )

      done, pending = await asyncio.wait(
          [receive_task, send_task], return_when=asyncio.FIRST_COMPLETED
      )
      logging.info(
          f"Main task completion: Done={len(done)}, Pending={len(pending)}"
      )

      current_active_audio_task = audio_stream_task
      if current_active_audio_task and not current_active_audio_task.done():
        logging.info(
            "A main task finished. Ensuring audio stream is stopped if active."
        )
        await audio_streaming_component.stop_audio_streaming()
        try:
          await asyncio.wait_for(current_active_audio_task, timeout=5.0)
          logging.info(
              "Audio streaming task gracefully stopped after main task"
              " completion."
          )
        except asyncio.TimeoutError:
          logging.warning(
              "Timeout waiting for audio stream to stop post main task."
              " Cancelling."
          )
          current_active_audio_task.cancel()
        except Exception as e_stream_stop:
          logging.error(
              f"Error during audio stream stop after main task: {e_stream_stop}"
          )
        if audio_stream_task is current_active_audio_task:
          audio_stream_task = None

      for task in pending:
        if not task.done():
          task.cancel()
          logging.info(f"Cancelled pending main task: {task.get_name()}")

      all_tasks_to_await = list(done) + list(pending)
      for task in all_tasks_to_await:
        try:
          await task
        except asyncio.CancelledError:
          logging.info(f"Main task {task.get_name()} cancelled as expected.")
        except Exception as e:
          logging.error(
              f"Error awaiting main task {task.get_name()}: {e}", exc_info=True
          )
      logging.info("All main tasks awaited.")

  except Exception as e:
    logging.error(f"Outer error in websocket_client: {e}", exc_info=True)
  finally:
    final_check_audio_task = audio_stream_task
    if final_check_audio_task and not final_check_audio_task.done():
      logging.warning("Performing final cleanup of active audio stream task.")
      await audio_streaming_component.stop_audio_streaming()
      try:
        await asyncio.wait_for(final_check_audio_task, timeout=2.0)
      except asyncio.TimeoutError:
        final_check_audio_task.cancel()
      except Exception:
        pass
      audio_stream_task = None
    agent_response_audio_player.cleanup_pyaudio_playback()
    logging.info("websocket_client function finished.")


if __name__ == "__main__":
  logging.info("Script's main execution block started.")
  if (
      APP_NAME == "hello_world"
      or USER_ID.startswith("your_user_id")
      or SESSION_ID.startswith("your_session_id")
  ):
    print(
        "WARNING: Using default/example APP_NAME, USER_ID, or SESSION_ID."
        " Please update these.",
        flush=True,
    )

  try:
    asyncio.run(websocket_client())
  except KeyboardInterrupt:
    logging.info("Client execution interrupted by user (KeyboardInterrupt).")
    print("\nClient interrupted. Exiting.", flush=True)
  except Exception as e:
    logging.critical(
        "A critical unhandled exception occurred in __main__.", exc_info=True
    )
    print(f"CRITICAL ERROR: {e}. Check logs. Exiting.", flush=True)
  finally:
    logging.info(
        "Script's main execution block finished. Shutting down logging."
    )
    logging.shutdown()
    print("Script execution finished.", flush=True)
