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

"""LiveKit connector for ADK live agents.

`LiveKitRunner` bridges an already-connected LiveKit room to an ADK `Runner`
over the `LiveRequestQueue` -> `run_live()` -> `Event` contract. See README.md
for the channels it publishes on and what clients see.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from collections.abc import Callable
import contextlib
import io
import json
import logging
import time
from typing import Any
from typing import Optional

from google.genai import types

from ...agents.live_request_queue import LiveRequest
from ...agents.live_request_queue import LiveRequestQueue
from ...agents.run_config import RunConfig
from ...errors.already_exists_error import AlreadyExistsError
from ...events.event import Event
from ...features import experimental
from ...features import FeatureName
from ...runners import Runner
from ._livekit_call import _register_call
from ._livekit_call import LiveKitCall
from ._livekit_call import SIP_CALL_ID_ATTRIBUTE
from ._livekit_call import SIP_PHONE_NUMBER_ATTRIBUTE
from ._livekit_call import SIP_TRUNK_PHONE_NUMBER_ATTRIBUTE
from ._rtc import EventTypes
from ._rtc import rtc
from ._transcripts import TranscriptSegment

try:
  from PIL import Image
except ImportError as e:
  raise ImportError(
      "Pillow is not installed. It is required to encode inbound video. "
      'Install it with `pip install "google-adk[livekit]"`.'
  ) from e

logger = logging.getLogger("google_adk." + __name__)

# LiveKit resamples inbound audio on subscribe, so the bridge only has to ask.
_INPUT_SAMPLE_RATE = 16000
_OUTPUT_SAMPLE_RATE = 24000
_NUM_CHANNELS = 1
_BYTES_PER_SAMPLE = 2  # 16-bit PCM.
# ADK identifies live input by rate in the mime type, not by a bare audio/pcm.
_AUDIO_MIME_TYPE = f"audio/pcm;rate={_INPUT_SAMPLE_RATE}"
_VIDEO_MIME_TYPE = "image/jpeg"
_OUTPUT_AUDIO_TRACK_NAME = "adk-agent-audio"

# Live models sample video rather than consume it at capture rate.
_VIDEO_FRAMES_PER_SECOND = 1.0
_VIDEO_MAX_WIDTH = 1024
_VIDEO_MAX_HEIGHT = 1024
_VIDEO_JPEG_QUALITY = 75

# LiveKit's standard channels, which their client SDKs already render.
LK_CHAT_TOPIC = "lk.chat"
_LK_AGENT_STATE_ATTRIBUTE = "lk.agent.state"

# Tool activity has no LiveKit convention, so it gets an ADK topic.
DATA_TOPIC = "adk"

# Agent state values LiveKit's voice-assistant components understand.
_STATE_INITIALIZING = "initializing"
_STATE_LISTENING = "listening"
_STATE_THINKING = "thinking"
_STATE_SPEAKING = "speaking"

# Digits arrive one event at a time; buffer them into one turn.
_DTMF_IDLE_FLUSH_SECONDS = 1.5
_DTMF_TERMINATORS = frozenset({"#"})

# Long enough to cover a phone ringing, short enough that a room nobody joins
# does not pin a worker for the length of a call.
_DEFAULT_PARTICIPANT_WAIT = 30.0

_USER_ROLE = "user"
_AGENT_ROLE = "agent"


class _EndOfPlayback:
  """Sentinel that lets the playback task drain and exit at end of call."""


_END_OF_PLAYBACK = _EndOfPlayback()


@experimental(FeatureName.LIVEKIT)
class LiveKitRunner:
  """Bridges a LiveKit room to an ADK `Runner.run_live()` session.

  One instance drives one call. Construct it once the room is connected, then
  `await start()`::

      room = rtc.Room()
      await room.connect(livekit_url, agent_token)
      await LiveKitRunner(
          runner=runner, room=room, user_id="alice", session_id=room_name
      ).start()
  """

  def __init__(
      self,
      runner: Runner,
      room: rtc.Room,
      *,
      user_id: str,
      session_id: str,
      run_config: Optional[RunConfig] = None,
      create_session: bool = True,
      wait_for_participant: Optional[float] = _DEFAULT_PARTICIPANT_WAIT,
  ):
    """Initializes the runner.

    Args:
      runner: An unmodified ADK `Runner`.
      room: An already-connected LiveKit room.
      user_id: The ADK user id for the session.
      session_id: The ADK session id for the session.
      run_config: Extends the defaults rather than replacing them. Audio out
        and session resumption are filled in if you leave them unset;
        everything else, transcription included, is yours.
      create_session: Whether to create the ADK session if it is missing.
      wait_for_participant: Seconds to wait for a caller to join before
        opening the model connection. Pass None to start immediately.
    """
    self._runner = runner
    self._room = room
    self._user_id = user_id
    self._session_id = session_id
    self._run_config = _with_live_defaults(run_config)
    self._create_session = create_session
    self._wait_for_participant = wait_for_participant
    self._queue = LiveRequestQueue()

    self._audio_source = rtc.AudioSource(_OUTPUT_SAMPLE_RATE, _NUM_CHANNELS)
    self._audio_track = rtc.LocalAudioTrack.create_audio_track(
        _OUTPUT_AUDIO_TRACK_NAME, self._audio_source
    )
    # Playback is paced by the audio source, so it gets its own task rather
    # than throttling the event pump to realtime.
    self._playback: asyncio.Queue[bytes | _EndOfPlayback] = asyncio.Queue()
    self._playback_task: Optional[asyncio.Task[None]] = None
    self._output_track_sid: Optional[str] = None
    self._forward_tasks: set[asyncio.Task[None]] = set()
    self._ended = asyncio.Event()
    # Set at start, so a tool running off the loop can still end the call.
    self._loop: Optional[asyncio.AbstractEventLoop] = None
    self._agent_state: Optional[str] = None
    self._last_caller_state: dict[str, Any] = {}
    # One in-flight transcription segment per side of the call: ADK reports
    # inbound speech as the user's, whoever in the room actually said it.
    self._segments: dict[str, TranscriptSegment] = {}
    self._dtmf_digits: list[str] = []
    self._dtmf_identity: Optional[str] = None
    self._dtmf_flush_handle: Optional[asyncio.TimerHandle] = None
    self._call = LiveKitCall(
        room=room,
        user_id=user_id,
        session_id=session_id,
        hang_up_callback=self._end_session,
    )

  @property
  def call(self) -> LiveKitCall:
    """The call handle tools reach through `current_call()`."""
    return self._call

  async def start(self) -> None:
    """Runs the live session until the call ends.

    Returns when the caller hangs up, the room closes, or `run_live` finishes.
    Releases the room and closes the queue either way, setup failures included.

    Raises:
      RuntimeError: If another call is already in progress on this session.
    """
    self._loop = asyncio.get_running_loop()
    # Named up front: teardown runs even if setup fails before these exist.
    session: Optional[asyncio.Task[None]] = None
    hung_up: Optional[asyncio.Task[bool]] = None
    # Registered around the whole call rather than around `run_live`, so a
    # tool still resolves it during teardown, and deregistered on the way out
    # however the call ends.
    with _register_call(self._call):
      try:
        await self._publish_output_audio_track()
        await self._set_agent_state(_STATE_INITIALIZING)
        # Before the session is created, so caller identity is in the session
        # state from the first turn rather than arriving later as a delta.
        await self._await_participant()
        if self._create_session:
          await self._ensure_session()
        self._subscribe_existing_tracks()
        self._subscribe_room_events()

        # The session ends on whichever comes first: `run_live` finishing, or
        # the call ending. Hanging up cannot be signalled by closing the queue
        # -- `run_live` reads the resulting connection close as a dropped
        # connection and reconnects on the resumption handle -- so it is
        # tracked separately and cancels the event pump.
        session = asyncio.create_task(self._forward_events())
        hung_up = asyncio.create_task(self._ended.wait())
        done, _ = await asyncio.wait(
            {session, hung_up}, return_when=asyncio.FIRST_COMPLETED
        )
        if session in done:
          session.result()  # Re-raise whatever ended the session.
      finally:
        if self._dtmf_flush_handle is not None:
          self._dtmf_flush_handle.cancel()
        started: list[asyncio.Task[Any]] = [
            task for task in (session, hung_up) if task is not None
        ]
        pending = [*started, *self._forward_tasks]
        for task in pending:
          task.cancel()
        # Awaited, and not merely cancelled, because a forwarder's `finally`
        # still pushes onto the queue we are about to close.
        await asyncio.gather(*pending, return_exceptions=True)
        self._unsubscribe_room_events()
        self._queue.close()

  def _subscribe_room_events(self) -> None:
    """Wires the room's callbacks to this session."""
    for event, handler in self._room_handlers():
      self._room.on(event, handler)
    self._room.register_text_stream_handler(LK_CHAT_TOPIC, self._on_chat_stream)

  def _unsubscribe_room_events(self) -> None:
    """Releases the room, so a second session can reuse it.

    LiveKit refuses a duplicate handler on a text-stream topic.
    """
    for event, handler in self._room_handlers():
      with contextlib.suppress(Exception):
        self._room.off(event, handler)
    with contextlib.suppress(Exception):
      self._room.unregister_text_stream_handler(LK_CHAT_TOPIC)

  def _room_handlers(
      self,
  ) -> tuple[tuple[EventTypes, Callable[..., None]], ...]:
    return (
        ("track_subscribed", self._on_track_subscribed),
        ("data_received", self._on_data_received),
        ("sip_dtmf_received", self._on_dtmf_received),
        ("disconnected", self._on_disconnected),
        ("participant_disconnected", self._on_participant_disconnected),
        ("participant_attributes_changed", self._on_participant_attributes),
    )

  async def _await_participant(self) -> None:
    """Waits for a caller to join before the model connection is opened.

    A live connection fixes its tool declarations when it opens, and
    `LiveKitToolset` reads the room to decide what to offer. Starting first
    would hide the telephony tools for the whole call.
    """
    if self._wait_for_participant is None:
      return

    joined: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    def _on_participant_connected(participant: rtc.RemoteParticipant) -> None:
      del participant  # Any caller will do; identity is read off the room.
      if not joined.done():
        joined.set_result(None)

    # Registered before the room is inspected, so a participant arriving in
    # between is caught by the handler rather than missed by both.
    self._room.on("participant_connected", _on_participant_connected)
    try:
      if self._room.remote_participants:
        return
      logger.debug(
          "Waiting up to %ss for a caller.", self._wait_for_participant
      )
      await asyncio.wait_for(joined, self._wait_for_participant)
    except asyncio.TimeoutError:
      # An agent alone in a room is odd but not an error, so start anyway.
      logger.warning(
          "No caller joined room %s within %ss; starting the session anyway."
          " A caller arriving later is served, but tools that depend on the"
          " call kind are resolved by then.",
          self._room.name,
          self._wait_for_participant,
      )
    finally:
      with contextlib.suppress(Exception):
        self._room.off("participant_connected", _on_participant_connected)

  async def _ensure_session(self) -> None:
    """Creates the ADK session for this room if it does not exist yet.

    `Runner` raises `SessionNotFoundError` rather than creating one, and a
    freshly joined room has none.
    """
    session_service = self._runner.session_service
    session = await session_service.get_session(
        app_name=self._runner.app_name,
        user_id=self._user_id,
        session_id=self._session_id,
    )
    if session is not None:
      return
    try:
      await session_service.create_session(
          app_name=self._runner.app_name,
          user_id=self._user_id,
          session_id=self._session_id,
          state=self._caller_state(),
      )
    except AlreadyExistsError:
      # A retried dispatch can land two workers on one room; the loser just
      # uses the session the winner made.
      logger.debug("Session %s already created concurrently.", self._session_id)

  # -- Caller identity -------------------------------------------------------

  def _caller_state(self) -> dict[str, Any]:
    """Session state describing the caller, copied from SIP attributes.

    Empty unless the call arrived over SIP.
    """
    participant = self._call.sip_participant
    if participant is None:
      return {}
    attributes = self._call.sip_attributes()
    return {
        "livekit_is_phone_call": True,
        "livekit_caller_identity": participant.identity,
        "livekit_caller_phone_number": attributes.get(
            SIP_PHONE_NUMBER_ATTRIBUTE
        ),
        "livekit_called_phone_number": attributes.get(
            SIP_TRUNK_PHONE_NUMBER_ATTRIBUTE
        ),
        "livekit_sip_call_id": attributes.get(SIP_CALL_ID_ATTRIBUTE),
        "livekit_sip_attributes": attributes,
    }

  def _on_participant_attributes(
      self,
      changed: dict[str, str],
      participant: rtc.Participant,
  ) -> None:
    """Refreshes caller state when LiveKit fills in SIP attributes late.

    Attributes mapped from `X-*` SIP headers arrive asynchronously, so they
    are routinely absent when the participant joins.
    """
    del changed  # Every attribute is re-read from the participant.
    # The agent's own `lk.agent.state` writes come back through here too.
    if participant is self._room.local_participant:
      return
    state = self._caller_state()
    if not state or state == self._last_caller_state:
      return
    self._last_caller_state = state
    self._queue.send(LiveRequest(state_delta=state))

  # -- Lifecycle ------------------------------------------------------------

  def _on_disconnected(self, *args: Any) -> None:
    """Ends the session when this participant's room connection drops."""
    del args  # LiveKit passes a disconnect reason on some versions.
    logger.info("Room disconnected; ending live session.")
    self._end_session()

  def _on_participant_disconnected(self, *args: Any) -> None:
    """Ends the session once the last remote participant has left.

    Dispatch tears a worker down on its own, but a room joined directly would
    leave the agent alone in it holding a model connection open.
    """
    del args  # The departing participant; identity is not needed here.
    if not self._room.remote_participants:
      logger.info("Last participant left; ending live session.")
      self._end_session()

  def _end_session(self) -> None:
    """Signals `start` to stop driving the session.

    Safe from any thread: ADK runs sync tools on a pool, and `asyncio.Event`
    is not thread-safe.
    """
    loop = self._loop
    if loop is None or _running_loop() is loop:
      self._ended.set()
    else:
      loop.call_soon_threadsafe(self._ended.set)

  # -- Bridge 1: inbound (room media track -> LiveRequestQueue) --------------

  def _on_track_subscribed(
      self,
      track: rtc.Track,
      publication: rtc.TrackPublication,
      participant: rtc.RemoteParticipant,
  ) -> None:
    """Spawns a forwarder when a remote participant publishes a track."""
    self._spawn_forwarder(
        track, track_sid=publication.sid, identity=participant.identity
    )

  def _subscribe_existing_tracks(self) -> None:
    """Forwards tracks already present when the worker joined the room."""
    for participant in self._room.remote_participants.values():
      for publication in participant.track_publications.values():
        if publication.track is not None:
          self._spawn_forwarder(
              publication.track,
              track_sid=publication.sid,
              identity=participant.identity,
          )

  def _spawn_forwarder(
      self,
      track: rtc.Track,
      *,
      track_sid: Optional[str] = None,
      identity: Optional[str] = None,
  ) -> None:
    if track.kind == rtc.TrackKind.KIND_AUDIO:
      task = asyncio.create_task(
          self._forward_audio(track, track_sid=track_sid, identity=identity)
      )
    elif track.kind == rtc.TrackKind.KIND_VIDEO:
      task = asyncio.create_task(self._forward_video(track))
    else:
      logger.debug("Ignoring track of unsupported kind: %s", track.kind)
      return
    self._forward_tasks.add(task)
    task.add_done_callback(self._forward_tasks.discard)

  async def _forward_audio(
      self,
      track: rtc.Track,
      *,
      track_sid: Optional[str] = None,
      identity: Optional[str] = None,
  ) -> None:
    """Streams a room audio track into the queue as 16kHz PCM blobs.

    Args:
      track: The remote audio track to forward.
      track_sid: Attached to this speaker's transcripts, so clients can tell
        the captions apart.
      identity: Published as the sender of this speaker's transcripts.
    """
    self._segments.setdefault(
        _USER_ROLE, TranscriptSegment(track_sid=track_sid, identity=identity)
    ).bind(track_sid=track_sid, identity=identity)
    audio_stream = rtc.AudioStream(
        track, sample_rate=_INPUT_SAMPLE_RATE, num_channels=_NUM_CHANNELS
    )
    try:
      async for event in audio_stream:
        self._queue.send_realtime(
            types.Blob(
                mime_type=_AUDIO_MIME_TYPE,
                data=bytes(event.frame.data),
            )
        )
    finally:
      # The track ended, so flush the model's buffer or a server-VAD turn
      # hangs waiting for input that will never arrive.
      self._queue.send_audio_stream_end()

  async def _forward_video(self, track: rtc.Track) -> None:
    """Streams a room video track into the queue as JPEG image blobs."""
    video_stream = rtc.VideoStream(track)
    min_interval = 1.0 / _VIDEO_FRAMES_PER_SECOND
    next_frame_at = 0.0
    async for event in video_stream:
      now = time.monotonic()
      if now < next_frame_at:
        continue
      next_frame_at = now + min_interval
      # CPU-bound; off the loop so a large frame cannot stall audio.
      jpeg = await asyncio.to_thread(_encode_jpeg, event.frame)
      self._queue.send_realtime(
          types.Blob(mime_type=_VIDEO_MIME_TYPE, data=jpeg)
      )

  def _on_chat_stream(
      self, reader: rtc.TextStreamReader, participant_identity: str
  ) -> None:
    """Reads a text message off LiveKit's chat topic as a user turn."""
    # Keypad entries go out on this topic too; reading one back would feed
    # the model its own input.
    if participant_identity == self._room.local_participant.identity:
      return
    task = asyncio.create_task(self._read_chat_stream(reader))
    # Referenced, or the garbage collector can cancel the read mid-message.
    self._forward_tasks.add(task)
    task.add_done_callback(self._forward_tasks.discard)

  async def _read_chat_stream(self, reader: rtc.TextStreamReader) -> None:
    text = await reader.read_all()
    self._send_user_text(text)

  def _on_data_received(self, packet: rtc.DataPacket) -> None:
    """Forwards a text message on the ADK topic into the session.

    Superseded by `lk.chat`; kept for clients written against this topic.
    """
    if packet.topic != DATA_TOPIC:
      return
    text = _inbound_text(packet.data)
    if text:
      self._send_user_text(text)

  def _send_user_text(self, text: str) -> None:
    if not text or not text.strip():
      return
    self._queue.send_content(
        types.Content(role="user", parts=[types.Part(text=text)])
    )

  # -- Bridge 1b: inbound DTMF ----------------------------------------------

  def _on_dtmf_received(self, dtmf: rtc.SipDTMF) -> None:
    """Buffers a keypress, because forwarding each one starts a model turn."""
    digit = dtmf.digit or ""
    if not digit:
      return
    # Captured here because the flush can run from a timer, with no event.
    self._dtmf_identity = getattr(dtmf.participant, "identity", None)
    self._dtmf_digits.append(digit)
    if self._dtmf_flush_handle is not None:
      self._dtmf_flush_handle.cancel()
      self._dtmf_flush_handle = None
    if digit in _DTMF_TERMINATORS:
      self._flush_dtmf()
      return
    self._dtmf_flush_handle = asyncio.get_running_loop().call_later(
        _DTMF_IDLE_FLUSH_SECONDS, self._flush_dtmf
    )

  def _flush_dtmf(self) -> None:
    """Hands the buffered keypresses to the model as a user turn."""
    self._dtmf_flush_handle = None
    if not self._dtmf_digits:
      return
    digits = "".join(self._dtmf_digits)
    identity = self._dtmf_identity
    self._dtmf_digits.clear()
    self._dtmf_identity = None
    # Two strings on purpose: rewording the model's prompt should not rewrite
    # the conversation record.
    self._send_user_text(
        f"The caller pressed these keys on their phone keypad: {digits}"
    )
    task = asyncio.create_task(self._publish_keypad_entry(digits, identity))
    # Referenced, or the garbage collector can cancel the publish mid-write.
    self._forward_tasks.add(task)
    task.add_done_callback(self._forward_tasks.discard)

  async def _publish_keypad_entry(
      self, digits: str, identity: Optional[str]
  ) -> None:
    """Publishes a keypad entry on the chat topic, as the caller.

    LiveKit relays the tones but not the turn assembled from them, so without
    this a transcript shows the agent answering nothing. Chat rather than
    `lk.transcription`, which is for transcribed audio and names the track it
    came from; a keypress is neither.
    """
    try:
      writer = await self._room.local_participant.stream_text(
          topic=LK_CHAT_TOPIC, sender_identity=identity
      )
      await writer.write(digits)
      await writer.aclose()
    except Exception:  # pylint: disable=broad-except
      # Cosmetic; the model already has the digits.
      logger.exception("Failed to publish a keypad entry.")

  # -- Bridge 2: outbound (Event stream -> room) ----------------------------

  async def _forward_events(self) -> None:
    """Drives `run_live` and pushes agent output back into the room."""
    self._start_playback()
    try:
      events = self._runner.run_live(
          user_id=self._user_id,
          session_id=self._session_id,
          live_request_queue=self._queue,
          run_config=self._run_config,
      )
      await self._set_agent_state(_STATE_LISTENING)
      # `aclosing` so cancelling this task also unwinds `run_live` and
      # closes the model connection.
      async with contextlib.aclosing(events) as event_stream:
        await self._pump_events(event_stream)
      # Ended on its own, so let queued speech finish rather than cutting
      # the agent off mid-word.
      await self._drain_playback()
    finally:
      # Suppressed: an exception from the pump is the interesting one.
      with contextlib.suppress(Exception):
        await self._stop_playback()
      with contextlib.suppress(Exception):
        await self._close_segments()

  async def _pump_events(self, event_stream: AsyncIterator[Event]) -> None:
    """Pushes each agent event out to the room."""
    async for event in event_stream:
      # Barge-in: drop queued speech, or the agent talks over the user for
      # as long as the buffer lasts.
      if event.interrupted:
        await self._drop_pending_playback()
        await self._close_segments()
        await self._set_agent_state(_STATE_LISTENING)

      audio = _audio_out(event)
      for chunk in audio:
        self._playback.put_nowait(chunk)

      await self._publish_transcripts(event)

      payloads = _data_out(event)
      for payload in payloads:
        await self._room.local_participant.publish_data(
            payload, topic=DATA_TOPIC
        )

      await self._set_agent_state(_next_agent_state(event, bool(audio)))

  def _start_playback(self) -> None:
    """Starts the task that feeds queued speech to the room."""
    self._playback_task = asyncio.create_task(self._playback_loop())
    self._playback_task.add_done_callback(self._on_playback_done)

  def _on_playback_done(self, task: asyncio.Task[None]) -> None:
    """Ends the call if playback died, instead of going quietly mute.

    Nothing else awaits this task while the call is running.
    """
    if task.cancelled():
      return
    error = task.exception()
    if error is not None:
      logger.error("Audio playback failed; ending the call.", exc_info=error)
      self._end_session()

  async def _playback_loop(self) -> None:
    """Feeds queued speech to the room at playback pace.

    `capture_frame` blocks once the buffer is full, which is why this is not
    inline in the event pump.
    """
    while True:
      chunk = await self._playback.get()
      if isinstance(chunk, _EndOfPlayback):
        return
      await self._audio_source.capture_frame(
          rtc.AudioFrame(
              data=chunk,
              sample_rate=_OUTPUT_SAMPLE_RATE,
              num_channels=_NUM_CHANNELS,
              samples_per_channel=len(chunk) // _BYTES_PER_SAMPLE,
          )
      )

  async def _drain_playback(self) -> None:
    """Waits for every queued frame to be handed over and played out."""
    if self._playback_task is None:
      return
    await self._playback.put(_END_OF_PLAYBACK)
    await self._playback_task
    self._playback_task = None
    # The source still holds up to `queue_size_ms` after the last frame is
    # handed over, and returning now clips the agent's last words.
    await self._audio_source.wait_for_playout()

  async def _stop_playback(self) -> None:
    """Stops playback immediately, discarding anything still queued."""
    task, self._playback_task = self._playback_task, None
    if task is None:
      return
    task.remove_done_callback(self._on_playback_done)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await task

  async def _drop_pending_playback(self) -> None:
    """Discards speech that has not been played yet.

    Playback is torn down rather than drained: the task may be suspended
    inside `capture_frame` holding a chunk, and clearing the source's buffer
    is what would release it.
    """
    await self._stop_playback()
    while True:
      try:
        self._playback.get_nowait()
      except asyncio.QueueEmpty:
        break
    self._audio_source.clear_queue()
    self._start_playback()

  async def _publish_output_audio_track(self) -> None:
    publication = await self._room.local_participant.publish_track(
        self._audio_track, rtc.TrackPublishOptions()
    )
    self._output_track_sid = publication.sid

  # -- Outbound transcripts --------------------------------------------------

  async def _publish_transcripts(self, event: Event) -> None:
    """Mirrors an event's transcription onto LiveKit's transcription topic.

    Partials carry an incremental fragment and finals the whole segment,
    which maps onto LiveKit's interim/final stream pair.
    """
    for transcription, role in (
        (event.input_transcription, _USER_ROLE),
        (event.output_transcription, _AGENT_ROLE),
    ):
      if transcription is None or not transcription.text:
        continue
      await self._publish_transcript(
          role=role, text=transcription.text, final=not event.partial
      )

  async def _publish_transcript(
      self, *, role: str, text: str, final: bool
  ) -> None:
    segment = self._segments.setdefault(role, TranscriptSegment())
    if role == _AGENT_ROLE:
      segment.bind(track_sid=self._output_track_sid, identity=None)
    # Captions are cosmetic and must never take down the call.
    try:
      local = self._room.local_participant
      if final:
        await segment.finish(local, text)
      else:
        await segment.write(local, text)
    except Exception:  # pylint: disable=broad-except
      logger.exception("Failed to publish a %s transcript.", role)

  async def _close_segments(self) -> None:
    """Closes any interim transcription stream left open."""
    for role, segment in self._segments.items():
      try:
        await segment.close()
      except Exception:  # pylint: disable=broad-except
        # Logged, not suppressed: an unterminated stream leaves a caption
        # hanging open on every client, and that is the only symptom.
        logger.exception("Failed to close the %s transcript stream.", role)
      segment.reset()

  # -- Outbound agent state --------------------------------------------------

  async def _set_agent_state(self, state: Optional[str]) -> None:
    """Publishes the agent's state for LiveKit's voice-assistant UI."""
    if state is None or state == self._agent_state:
      return
    try:
      await self._room.local_participant.set_attributes(
          {_LK_AGENT_STATE_ATTRIBUTE: state}
      )
    except Exception:  # pylint: disable=broad-except
      # Cached only on success, or the next identical transition would
      # short-circuit and strand the client's indicator.
      logger.exception("Failed to publish agent state %s.", state)
      return
    self._agent_state = state


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
  """Returns the loop running on this thread, or None if there is none."""
  try:
    return asyncio.get_running_loop()
  except RuntimeError:
    return None


def _next_agent_state(event: Event, has_audio: bool) -> Optional[str]:
  """Maps an event to the state LiveKit clients should show, if it changed."""
  if event.turn_complete:
    return _STATE_LISTENING
  if has_audio or (
      event.output_transcription and event.output_transcription.text
  ):
    return _STATE_SPEAKING
  if event.get_function_calls():
    return _STATE_THINKING
  return None


def _with_live_defaults(run_config: Optional[RunConfig]) -> RunConfig:
  """Completes a run config with the settings a voice call needs.

  Only unset fields are filled, so a caller's own choices survive and a config
  supplied for one reason does not silently drop the voice basics.

  Two fields need filling. `response_modalities`, because the transport
  carries speech and ADK's own default is text. `session_resumption`, because
  the history ADK replays on reconnect is assembled before the call, so
  without a handle a reconnect loses the conversation.

  Transcription is deliberately not touched. Both `RunConfig` transcription
  fields already default to enabled, which is what feeds `lk.transcription`,
  so `None` here is a caller switching captions off rather than a caller who
  forgot -- and overriding that would leave no way to switch them off at all.

  Args:
    run_config: The caller's config, or None to build one from scratch.

  Returns:
    A new config; the caller's own instance is left untouched.
  """
  run_config = run_config or RunConfig()
  updates: dict[str, Any] = {}

  if run_config.response_modalities is None:
    updates["response_modalities"] = [types.Modality.AUDIO]
  if run_config.session_resumption is None:
    updates["session_resumption"] = types.SessionResumptionConfig()

  return run_config.model_copy(update=updates)


def _encode_jpeg(frame: rtc.VideoFrame) -> bytes:
  """Encodes a room video frame as a downscaled JPEG.

  `VideoFrame.convert` only changes pixel layout, so the frame needs a real
  encoder before it can be sent as `image/jpeg`.
  """
  if frame.type != rtc.VideoBufferType.RGBA:
    frame = frame.convert(rtc.VideoBufferType.RGBA)
  image = Image.frombytes(
      "RGBA", (frame.width, frame.height), bytes(frame.data)
  ).convert("RGB")
  image.thumbnail((_VIDEO_MAX_WIDTH, _VIDEO_MAX_HEIGHT))
  buffer = io.BytesIO()
  image.save(buffer, "JPEG", quality=_VIDEO_JPEG_QUALITY)
  return buffer.getvalue()


def _inbound_text(data: bytes) -> Optional[str]:
  """Extracts the text of an inbound `{"type": "text", "text": ...}` message."""
  with contextlib.suppress(UnicodeDecodeError, json.JSONDecodeError):
    message = json.loads(data.decode("utf-8"))
    if isinstance(message, dict) and message.get("type") == "text":
      text = message.get("text")
      if isinstance(text, str) and text:
        return text
  logger.debug("Ignoring unrecognized inbound data message.")
  return None


def _audio_out(event: Event) -> list[bytes]:
  """Extracts raw output audio (24kHz PCM) from an event, if any."""
  if not (event.content and event.content.parts):
    return []
  blobs: list[bytes] = []
  for part in event.content.parts:
    inline_data = part.inline_data
    if (
        inline_data is not None
        and inline_data.data
        and (inline_data.mime_type or "").startswith("audio/")
    ):
      blobs.append(inline_data.data)
  return blobs


def _data_out(event: Event) -> list[bytes]:
  """Extracts tool activity from an event, for the ADK data topic.

  Each payload carries the ADK function call id, which is what pairs a result
  with its call. See README.md.
  """
  payloads: list[bytes] = []
  if not (event.content and event.content.parts):
    return payloads

  for part in event.content.parts:
    if part.function_call is not None:
      payloads.append(
          _encode({
              "type": "function_call",
              "id": part.function_call.id,
              "name": part.function_call.name,
              "args": part.function_call.args,
          })
      )
    elif part.function_response is not None:
      payloads.append(
          _encode({
              "type": "function_response",
              "id": part.function_response.id,
              "name": part.function_response.name,
              "response": part.function_response.response,
          })
      )
  return payloads


def _encode(payload: dict[str, Any]) -> bytes:
  return json.dumps(payload).encode("utf-8")
