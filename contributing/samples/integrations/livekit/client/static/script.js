// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Browser client for the ADK x LiveKit dice agent, on LiveKit's own client SDK.
// Everything but `handleAdkData` rides LiveKit's standard channels, so this
// file works against any LiveKit agent.

const { Room, RoomEvent, Track } = LivekitClient;

// LiveKit's own channels. Every LiveKit client SDK speaks these.
const TRANSCRIPTION_TOPIC = 'lk.transcription';
const CHAT_TOPIC = 'lk.chat';
const AGENT_STATE_ATTRIBUTE = 'lk.agent.state';

// Must match `DATA_TOPIC` in google.adk.integrations.livekit.
const ADK_TOPIC = 'adk';

const talkBtn = document.getElementById('talk');
const cameraBtn = document.getElementById('camera');
const cameraRow = document.getElementById('camera-row');
const previewEl = document.getElementById('preview');
const stateEl = document.getElementById('state');
const dotEl = document.getElementById('dot');
const logEl = document.getElementById('log');
const audioContainer = document.getElementById('audio-container');
const transcriptEl = document.getElementById('transcript');
const sayForm = document.getElementById('say-form');
const sayInput = document.getElementById('say');
const sendBtn = document.getElementById('send');
const toolsEl = document.getElementById('tools');
const toolsEmptyEl = document.getElementById('tools-empty');

const decoder = new TextDecoder();

let room = null;
// Keyed by LiveKit segment id, so a final caption replaces its interim.
const captions = new Map();
// Keyed by ADK function call id, so a result lands on the call it answers.
const toolCalls = new Map();

/**
 * Appends a timestamped line to the connection log.
 * @param {string} message Connection or error text; tool activity has its own
 *     panel.
 */
function log(message) {
  const time = new Date().toLocaleTimeString();
  logEl.textContent += `[${time}] ${message}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

/**
 * Shows a plain connection status and tints the indicator dot.
 * @param {string} label Status text to display.
 * @param {boolean} connected Whether the room is connected.
 */
function setState(label, connected) {
  stateEl.textContent = label;
  dotEl.classList.toggle('connected', Boolean(connected));
}

/**
 * Shows what the agent is doing and animates the indicator dot to match.
 * @param {string} agentState One of `listening`, `thinking` or `speaking`.
 */
function setAgentState(agentState) {
  stateEl.textContent = `Agent is ${agentState}`;
  dotEl.classList.remove('speaking', 'thinking');
  if (agentState === 'speaking' || agentState === 'thinking') {
    dotEl.classList.add(agentState);
  }
}

/**
 * Reports whether an identity belongs to the agent.
 *
 * Asked of the participant rather than inferred from "not me", which breaks as
 * soon as a third party joins the room.
 * @param {string} identity Participant identity to test.
 * @returns {boolean} True if that participant is the agent.
 */
function isAgentIdentity(identity) {
  return Boolean(room?.getParticipantByIdentity(identity)?.isAgent);
}

/**
 * Appends an empty transcript line, styled for whichever side is speaking.
 * @param {boolean} isAgent Whether the agent is the speaker.
 * @returns {!HTMLParagraphElement} The line, ready for text.
 */
function newTranscriptLine(isAgent) {
  transcriptEl.querySelector('.placeholder')?.remove();
  const line = document.createElement('p');
  line.className = `line line-${isAgent ? 'agent' : 'user'}`;
  transcriptEl.appendChild(line);
  return line;
}

/**
 * Renders one transcription segment, replacing its interim text when final.
 *
 * Each utterance arrives as an interim stream then a final one, sharing a
 * segment id. `lk.transcription_final` says which is which.
 * @param {!TextStreamReader} reader Stream carrying this segment.
 * @param {{identity: string}} participantInfo Who is speaking.
 */
async function handleTranscription(reader, participantInfo) {
  const { attributes } = reader.info;
  const segmentId = attributes['lk.segment_id'] ?? reader.info.id;
  const isAgent = isAgentIdentity(participantInfo.identity);

  let line = captions.get(segmentId);
  if (!line) {
    line = newTranscriptLine(isAgent);
    captions.set(segmentId, line);
  }

  // The final stream carries the whole utterance, so replace rather than
  // append: draining it as chunks would render the text twice.
  if (attributes['lk.transcription_final'] === 'true') {
    line.textContent = await reader.readAll();
    captions.delete(segmentId);
  } else {
    for await (const chunk of reader) {
      line.textContent += chunk;
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }
  }
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

/**
 * Renders a typed turn or a phone keypad entry.
 *
 * Our own messages are echoed locally on send, since LiveKit does not deliver
 * a stream back to its sender.
 * @param {!TextStreamReader} reader Stream carrying the message.
 * @param {{identity: string}} participantInfo Who sent it.
 */
async function handleChat(reader, participantInfo) {
  if (participantInfo.identity === room?.localParticipant?.identity) {
    return;
  }
  const text = await reader.readAll();
  if (text) {
    newTranscriptLine(isAgentIdentity(participantInfo.identity)).textContent =
      text;
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
  }
}

/**
 * Mirrors the agent's published state into the UI.
 *
 * The same attribute LiveKit's React components read to drive waveforms.
 * @param {!Object<string, string>} changed Attributes that just changed.
 */
function handleAgentState(changed) {
  const state = changed[AGENT_STATE_ATTRIBUTE];
  if (state) {
    setAgentState(state);
  }
}

/**
 * Routes tool activity into the panel. The only ADK-specific handler.
 *
 * Calls and results arrive as JSON `{type, id, name, args|response}`. Pair the
 * two halves on `id`, not `name`, which collides once a tool is called twice
 * in a turn.
 * @param {!Uint8Array} payload Raw JSON body.
 * @param {*} _participant Unused; part of LiveKit's `DataReceived` signature.
 * @param {*} _kind Unused; part of LiveKit's `DataReceived` signature.
 * @param {string} topic Data topic; anything but `adk` is ignored.
 */
function handleAdkData(payload, _participant, _kind, topic) {
  if (topic !== ADK_TOPIC) {
    return;
  }
  let message;
  try {
    message = JSON.parse(decoder.decode(payload));
  } catch (err) {
    log(`Ignoring unparseable data message: ${err}`);
    return;
  }
  if (message.type === 'function_call') {
    addToolCall(message);
  } else if (message.type === 'function_response') {
    resolveToolCall(message);
  }
}

/**
 * Builds the key that pairs a tool result with the call it answers.
 * @param {{id: ?string, name: string}} message Tool call or result.
 * @returns {string} The call id, or a name-based fallback.
 */
function toolKey(message) {
  return message.id ?? `${message.name}:no-id`;
}

/**
 * Adds a pending row to the tool panel for an outgoing call.
 * @param {{id: ?string, name: string, args: *}} message The tool call.
 */
function addToolCall(message) {
  toolsEmptyEl.hidden = true;

  const row = document.createElement('li');
  row.className = 'tool tool-pending';
  row.innerHTML = `
    <div class="tool-head">
      <span class="tool-name"></span>
      <span class="tool-status">calling…</span>
    </div>
    <pre class="tool-args"></pre>
    <pre class="tool-result" hidden></pre>`;
  // textContent, not innerHTML: tool names and arguments come from the model.
  row.querySelector('.tool-name').textContent = message.name;
  row.querySelector('.tool-args').textContent = formatJson(message.args);

  toolsEl.appendChild(row);
  toolCalls.set(toolKey(message), row);
  toolsEl.scrollTop = toolsEl.scrollHeight;
}

/**
 * Fills in the result on the row for the call it answers.
 * @param {{id: ?string, name: string, response: *}} message The tool result.
 */
function resolveToolCall(message) {
  const row = toolCalls.get(toolKey(message));
  if (!row) {
    // No call on screen: page opened mid-session. Show it rather than drop it.
    addToolCall({ ...message, args: undefined });
    resolveToolCall(message);
    return;
  }
  toolCalls.delete(toolKey(message));

  row.classList.remove('tool-pending');
  row.classList.add('tool-done');
  row.querySelector('.tool-status').textContent = 'returned';

  const result = row.querySelector('.tool-result');
  result.textContent = formatJson(message.response);
  result.hidden = false;
  toolsEl.scrollTop = toolsEl.scrollHeight;
}

/**
 * Renders a value as JSON: compact when it fits on a line, indented otherwise.
 * @param {*} value Tool arguments or response.
 * @returns {string} Display text, empty when there is nothing to show.
 */
function formatJson(value) {
  if (value === undefined) {
    return '';
  }
  try {
    const compact = JSON.stringify(value);
    return compact.length <= 56 ? compact : JSON.stringify(value, null, 2);
  } catch (err) {
    return String(value);
  }
}

/**
 * Fetches a token, joins the room, and turns the microphone on.
 */
async function connect() {
  talkBtn.disabled = true;
  setState('Connecting…', false);
  log('Requesting a room and an access token…');

  const resp = await fetch('/token');
  if (!resp.ok) {
    setState('Error', false);
    log(`Token request failed: ${resp.status} ${await resp.text()}`);
    talkBtn.disabled = false;
    return;
  }
  const { url, token, room: roomName } = await resp.json();
  log(`Joining room "${roomName}"…`);

  room = new Room();

  room.on(RoomEvent.TrackSubscribed, (track) => {
    if (track.kind === Track.Kind.Audio) {
      log('Agent audio connected.');
      audioContainer.appendChild(track.attach());
    }
  });

  room.on(RoomEvent.ParticipantAttributesChanged, handleAgentState);
  room.on(RoomEvent.DataReceived, handleAdkData);

  room.on(RoomEvent.ParticipantConnected, (participant) => {
    log(`Participant joined: ${participant.identity}`);
  });

  room.on(RoomEvent.Disconnected, () => {
    setState('Call ended', false);
    dotEl.classList.remove('speaking', 'thinking');
    log('Disconnected from the room.');
    captions.clear();
    toolCalls.clear();
    talkBtn.disabled = false;
    talkBtn.textContent = 'Start talking';
    sayInput.disabled = true;
    sendBtn.disabled = true;
    cameraRow.hidden = true;
    previewEl.hidden = true;
    cameraBtn.textContent = 'Turn on camera';
  });

  // Both arrive as text streams, so register before connecting.
  room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, handleTranscription);
  room.registerTextStreamHandler(CHAT_TOPIC, handleChat);

  await room.connect(url, token);
  setState('Connected', true);
  log('Connected. Enabling microphone…');

  await room.localParticipant.setMicrophoneEnabled(true);
  setState('Ready', true);
  log('Microphone live. Speak, or type below.');

  talkBtn.textContent = 'Hang up';
  talkBtn.disabled = false;
  sayInput.disabled = false;
  sendBtn.disabled = false;
  // Offered, not enabled: most of this demo needs no camera permission.
  cameraRow.hidden = false;
}

/**
 * Turns the camera on or off.
 *
 * Publishing the track is all the client does; the connector samples it.
 */
async function toggleCamera() {
  if (!room) {
    return;
  }
  cameraBtn.disabled = true;
  const turningOn = !room.localParticipant.isCameraEnabled;
  try {
    await room.localParticipant.setCameraEnabled(turningOn);
    showPreview(turningOn);
    cameraBtn.textContent = turningOn ? 'Turn off camera' : 'Turn on camera';
    log(
      turningOn
        ? 'Camera on. Try asking what the agent can see.'
        : 'Camera off.',
    );
  } catch (err) {
    // Denied permission or no device. Neither should kill the call.
    log(
      `Could not toggle the camera: ${err && err.message ? err.message : err}`,
    );
  } finally {
    cameraBtn.disabled = false;
  }
}

/**
 * Shows or hides the local preview of what the agent is being sent.
 * @param {boolean} visible Whether to attach the camera track.
 */
function showPreview(visible) {
  const publication = room?.localParticipant?.getTrackPublication(
    Track.Source.Camera,
  );
  const track = publication?.videoTrack;
  if (visible && track) {
    track.attach(previewEl);
    previewEl.hidden = false;
  } else {
    if (track) {
      track.detach(previewEl);
    }
    previewEl.hidden = true;
  }
}

/**
 * Leaves the room, if one is connected.
 */
async function disconnect() {
  if (room) {
    await room.disconnect();
    room = null;
  }
}

sayForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = sayInput.value.trim();
  if (!text || !room) {
    return;
  }
  sayInput.value = '';
  try {
    // LiveKit's standard chat channel; no ADK-specific encoding needed.
    await room.localParticipant.sendText(text, { topic: CHAT_TOPIC });
  } catch (err) {
    // Not echoed to the transcript, because the agent never received it.
    log(`Could not send "${text}": ${err && err.message ? err.message : err}`);
    sayInput.value = text;
    return;
  }
  newTranscriptLine(false).textContent = text;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
});

cameraBtn.addEventListener('click', toggleCamera);

talkBtn.addEventListener('click', async () => {
  if (room) {
    await disconnect();
  } else {
    try {
      await connect();
    } catch (err) {
      setState('Error', false);
      log(`Error: ${err && err.message ? err.message : err}`);
      talkBtn.disabled = false;
    }
  }
});
