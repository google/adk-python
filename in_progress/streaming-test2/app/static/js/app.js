// DOM Elements
const camStatus = document.getElementById('cameraAgentStatus');
const camLog = document.getElementById('cameraLogOutput');
const audStatus = document.getElementById('audioAgentStatus');
const audLog = document.getElementById('audioLogOutput');
const startBtn = document.getElementById('startAudioButton');
const stopBtn = document.getElementById('stopAudioButton');
const textIn = document.getElementById('textInput');
const sendBtn = document.getElementById('sendTextButton');
const agentSpeech = document.getElementById('currentAgentResponse');

// Import the audio worklet helpers from ADK example
import { startAudioPlayerWorklet } from "./audio-player.js";
import { startAudioRecorderWorklet, stopMicrophone } from "./audio-recorder.js";

let sse = null;
let audioPlayerNode;
let audioPlayerContext;
let audioRecorderNode;
let audioRecorderContext;
let micStream;

let is_audio = false; // Tracks audio mode
const sessId = Date.now().toString() + Math.random().toString().substring(3);
console.log("Client Session ID:", sessId);

// --- Utility Functions ---
function b64ToAb(b64) { const s = atob(b64); const l = s.length; const u = new Uint8Array(l); while(l--) u[l] = s.charCodeAt(l); return u.buffer; }
function abToB64(ab) { let r = ''; const b = new Uint8Array(ab); const l = b.byteLength; for(let i=0;i<l;i++) r+=String.fromCharCode(b[i]); return btoa(r); }

// --- Log Fetching ---
async function fetchLog(type, el) {
    try {
        const res = await fetch(`/logs/${type}`);
        if(!res.ok) { el.textContent = `Error: ${res.statusText}`; return; }
        const logs = await res.json();
        if(logs.error) el.textContent = `Error in log data: ${logs.error}`;
        else if(Array.isArray(logs)) el.textContent = logs.map(l => `[${l.timestamp || ''}] ${l.speaker || l.type || ''}: ${l.text || l.comment_by_llm || l.description_by_cv || ''}`).join('\n');
        else el.textContent = "Bad log format.";
    } catch (e) { el.textContent = `Failed to fetch ${type} log: ${e}`; }
}
async function fetchCamStatus() { try { const r = await fetch('/agent/camera/status'); const d = await r.json(); camStatus.textContent = `Status: ${d.status}${d.monitoring?' (Monitoring)':''}`; } catch (e) { camStatus.textContent = "Status: Error"; }}

// --- SSEConnection for Audio Agent ---
function connectSSE() {
    if(sse) sse.close();
    const sseUrl = `/events/${sessId}?is_audio=${is_audio}`; // is_audio is a global
    sse = new EventSource(sseUrl);
    sse.onopen = () => {
        console.log("is_audio = ", is_audio, "SSE connection opened.");
        audStatus.textContent = is_audio?"Audio Connected, Listening...":"Text Connected";
        startBtn.disabled=true; // Start button now disabled
        stopBtn.disabled=false; // Stop button enabled
        textIn.disabled=false;
        sendBtn.disabled=false;
        agentSpeech.textContent="";
    };
    sse.onmessage = (evt) => {
        const msg = JSON.parse(evt.data); // console.log("incoming", msg);
        if(msg.error) { console.error("SSE Server Error:", msg.error); audStatus.textContent = `Error: ${msg.error}`; agentSpeech.textContent = `Error: ${msg.error}`; return; }
        if(msg.turn_complete) {
            console.log("Agent turn done.");
            if(audStatus.textContent.includes("Speaking") || audStatus.textContent.includes("Responding")) {
                audStatus.textContent = is_audio?"Audio Connected, Listening...":"Text Connected";
            }
            return;
        }
        if(!is_audio && msg.mime_type === "text/plain") { // Text only mode
            agentSpeech.textContent += msg.data;
            audStatus.textContent = "Agent Responding (text)...";
        } else if(is_audio) {
            if((msg.mime_type==="audio/pcm" || msg.mime_type==="audio/opus") && audioPlayerNode && msg.data) {
                audioPlayerNode.port.postMessage(b64ToAb(msg.data));
                audStatus.textContent="Agent Speaking...";
                agentSpeech.textContent=""; // Clear text when audio comes
            } else if(msg.mime_type==="text/plain") { // Text accompanying audio or transcription
                // Don't change audioStatus if already speaking, just show transcription
                agentSpeech.textContent += msg.data;
            }
        }
        fetchLog('audio', audLog);
    };
    sse.onerror = (e) => {
        console.error("SSE connection error or closed.", e);
        audStatus.textContent = "Status: Connection Error. Retrying...";
        if(sse) sse.close();
        // Automatic reconnect without user intervention
        setTimeout(() => connectSSE(), 3000);
    };
}

// --- Sending Messages ---
async function sendAudio(ab) { // ab is ArrayBuffer
    if(!sse || sse.readyState !== EventSource.OPEN) { console.warn("SSE not open. Cannot send audio."); return; }
    try { const r = await fetch(`/send/${sessId}`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({mime_type: 'audio/pcm', data:abToB64(ab)})}); if(!r.ok) console.error("Send audio error:", r.statusText); } catch (e) { console.error("Send audio exception:", e); }
}
async function sendText(txt) {
    if(!sse || sse.readyState !== EventSource.OPEN) { alert("Not connected to audio agent!"); return; }
    if(!txt.trim()) return;
    try { const r = await fetch(`/send/${sessId}`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({mime_type: 'text/plain', data:txt})}); if(!r.ok) alert(`Send error: ${r.statusText}`); else agentSpeech.textContent=""; } catch (e) { alert(`Send exception: ${e}`);}
    textIn.value = "";
}

// --- Audio Handling ---
// Callback for the audio recorder worklet
function audioRecorderHandler(pcmData) { // pcmData is an ArrayBuffer from audio-recorder.js
sendAudio(pcmData);
}

async function startAudioFlow() {
    is_audio = true;
    audStatus.textContent = "Status: Initializing audio...";
    try {
        if(!audioPlayerNode || !audioPlayerContext) {
            [audioPlayerNode, audioPlayerContext] = await startAudioPlayerWorklet();
        }
        if(!audioRecorderNode || !audioRecorderContext || !micStream) {
            [audioRecorderNode, audioRecorderContext, micStream] = await startAudioRecorderWorklet(audioRecorderHandler);
        }

        connectSSE(); // Connect SSE, is_audio is true by default or already set
        audStatus.textContent = "Status: Microphone active, recording...";
        startBtn.disabled = true;
        stopBtn.disabled = false;
        textIn.disabled = false;
        sendBtn.disabled = false;
    } catch (e) {
        console.error("Error starting audio flow: ", e);
        audStatus.textContent = "Error: Failed to start audio.";
        alert("Could not start audio. Check console and microphone permissions.");
        is_audio = false; // Reset if failed
    }
}

function stopAudioFlow() {
    is_audio = false;
    if(micStream) {
        stopMicrophone(micStream);
        micStream = null; // Reset micStream
    }
    if(audioRecorderNode) {
        audioRecorderNode.disconnect();
        // AudioRecorderNode and context might be reused or should be closed if no longer needed
    }
    if(sse) {
        sse.close();
        console.log("is_audio = ", is_audio, "SSE connection closed by client.");
    }
    audStatus.textContent = "Status: Idle";
    startBtn.disabled = false;
    stopBtn.disabled = true;
    textIn.disabled = true;
    sendBtn.disabled = true;
}

// --- Event Listeners ---
startBtn.addEventListener('click', startAudioFlow);
stopBtn.addEventListener('click', stopAudioFlow);

sendBtn.addEventListener('click', () => {
    if(!sse || sse.readyState !== EventSource.OPEN && is_audio ){
        // If audio is intended but not connected, try to start it
        console.warn("Audio not connected, attempting to start before sending text.");
        startAudioFlow(); // Attempt to re-establish
        setTimeout(() => sendText(textIn.value), 1000); // Try sending after a delay
        return;
    }
    sendText(textIn.value);
});
textIn.addEventListener('keypress', (e) => {
    if(e.key==='Enter') { sendText(textIn.value); }
});

// --- Initialization ---
function init() {
    console.log("App init, session:", sessId);
    fetchLog('camera',camLog); fetchLog('audio',audLog); fetchCamStatus();
    setInterval(()=>{ fetchLog('camera',camLog); fetchLog('audio',audLog); fetchCamStatus(); }, 5000);
    // Don't automatically setup audio or connect SSE on load.
    // User needs to click 'Start Audio Interaction'
    // setupAudio();
    audStatus.textContent = "Status: Idle";
}
init();
