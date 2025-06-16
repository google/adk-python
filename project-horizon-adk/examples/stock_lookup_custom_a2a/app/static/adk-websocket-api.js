// app/static/adk-websocket-api.js

// Ensure EventEmitter3 is loaded (e.g., via CDN in index.html)
const EventEmitter = window.EventEmitter3;

// --- Assume these classes are available globally or via imports ---
// --- You MUST copy the corresponding .js files from Pastra   ---
// --- into ./audio/ and ./utils/                           ---
import { AudioRecorder } from './audio/audio-recorder.js';
import { AudioStreamer } from './audio/audio-streamer.js';
import { base64ToArrayBuffer } from './utils/utils.js';
// ---------------------------------------------------------------

class ADKWebSocketAPI extends EventEmitter {
    constructor() {
        super(); // Initialize EventEmitter
        this.ws = null;
        this.audioContext = null; // Keep initialization lazy but ensure resumed
        this.audioRecorder = new AudioRecorder();
        this.audioStreamer = null;
        this.isRecording = false;
        this.isMuted = false; // Start unmuted
        this.isConnected = false;
        this.sessionID = null;
        this.statusIndicator = document.getElementById('statusIndicator');
        this.chatOutput = document.getElementById('chatOutput');
        this.connectButton = document.getElementById('connectButton');
        this.stopButton = document.getElementById('stopButton');
        this.micButton = document.getElementById('micButton');
        this.connectContainer = document.getElementById('connectButtonContainer');
        this.mediaContainer = document.getElementById('mediaButtonsContainer');

        this._bindUIEvents();
        console.log("ADKWebSocketAPI initialized."); // Log constructor finish
    }

    _updateStatusIndicator(status) { // 'offline', 'connecting', 'online', 'error'
        this.statusIndicator.className = `status-indicator ${status}`;
        this.statusIndicator.title = status.charAt(0).toUpperCase() + status.slice(1);
    }

    _logToUI(message, type = 'info') { // type: 'user', 'agent', 'info', 'error'
        const p = document.createElement('p');
        p.textContent = message;
        p.className = `${type}-message`; // Apply class based on type
        this.chatOutput.appendChild(p);
        // Auto-scroll to bottom
        this.chatOutput.scrollTop = this.chatOutput.scrollHeight;
    }

    // *** CHANGE 1: Make initializeAudio async ***
    async initializeAudio() {
        try {
            if (!this.audioContext) {
                console.log("Creating AudioContext...");
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 24000 // Matches streamer and likely ADK output
                });
                console.log("AudioContext created, state:", this.audioContext.state);
            }
            // Resume context on user interaction (required by browsers)
            if (this.audioContext.state === 'suspended') {
                console.log("Attempting to resume suspended AudioContext...");
                await this.audioContext.resume();
                console.log("AudioContext resumed, new state:", this.audioContext.state);
            }
            // Add log right before creating streamer
            if (!this.audioStreamer && this.audioContext) {
                 console.log(`Creating AudioStreamer with context state: ${this.audioContext.state}`);
                 this.audioStreamer = new AudioStreamer(this.audioContext);
                 this.audioStreamer.onComplete = () => console.log("AudioStreamer: Playback complete.");
                 console.log("AudioStreamer created.");
            }
            console.log("Audio initialization/resume complete.");
            return true; // Indicate success
        } catch (error) {
             console.error("Failed to initialize audio:", error);
             this._logToUI(`Error initializing audio: ${error.message}`, "error");
             return false; // Indicate failure
        }
    }

    // *** CHANGE 2: Modify connect to call initializeAudio ***
    async connect() { // Make connect async
        this.sessionID = uuid.v4();
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Use the port the current page is served from
        const backendPort = window.location.port || (wsProtocol === "wss:" ? "443" : "80"); 
        const wsEndpoint = `${wsProtocol}//${window.location.hostname}:${backendPort}/ws/${this.sessionID}`;

        console.log('Attempting to connect WebSocket to:', wsEndpoint);
        this._logToUI("Connecting to agent...", "info");
        this._updateStatusIndicator("connecting");

        return new Promise((resolve, reject) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                console.warn("WebSocket already open.");
                resolve();
                return;
            }

            try {
                 this.ws = new WebSocket(wsEndpoint);
            } catch (error) {
                 console.error("WebSocket creation failed:", error);
                 this._logToUI(`WebSocket Error: ${error.message}`, "error");
                 this._updateStatusIndicator("error");
                 reject(error);
                 return;
            }


            this.ws.onopen = async () => { // Make onopen async
                console.log('WebSocket connection established.');
                this.isConnected = true;

                console.log("Attempting to initialize audio system on connect...");
                const audioInitialized = await this.initializeAudio();
                console.log(`Audio initialization result: ${audioInitialized}`);
                console.log(`AudioContext state after init: ${this.audioContext?.state}`); // Log state again

                if (audioInitialized) {
                    this._updateStatusIndicator("online"); // Set status *after* successful audio init
                    this._logToUI("Connected! Audio ready. Press Mic to talk.", "info");
                    // Update UI: Hide Connect, Show Controls
                    this.connectContainer.classList.add('hidden');
                    this.mediaContainer.classList.remove('hidden');
                    this.micButton.disabled = false; // Enable mic button
                    resolve(); // Resolve the promise after audio is ready
                } else {
                    this._updateStatusIndicator("error"); // Show error if audio failed
                     this._logToUI("Connected, but failed to initialize audio playback.", "error");
                     // Still show connected, but warn user playback might fail
                     this.connectContainer.classList.add('hidden');
                     this.mediaContainer.classList.remove('hidden');
                     this.micButton.disabled = false; // Mic might still work for input
                     // Reject or resolve based on whether audio playback is critical
                     reject(new Error("Audio initialization failed")); // Reject if audio must work
                }
            };

            this.ws.onmessage = async (event) => { // Keep async
                try {
                    const message = JSON.parse(event.data);
                    console.log(`WebSocket Message Received: Type=${message.type}, ContextState=${this.audioContext?.state}`); // Log type and state

                    if (message.type === 'audio') {
                        const canPlay = !!this.audioStreamer && this.audioContext?.state === 'running';
                        console.log(`Audio message received. Can play? ${canPlay}`); // Log the check result

                        if (canPlay) {
                            const audioBytes = base64ToArrayBuffer(message.data);
                            console.log(`Queuing ${audioBytes.byteLength} audio bytes for playback.`);
                            this.audioStreamer.addPCM16(new Uint8Array(audioBytes));
                        } else {
                             console.warn("Audio streamer not ready or context not running. Cannot play audio chunk.", { streamer: !!this.audioStreamer, contextState: this.audioContext?.state });
                             // Try resuming again *just in case*
                             if(this.audioContext && this.audioContext.state === 'suspended') {
                                 console.log("Attempting to resume suspended context during message handling...");
                                 await this.audioContext.resume();
                                 console.log(`Context state after resume attempt: ${this.audioContext.state}`);
                                 // If resume succeeded, maybe try adding the chunk again? Careful about race conditions.
                                 if (this.audioContext.state === 'running' && this.audioStreamer) {
                                     console.log("Context resumed, retrying addPCM16 for the missed chunk.");
                                     const audioBytes = base64ToArrayBuffer(message.data);
                                     this.audioStreamer.addPCM16(new Uint8Array(audioBytes));
                                 }
                             }
                        }
                    } else if (message.type === 'text') {
                        console.log("Received text message:", message.data);
                        this._logToUI(message.data, "agent");
                    } else if (message.type === 'interrupted') {
                        console.log("Received interruption signal from server.");
                        if (this.audioStreamer) {
                            console.log("Stopping audio streamer due to interruption.");
                            this.audioStreamer.stop();
                        }
                        this._logToUI("Agent response interrupted.", "info");
                    } else if (message.type === 'turn_complete') {
                         console.log("Received turn_complete message.");
                         if (this.audioStreamer) this.audioStreamer.complete();
                    } else if (message.type === 'error') {
                        console.error("Received error message:", message.data);
                        this._logToUI(`Server Error: ${message.data}`, "error");
                    }

                } catch (error) {
                    console.error('Error processing WebSocket message:', error);
                    this._logToUI("Error processing server message.", "error");
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket Error:', error);
                this.isConnected = false;
                this._updateStatusIndicator("error");
                // Try to log the specific event if possible, might give clues
                let errorMsg = "Connection error.";
                // Check if it's a proper ErrorEvent (browser standard)
                if (error instanceof ErrorEvent && error.message) {
                    errorMsg = `Connection error: ${error.message}`;
                } else if (typeof error === 'string') {
                     errorMsg = `Connection error: ${error}`;
                }
                 console.error("Detailed WebSocket error event:", error); // Log the raw event
                this._logToUI(errorMsg, "error");
                this._resetUI();
                reject(error instanceof Error ? error : new Error(errorMsg)); // Reject promise on error
            };


            this.ws.onclose = (event) => {
                console.log('WebSocket connection closed:', event.code, event.reason);
                this.isConnected = false;
                this._updateStatusIndicator("offline");
                this._logToUI("Connection closed.", "info");
                this._resetUI(); // Reset UI on close
                // Optionally trigger automatic reconnection here
                // If the promise hasn't resolved/rejected yet (e.g., closed during initial connection)
                // reject(new Error(`WebSocket closed prematurely: Code ${event.code}`));
            };
        });
    }

    _sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // console.debug("Sending WebSocket Message:", message.type); // Keep logging minimal
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('WebSocket is not open. Cannot send message.');
            this._logToUI("Connection lost.", "error");
            this._resetUI();
        }
    }

    sendAudioChunk(base64Audio) {
        if (!this.isMuted) { // Only send if not muted
             this._sendMessage({ type: 'audio', data: base64Audio });
        }
    }

    sendTextMessage(text) {
        this._logToUI(text, "user"); // Log user text to chat
        this._sendMessage({ type: 'text', data: text });
        this.endUserTurn(); // End turn immediately after sending text
    }

    endUserTurn() {
        console.log("Signaling end of user turn.");
        this._sendMessage({ type: 'end_of_turn' });
    }

    // *** CHANGE 4: Make _startRecording async ***
    async _startRecording() { // Make async
        if (this.isRecording) return;
        try {
            // Ensure audio context is ready before starting recorder
            // Although initializeAudio is called on connect, call it again here
            // to be absolutely sure and handle cases where it might have been suspended.
            const audioReady = await this.initializeAudio();
            if (!audioReady) {
                 throw new Error("Audio context could not be initialized/resumed.");
            }

            await this.audioRecorder.start();
            this.isRecording = true;
            this.isMuted = false; // Start unmuted
            console.log("Audio recording started.");
            this._logToUI("🎤 Recording... (Click Mic again to stop)", "info");
            this.micButton.classList.add('active');
            this.micButton.title = "Stop Recording";
            this.micButton.querySelector('.material-symbols-outlined').textContent = 'stop_circle'; // Change icon

            // Send audio chunks
            this.audioRecorder.on('data', (base64Data) => {
                if (this.isConnected && !this.isMuted) { // Check connection and mute state
                    this.sendAudioChunk(base64Data);
                }
            });
        } catch (error) {
            console.error('Error starting recording:', error);
            this._logToUI(`Error starting microphone: ${error.message}`, "error");
            // Reset button state if recording failed to start
            this.isRecording = false;
            this.micButton.classList.remove('active');
            this.micButton.title = "Start Recording";
            this.micButton.querySelector('.material-symbols-outlined').textContent = 'mic';
        }
    }

    _stopRecording() {
        if (!this.isRecording) return;
        this.audioRecorder.stop();
        this.isRecording = false;
        this.isMuted = false; // Reset mute state
        console.log("Audio recording stopped.");
        this._logToUI("Recording stopped.", "info");
        this.micButton.classList.remove('active');
        this.micButton.classList.remove('muted'); // Ensure mute style is off
        this.micButton.title = "Start Recording";
        this.micButton.querySelector('.material-symbols-outlined').textContent = 'mic'; // Reset icon
        this.endUserTurn(); // Signal end of input turn
        if (this.audioStreamer) this.audioStreamer.stop(); // Stop playback if any was happening
    }

    // No toggleMute needed for simplified UI, just start/stop

    disconnect() {
        if (this.ws) {
            console.log("Disconnecting WebSocket.");
            this._stopRecording(); // Ensure recording stops if active
            if (this.audioStreamer) this.audioStreamer.stop();
            this.ws.close(); // Triggers onclose handler
        }
        this._resetUI(); // Update UI immediately
    }

    _resetUI() {
        this.isConnected = false;
        this.isRecording = false;
        this.isMuted = false;
        this.connectContainer.classList.remove('hidden');
        this.mediaContainer.classList.add('hidden');
        this.micButton.disabled = true;
        this.micButton.classList.remove('active', 'muted');
        this.micButton.querySelector('.material-symbols-outlined').textContent = 'mic';
        this.micButton.title = "Start/Stop Recording";
         // Don't clear chat on disconnect/error, maybe add separator?
        // this.chatOutput.innerHTML = '<p class="info-message">Disconnected. Click Connect to restart.</p>';
    }

    // *** CHANGE 5: Make button handler async ***
    _bindUIEvents() {
        this.connectButton.onclick = async () => { // Make async
            this.connectButton.disabled = true;
            this._logToUI("Connecting...", "info");
            try {
                await this.connect(); // Await the connect promise
                 // Success is handled within connect's onopen
            } catch (error) {
                // Error handling moved inside connect's onerror/onclose/reject
                 console.error("Connect button click failed:", error);
                 // LogUI and status indicator are handled by connect error paths
                this.connectButton.disabled = false; // Re-enable on failure
                 this._updateStatusIndicator("error"); // Ensure error state shown
                 // Optional: Log a more specific message if connect rejected
                 this._logToUI(`Connection Failed: ${error.message || 'Unknown error'}`, "error");
            }
        };

        this.stopButton.onclick = () => {
            this.disconnect();
        };

        this.micButton.onclick = async () => { // Make async because _startRecording is async
            if (!this.isConnected) return;
            if (this.isRecording) {
                this._stopRecording();
            } else {
                await this._startRecording(); // Await start recording
            }
        };
    }
}

// Initialize the API and make it globally accessible (or use modules)
window.adkApi = new ADKWebSocketAPI();
console.log("adkApi created."); // Log instance creation