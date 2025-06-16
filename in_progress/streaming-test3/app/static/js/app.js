// app.js

const socket = new WebSocket("ws://" + location.host + "/ws");

const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");
const responseDiv = document.getElementById("response");

const recorder = new AudioRecorder();
const videoCanvas = document.getElementById('videoCanvas');
const videoElement = document.getElementById('video');
const context = videoCanvas.getContext('2d');

let cameraInterval;

// --- WebSocket Event Handlers ---

socket.onopen = () => {
    console.log("WebSocket connection opened.");
    startButton.disabled = false;
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Use the agent name (now with underscores) to create a valid ID
    const agentId = data.agent.replace(/_/g, '-');
    let agentDiv = document.getElementById(`agent-${agentId}`);
    
    if (!agentDiv) {
        agentDiv = document.createElement("div");
        agentDiv.id = `agent-${agentId}`;
        agentDiv.className = 'agent-response';
        
        const title = document.createElement('h3');
        // Replace underscores with spaces for a nicer display name
        title.textContent = `${data.agent.replace(/_/g, ' ')} Output:`;
        agentDiv.appendChild(title);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'agent-content';
        agentDiv.appendChild(contentDiv);
        
        responseDiv.appendChild(agentDiv);
    }

    const contentDiv = agentDiv.querySelector('.agent-content');
    contentDiv.innerHTML += `<span>${data.response}</span>`;
    
    responseDiv.scrollTop = responseDiv.scrollHeight;
};

socket.onclose = () => {
    console.log("WebSocket connection closed.");
    startButton.disabled = true;
    stopButton.disabled = true;
};

// --- Camera and Video Logic ---

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoElement.srcObject = stream;

    cameraInterval = setInterval(() => {
        context.drawImage(videoElement, 0, 0, videoCanvas.width, videoCanvas.height);
        const data = videoCanvas.toDataURL('image/jpeg');
        
        // **BUG FIX:** Use the new, valid agent name.
        socket.send(JSON.stringify({
            agent: "Camera_Agent",
            payload: data
        }));
    }, 500); 
}


// --- Button Event Handlers ---

startButton.onclick = async () => {
    await recorder.start(
        (audioData) => {
            // **BUG FIX:** Use the new, valid agent name.
            socket.send(JSON.stringify({
                agent: "Audio_Agent",
                payload: audioData
            }));
        }
    );
    await setupCamera();
    
    startButton.disabled = true;
    stopButton.disabled = false;
    console.log("Recording and camera started.");
};

stopButton.onclick = () => {
    recorder.stop();
    clearInterval(cameraInterval);
    
    startButton.disabled = false;
    stopButton.disabled = true;
    console.log("Recording and camera stopped.");
};