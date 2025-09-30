// Security BigQuery Agent Web Interface
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const statusIndicator = document.getElementById('status');
    const examplesContainer = document.getElementById('examples');

    // Check backend status
    checkBackendStatus();
    setInterval(checkBackendStatus, 30000); // Check every 30 seconds

    // Load example queries
    loadExamples();

    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            sendMessage();
        }
    });

    function checkBackendStatus() {
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                if (data.adk_backend === 'healthy') {
                    statusIndicator.textContent = '● Connected';
                    statusIndicator.style.color = '#4ade80';
                } else {
                    statusIndicator.textContent = '● Backend Offline';
                    statusIndicator.style.color = '#ef4444';
                }
            })
            .catch(error => {
                statusIndicator.textContent = '● Connection Error';
                statusIndicator.style.color = '#ef4444';
            });
    }

    function loadExamples() {
        fetch('/examples')
            .then(response => response.json())
            .then(examples => {
                examples.forEach(example => {
                    const btn = document.createElement('button');
                    btn.className = 'example-btn';
                    btn.textContent = example;
                    btn.addEventListener('click', function() {
                        userInput.value = example;
                        userInput.focus();
                    });
                    examplesContainer.appendChild(btn);
                });
            });
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to chat
        addMessage(message, 'user');

        // Clear input
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        // Show loading
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message agent-message';
        loadingDiv.innerHTML = '<div class="loading"></div> Analyzing...';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Send to backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading
            chatMessages.removeChild(loadingDiv);

            if (data.error) {
                addMessage(`Error: ${data.error}`, 'error');
            } else {
                addMessage(data.response, 'agent');
            }
        })
        .catch(error => {
            // Remove loading
            chatMessages.removeChild(loadingDiv);
            addMessage(`Connection error: ${error.message}`, 'error');
        })
        .finally(() => {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        });
    }

    function addMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        if (type === 'agent') {
            // Format agent response with markdown-like processing
            message = formatAgentMessage(message);
        }

        messageDiv.innerHTML = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function formatAgentMessage(message) {
        // Convert markdown-style formatting
        message = message
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`(.*?)`/g, '<code>$1</code>')
            // Line breaks
            .replace(/\n/g, '<br>')
            // Bullet points
            .replace(/^- /gm, '• ')
            // Headers
            .replace(/^### (.*?)$/gm, '<h4>$1</h4>')
            .replace(/^## (.*?)$/gm, '<h3>$1</h3>')
            .replace(/^# (.*?)$/gm, '<h2>$1</h2>');

        return message;
    }

    // Add welcome message
    setTimeout(() => {
        addMessage(
            "👋 Hello! I'm your Security BigQuery Agent.\n\n" +
            "I can help you analyze your GCP security posture using BigQuery. " +
            "Try asking me about:\n" +
            "• Firewall rules and network security\n" +
            "• IAM accounts and permissions\n" +
            "• Compute instances and external IPs\n" +
            "• Storage bucket exposure\n" +
            "• Security findings and compliance\n\n" +
            "What would you like to know?",
            'agent'
        );
    }, 500);
});