// Security BigQuery Agent Web Interface with streaming support
// Adds streaming chat, status monitoring, and an agent overview panel.
document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const statusIndicator = document.getElementById('status');
    const examplesContainer = document.getElementById('examples');
    const overviewToggle = document.getElementById('overview-toggle');
    const overviewPanel = document.getElementById('agent-overview');
    const overviewClose = document.getElementById('overview-close');
    const instructionContainer = document.getElementById('instruction-sections');
    const expandInstructionsBtn = document.getElementById('instructions-expand');
    const collapseInstructionsBtn = document.getElementById('instructions-collapse');
    const toolList = document.getElementById('tool-list');
    const toolFilterInput = document.getElementById('tool-filter');
    const modelLabel = document.getElementById('model');

    let instructionSections = [];
    let toolCatalogue = [];

    checkBackendStatus();
    setInterval(checkBackendStatus, 30000);
    loadExamples();
    loadAgentInfo();

    sendBtn?.addEventListener('click', () => sendMessage());
    userInput?.addEventListener('keypress', event => {
        if (event.key === 'Enter' && !event.shiftKey) {
            sendMessage();
        }
    });

    overviewToggle?.addEventListener('click', () => openOverview());
    overviewClose?.addEventListener('click', closeOverview);
    overviewPanel?.addEventListener('click', event => {
        if (event.target === overviewPanel) {
            closeOverview();
        }
    });

    document.addEventListener('keydown', event => {
        if (event.key === 'Escape' && overviewPanel && !overviewPanel.classList.contains('hidden')) {
            closeOverview();
        }
    });

    expandInstructionsBtn?.addEventListener('click', () => toggleInstructionDetails(true));
    collapseInstructionsBtn?.addEventListener('click', () => toggleInstructionDetails(false));
    toolFilterInput?.addEventListener('input', () => {
        renderToolList(toolCatalogue, toolFilterInput.value);
    });

    async function sendMessage() {
        if (!userInput) return;
        const message = userInput.value.trim();
        if (!message) return;

        addMessage(message, 'user');
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message agent-message';
        loadingDiv.innerHTML = '<div class="loading"></div> Analyzing...';
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            await streamAgentResponse(message, loadingDiv);
        } catch (error) {
            console.warn('Streaming failed, falling back to non-streaming request', error);
            if (loadingDiv.parentNode) {
                loadingDiv.parentNode.removeChild(loadingDiv);
            }
            await fallbackChatRequest(message);
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    async function streamAgentResponse(message, loadingDiv) {
        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        if (!response.ok || !response.body) {
            throw new Error(`Streaming not available (status ${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let messageDiv = null;
        let aggregatedText = '';

        const processSegment = (segment) => {
            const lines = segment.split('\n');
            let eventName = 'message';
            let dataPayload = '';

            lines.forEach(line => {
                if (line.startsWith('event:')) {
                    eventName = line.replace('event:', '').trim();
                } else if (line.startsWith('data:')) {
                    dataPayload += line.replace('data:', '').trim();
                }
            });

            if (!dataPayload) return;

            const payload = JSON.parse(dataPayload);

            if (eventName === 'start') {
                if (loadingDiv.parentNode) {
                    loadingDiv.parentNode.removeChild(loadingDiv);
                }
                messageDiv = createMessageElement('agent');
                chatMessages.appendChild(messageDiv);
            } else if (eventName === 'token') {
                aggregatedText += payload.text || '';
                if (!messageDiv) {
                    messageDiv = createMessageElement('agent');
                    chatMessages.appendChild(messageDiv);
                }
                messageDiv.innerHTML = formatAgentMessage(aggregatedText);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else if (eventName === 'end') {
                if (!messageDiv) {
                    messageDiv = createMessageElement('agent');
                    chatMessages.appendChild(messageDiv);
                }
                if (!aggregatedText) {
                    aggregatedText = payload.message || 'No response from agent.';
                }
                messageDiv.innerHTML = formatAgentMessage(aggregatedText);
            } else if (eventName === 'error') {
                throw new Error(payload.message || 'Streaming error');
            }
        };

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let boundary = buffer.indexOf('\n\n');

            while (boundary !== -1) {
                const segment = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);
                if (segment.trim()) {
                    processSegment(segment);
                }
                boundary = buffer.indexOf('\n\n');
            }
        }

        if (loadingDiv.parentNode) {
            loadingDiv.parentNode.removeChild(loadingDiv);
        }

        if (!messageDiv) {
            messageDiv = createMessageElement('agent');
            messageDiv.innerHTML = 'No response from agent.';
            chatMessages.appendChild(messageDiv);
        }
    }

    async function fallbackChatRequest(message) {
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message }),
            });
            const data = await response.json();
            if (data.error) {
                addMessage(`Error: ${data.error}`, 'error');
            } else {
                addMessage(data.response, 'agent');
            }
        } catch (error) {
            addMessage(`Connection error: ${error.message}`, 'error');
        }
    }

    function createMessageElement(type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        return messageDiv;
    }

    function addMessage(message, type) {
        const messageDiv = createMessageElement(type);
        if (type === 'agent') {
            message = formatAgentMessage(message);
        }
        messageDiv.innerHTML = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function formatAgentMessage(message) {
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>')
            .replace(/^- /gm, '• ')
            .replace(/^### (.*?)$/gm, '<h4>$1</h4>')
            .replace(/^## (.*?)$/gm, '<h3>$1</h3>')
            .replace(/^# (.*?)$/gm, '<h2>$1</h2>');
    }

    function openOverview() {
        if (!overviewPanel) return;
        overviewPanel.classList.remove('hidden');
        overviewPanel.setAttribute('aria-hidden', 'false');
        document.body.classList.add('modal-open');
        requestAnimationFrame(() => {
            overviewClose?.focus();
        });
    }

    function closeOverview() {
        if (!overviewPanel) return;
        overviewPanel.classList.add('hidden');
        overviewPanel.setAttribute('aria-hidden', 'true');
        document.body.classList.remove('modal-open');
        overviewToggle?.focus();
    }

    function toggleInstructionDetails(open) {
        if (!instructionContainer) return;
        const details = instructionContainer.querySelectorAll('details');
        details.forEach(detail => {
            detail.open = open;
        });
    }

    function renderInstructionSections(sections) {
        if (!instructionContainer) return;
        instructionContainer.innerHTML = '';

        if (!sections.length) {
            const emptyState = document.createElement('p');
            emptyState.className = 'section-body';
            emptyState.textContent = 'Instruction details are unavailable right now. Try again later.';
            instructionContainer.appendChild(emptyState);
            return;
        }

        sections.forEach((section, index) => {
            const wrapper = document.createElement('details');
            wrapper.open = index < 2;

            const summary = document.createElement('summary');
            summary.textContent = section.title;

            const body = document.createElement('div');
            body.className = 'section-body';
            body.innerHTML = formatAgentMessage(section.content || '');

            wrapper.appendChild(summary);
            wrapper.appendChild(body);
            instructionContainer.appendChild(wrapper);
        });
    }

    function renderToolList(tools, filterText = '') {
        if (!toolList) return;
        toolList.innerHTML = '';

        if (!Array.isArray(tools) || !tools.length) {
            const emptyItem = document.createElement('li');
            emptyItem.className = 'tool-card';
            emptyItem.textContent = 'No tools found.';
            toolList.appendChild(emptyItem);
            return;
        }

        const query = filterText.trim().toLowerCase();
        const filtered = tools.filter(tool => {
            const nameMatch = tool.name?.toLowerCase().includes(query);
            const descriptionMatch = tool.description?.toLowerCase().includes(query);
            const moduleMatch = tool.module?.toLowerCase().includes(query);
            return !query || nameMatch || descriptionMatch || moduleMatch;
        });

        if (!filtered.length) {
            const emptyItem = document.createElement('li');
            emptyItem.className = 'tool-card';
            emptyItem.textContent = 'No tools match your filter.';
            toolList.appendChild(emptyItem);
            return;
        }

        filtered.forEach(tool => {
            const item = document.createElement('li');
            item.className = 'tool-card';

            const title = document.createElement('h4');
            title.textContent = tool.name || 'Unnamed tool';

            const description = document.createElement('p');
            description.textContent = tool.description || 'No description available.';

            const module = document.createElement('p');
            const moduleLabel = document.createElement('span');
            moduleLabel.textContent = 'Module: ';
            const moduleCode = document.createElement('code');
            moduleCode.textContent = tool.module || 'unknown';
            module.appendChild(moduleLabel);
            module.appendChild(moduleCode);

            item.appendChild(title);
            item.appendChild(description);
            item.appendChild(module);

            toolList.appendChild(item);
        });
    }

    async function loadAgentInfo() {
        try {
            const response = await fetch('/agent-info');
            if (!response.ok) {
                throw new Error(`Failed to fetch agent info: ${response.status}`);
            }
            const data = await response.json();

            instructionSections = Array.isArray(data.instruction_sections) ? data.instruction_sections : [];
            toolCatalogue = Array.isArray(data.tools) ? data.tools : [];

            renderInstructionSections(instructionSections);
            renderToolList(toolCatalogue, toolFilterInput?.value || '');

            if (modelLabel && data.model) {
                modelLabel.textContent = `Model: ${data.model}`;
            }
        } catch (error) {
            console.error('Unable to load agent information', error);
            renderInstructionSections([]);
            renderToolList([]);
        }
    }

    function checkBackendStatus() {
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                const status = data.adk_backend || 'unreachable';
                if (status === 'healthy') {
                    statusIndicator.textContent = '● Connected';
                    statusIndicator.style.color = '#4ade80';
                } else if (status === 'unhealthy') {
                    statusIndicator.textContent = '● Backend Unhealthy';
                    statusIndicator.style.color = '#f59e0b';
                } else {
                    statusIndicator.textContent = '● Backend Offline';
                    statusIndicator.style.color = '#ef4444';
                }

                if (data.model && modelLabel) {
                    modelLabel.textContent = `Model: ${data.model}`;
                }
            })
            .catch(() => {
                statusIndicator.textContent = '● Connection Error';
                statusIndicator.style.color = '#ef4444';
            });
    }

    function loadExamples() {
        if (!examplesContainer) {
            return;
        }

        fetch('/examples')
            .then(response => response.json())
            .then(examples => {
                if (!Array.isArray(examples)) return;
                examples.forEach(example => {
                    const btn = document.createElement('button');
                    btn.className = 'example-btn';
                    btn.textContent = example;
                    btn.addEventListener('click', () => {
                        if (!userInput) return;
                        userInput.value = example;
                        userInput.focus();
                    });
                    examplesContainer.appendChild(btn);
                });
            })
            .catch(() => {
                // Silently ignore if examples endpoint is unavailable
            });
    }

    setTimeout(() => {
        addMessage(
            "👋 Hello! I'm your Security BigQuery Agent.<br><br>" +
            "I can help you analyze your GCP security posture using BigQuery. " +
            "Try asking me about:<br>" +
            "• Firewall rules and network security<br>" +
            "• IAM accounts and permissions<br>" +
            "• Compute instances and external IPs<br>" +
            "• Storage bucket exposure<br>" +
            "• Security findings and compliance<br><br>" +
            "What would you like to know?",
            'agent'
        );
    }, 500);
});
