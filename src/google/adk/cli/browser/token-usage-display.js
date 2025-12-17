/**
 * Token Usage and Cost Display
 *
 * This script monitors SSE events for token usage metadata and displays
 * the accumulated token counts and estimated costs in USD.
 * UI is integrated into the chat input area, matching the website's theme.
 */

(function() {
  'use strict';

  // State management
  let sessionTokenUsage = {
    totalPromptTokens: 0,
    totalOutputTokens: 0,
    totalCachedTokens: 0,
    totalCost: 0,
    totalTokens: 0,
    eventCount: 0
  };

  let isPopoverOpen = false;
  let buttonElement = null;
  let popoverElement = null;

  // Find the message input textarea
  function findMessageInput() {
    const selectors = [
      'textarea[placeholder*="message" i]',
      'textarea[placeholder*="Message" i]',
      'textarea[aria-label*="message" i]',
      'textarea',
      'input[type="text"]',
    ];

    for (const selector of selectors) {
      const el = document.querySelector(selector);
      if (el) {
        console.log('[Token Usage] Found input:', selector);
        return el;
      }
    }
    return null;
  }

  // Create button container next to input
  function createButtonContainer() {
    const input = findMessageInput();
    if (!input) {
      console.warn('[Token Usage] Could not find message input');
      return null;
    }

    // Find the parent container that holds the input and buttons
    let container = input.parentElement;

    // Look for a container that has multiple children (input + buttons)
    while (container && container.children.length < 2 && container !== document.body) {
      container = container.parentElement;
    }

    if (!container || container === document.body) {
      console.warn('[Token Usage] Could not find suitable container');
      return null;
    }

    console.log('[Token Usage] Found container:', container);

    // Create a wrapper div for our button
    const buttonWrapper = document.createElement('div');
    buttonWrapper.id = 'token-usage-wrapper';
    buttonWrapper.style.cssText = `
      display: inline-flex;
      align-items: center;
      margin: 0 8px;
    `;

    // Try to append to container
    container.appendChild(buttonWrapper);

    return buttonWrapper;
  }

  // Create the main button that shows cost and token count
  function createUsageButton() {
    const button = document.createElement('button');
    button.id = 'token-usage-button';
    button.type = 'button';
    button.setAttribute('aria-label', 'Token usage and cost');

    // Match the website's button styling
    button.style.cssText = `
      background: transparent;
      border: 1px solid rgba(128, 128, 128, 0.3);
      border-radius: 20px;
      padding: 6px 12px;
      font-family: inherit;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      color: inherit;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
      white-space: nowrap;
    `;

    button.innerHTML = `
      <span id="cost-display" style="font-weight: 600;">$0.00</span>
      <span style="opacity: 0.5;">|</span>
      <span id="token-count-display" style="opacity: 0.8;">0 tokens</span>
    `;

    button.addEventListener('mouseenter', () => {
      button.style.backgroundColor = 'rgba(128, 128, 128, 0.1)';
      button.style.borderColor = 'rgba(128, 128, 128, 0.5)';
    });

    button.addEventListener('mouseleave', () => {
      button.style.backgroundColor = 'transparent';
      button.style.borderColor = 'rgba(128, 128, 128, 0.3)';
    });

    button.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      togglePopover(button);
    });

    buttonElement = button;
    return button;
  }

  // Create the popover that shows detailed breakdown
  function createPopover() {
    const popover = document.createElement('div');
    popover.id = 'token-usage-popover';
    popover.style.cssText = `
      position: fixed;
      background: var(--surface-container, #2d2d2d);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 12px;
      padding: 16px;
      font-family: inherit;
      font-size: 13px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
      z-index: 10001;
      min-width: 200px;
      display: none;
      color: inherit;
    `;

    popover.innerHTML = `
      <div style="margin-bottom: 12px; padding-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.12);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="font-weight: 500; font-size: 14px;">Token Usage</span>
          <button id="reset-usage-btn" style="
            background: transparent;
            border: none;
            color: var(--primary, #8ab4f8);
            font-size: 12px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
          ">Reset</button>
        </div>
      </div>

      <div style="display: flex; flex-direction: column; gap: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="opacity: 0.7;">Input</span>
          <span id="popover-input-tokens" style="font-weight: 500;">–</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="opacity: 0.7;">Output</span>
          <span id="popover-output-tokens" style="font-weight: 500;">–</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center;">
          <span style="opacity: 0.7;">Cost</span>
          <span id="popover-cost" style="font-weight: 600;">–</span>
        </div>
      </div>

      <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.12);">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 12px; opacity: 0.6;">
          <span>Total events</span>
          <span id="popover-event-count">0</span>
        </div>
      </div>
    `;

    document.body.appendChild(popover);

    // Add reset button handler
    const resetBtn = document.getElementById('reset-usage-btn');
    if (resetBtn) {
      resetBtn.addEventListener('mouseenter', () => {
        resetBtn.style.backgroundColor = 'rgba(255, 255, 255, 0.08)';
      });
      resetBtn.addEventListener('mouseleave', () => {
        resetBtn.style.backgroundColor = 'transparent';
      });
      resetBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUsage();
      });
    }

    // Close popover when clicking outside
    document.addEventListener('click', (e) => {
      const popoverEl = document.getElementById('token-usage-popover');
      const buttonEl = document.getElementById('token-usage-button');
      if (isPopoverOpen &&
          popoverEl &&
          !popoverEl.contains(e.target) &&
          buttonEl &&
          !buttonEl.contains(e.target)) {
        closePopover();
      }
    });

    popoverElement = popover;
    return popover;
  }

  // Position popover relative to button
  function positionPopover(button) {
    const popover = document.getElementById('token-usage-popover');
    if (!popover || !button) return;

    const buttonRect = button.getBoundingClientRect();

    // Position above the button
    popover.style.bottom = `${window.innerHeight - buttonRect.top + 8}px`;
    popover.style.right = `${window.innerWidth - buttonRect.right}px`;
    popover.style.left = 'auto';
    popover.style.top = 'auto';
  }

  // Toggle popover visibility
  function togglePopover(button) {
    const popover = document.getElementById('token-usage-popover');
    if (!popover) return;

    isPopoverOpen = !isPopoverOpen;

    if (isPopoverOpen) {
      positionPopover(button);
      popover.style.display = 'block';
    } else {
      popover.style.display = 'none';
    }
  }

  // Close popover
  function closePopover() {
    const popover = document.getElementById('token-usage-popover');
    if (popover) {
      popover.style.display = 'none';
      isPopoverOpen = false;
    }
  }

  // Update the button display
  function updateButton() {
    const costDisplay = document.getElementById('cost-display');
    const tokenCountDisplay = document.getElementById('token-count-display');

    if (costDisplay) {
      const costFormatted = sessionTokenUsage.totalCost >= 0.01
        ? `$${sessionTokenUsage.totalCost.toFixed(2)}`
        : `$${sessionTokenUsage.totalCost.toFixed(4)}`;
      costDisplay.textContent = costFormatted;
    }

    if (tokenCountDisplay) {
      const totalTokens = sessionTokenUsage.totalPromptTokens + sessionTokenUsage.totalOutputTokens;
      tokenCountDisplay.textContent = `${totalTokens.toLocaleString()} token${totalTokens !== 1 ? 's' : ''}`;
    }
  }

  // Update the popover display
  function updatePopover() {
    const inputTokensEl = document.getElementById('popover-input-tokens');
    const outputTokensEl = document.getElementById('popover-output-tokens');
    const costEl = document.getElementById('popover-cost');
    const eventCountEl = document.getElementById('popover-event-count');

    if (inputTokensEl) {
      inputTokensEl.textContent = sessionTokenUsage.totalPromptTokens > 0
        ? sessionTokenUsage.totalPromptTokens.toLocaleString()
        : '–';
    }

    if (outputTokensEl) {
      outputTokensEl.textContent = sessionTokenUsage.totalOutputTokens > 0
        ? sessionTokenUsage.totalOutputTokens.toLocaleString()
        : '–';
    }

    if (costEl) {
      const costFormatted = sessionTokenUsage.totalCost >= 0.01
        ? `$${sessionTokenUsage.totalCost.toFixed(2)}`
        : sessionTokenUsage.totalCost > 0
        ? `$${sessionTokenUsage.totalCost.toFixed(4)}`
        : '–';
      costEl.textContent = costFormatted;
    }

    if (eventCountEl) {
      eventCountEl.textContent = sessionTokenUsage.eventCount.toString();
    }
  }

  // Update all displays
  function updateDisplay() {
    updateButton();
    updatePopover();
  }

  // Reset usage statistics
  function resetUsage() {
    sessionTokenUsage = {
      totalPromptTokens: 0,
      totalOutputTokens: 0,
      totalCachedTokens: 0,
      totalCost: 0,
      totalTokens: 0,
      eventCount: 0
    };
    updateDisplay();
  }

  // Process an event from the SSE stream
  function processEvent(eventData) {
    try {
      const event = JSON.parse(eventData);

      // Check if the event has usage metadata
      if (event.usageMetadata) {
        const metadata = event.usageMetadata;

        // Update token counts
        if (metadata.promptTokenCount) {
          sessionTokenUsage.totalPromptTokens += metadata.promptTokenCount;
        }
        if (metadata.candidatesTokenCount) {
          sessionTokenUsage.totalOutputTokens += metadata.candidatesTokenCount;
        }
        if (metadata.cachedContentTokenCount) {
          sessionTokenUsage.totalCachedTokens += metadata.cachedContentTokenCount;
        }

        // Update cost if available
        if (event.costUsd !== undefined && event.costUsd !== null) {
          sessionTokenUsage.totalCost += event.costUsd;
          sessionTokenUsage.eventCount++;
        }

        // Update the display
        updateDisplay();
      }
    } catch (e) {
      console.error('Error processing event for token usage:', e);
    }
  }

  // Intercept fetch requests to monitor SSE events
  const originalFetch = window.fetch;
  window.fetch = function(...args) {
    const request = args[0];

    // Check if this is a run_sse request
    if (typeof request === 'string' && request.includes('/run_sse')) {
      return originalFetch.apply(this, args).then(response => {
        // Clone the response so we can read it
        const clonedResponse = response.clone();

        // Process the SSE stream
        const reader = clonedResponse.body.getReader();
        const decoder = new TextDecoder();

        function readStream() {
          reader.read().then(({ done, value }) => {
            if (done) return;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.substring(6);
                if (data && data !== '[DONE]') {
                  processEvent(data);
                }
              }
            }

            readStream();
          });
        }

        readStream();

        return response;
      });
    }

    return originalFetch.apply(this, args);
  };

  // Try to inject the button
  function tryInject(retries = 15) {
    console.log(`[Token Usage] Injection attempt ${16 - retries}/15`);

    const wrapper = createButtonContainer();

    if (wrapper) {
      const button = createUsageButton();
      wrapper.appendChild(button);
      createPopover();
      console.log('[Token Usage] ✓ Button injected successfully');
      return true;
    } else if (retries > 0) {
      setTimeout(() => tryInject(retries - 1), 1000);
      return false;
    } else {
      console.warn('[Token Usage] ✗ Could not find suitable location after 15 attempts');

      // Fallback: Create floating button
      console.log('[Token Usage] Creating fallback floating button');
      const button = createUsageButton();
      button.style.position = 'fixed';
      button.style.bottom = '20px';
      button.style.right = '20px';
      button.style.zIndex = '10000';
      document.body.appendChild(button);
      createPopover();
      console.log('[Token Usage] ✓ Fallback button created');
      return true;
    }
  }

  // Initialize when the DOM is ready
  function initialize() {
    console.log('[Token Usage] Initializing...');

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        console.log('[Token Usage] DOM loaded, starting injection');
        tryInject();
      });
    } else {
      console.log('[Token Usage] DOM already loaded, starting injection');
      tryInject();
    }
  }

  initialize();
})();
