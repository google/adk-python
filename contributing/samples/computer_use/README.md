# Computer Use Agent

This directory contains a computer use agent that can operate a browser to complete user tasks. The agent uses Playwright to control a Chromium browser and can interact with web pages by taking screenshots, clicking, typing, and navigating.

This agent is to demo the usage of ComputerUseToolset.


## Overview

The computer use agent consists of:
- `agent.py`: Main agent configuration using Google's gemini-2.5-computer-use-preview-10-2025 model
- `playwright.py`: Playwright-based computer implementation for browser automation
- `requirements.txt`: Python dependencies

## Setup

### 1. Install Python Dependencies

Install the required Python packages from the requirements file:

```bash
uv pip install -r contributing/samples/computer_use/requirements.txt
```

### 2. Install Playwright Dependencies

Install Playwright's system dependencies for Chromium:

```bash
playwright install-deps chromium
```

### 3. Install Chromium Browser

Install the Chromium browser for Playwright:

```bash
playwright install chromium
```

## Usage

### Running the Agent

To start the computer use agent, run the following command from the project root:

```bash
adk web contributing/samples
```

This will start the ADK web interface where you can interact with the computer_use agent.

### Example Queries

Once the agent is running, you can send queries like:

```
find me a flight from SF to Hawaii on next Monday, coming back on next Friday. start by navigating directly to flights.google.com
```

The agent will:
1. Open a browser window
2. Navigate to the specified website
3. Interact with the page elements to complete your task
4. Provide updates on its progress

### Other Example Tasks

- Book hotel reservations
- Search for products online
- Fill out forms
- Navigate complex websites
- Research information across multiple pages

## Technical Details

- **Model**: Uses Google's `gemini-2.5-computer-use-preview-10-2025` model for computer use capabilities
- **Browser**: Automated Chromium browser via Playwright
- **Screen Size**: Configured for 600x800 resolution
- **Tools**: Uses ComputerUseToolset for screen capture, clicking, typing, and scrolling

## Troubleshooting

If you encounter issues:

1. **Playwright not found**: Make sure you've run both `playwright install-deps chromium` and `playwright install chromium`
2. **Dependencies missing**: Verify all packages from `requirements.txt` are installed
3. **Browser crashes**: Check that your system supports Chromium and has sufficient resources
4. **Permission errors**: Ensure your user has permission to run browser automation tools

## Notes

- The agent operates in a controlled browser environment
- Screenshots are taken to help the agent understand the current state
- The agent will provide updates on its actions as it works
- Be patient as complex tasks may take some time to complete
