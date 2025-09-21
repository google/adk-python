# A2A Root Sample Agent

This sample demonstrates how to use a **remote Agent-to-Agent (A2A) agent as the root agent** in the Agent Development Kit (ADK). This is a simplified approach where the main agent is actually a remote A2A service, also showcasing how to run remote agents using uvicorn command.

## Overview

The A2A Root sample consists of:

- **Root Agent** (`agent.py`): A remote A2A agent proxy as root agent that talks to a remote a2a agent running on a separate server
- **Remote Hello World Agent** (`remote_a2a/hello_world/agent.py`): The actual agent implementation that handles dice rolling and prime number checking running on remote server

## Architecture

```
┌─────────────────┐    ┌────────────────────┐
│   Root Agent    │───▶│   Remote Hello     │
│ (RemoteA2aAgent)│    │   World Agent      │
│ (localhost:8000)│    │  (localhost:8001)  │
└─────────────────┘    └────────────────────┘
```

## Key Features

### 1. **Remote A2A as Root Agent**
- The `root_agent` is a `RemoteA2aAgent` that connects to a remote A2A service
- Demonstrates how to use remote agents as the primary agent instead of local agents
- Shows the flexibility of the A2A architecture for distributed agent deployment

### 2. **Uvicorn Server Deployment**
- The remote agent is served using uvicorn, a lightweight ASGI server
- Demonstrates a simple way to deploy A2A agents without using the ADK CLI
- Shows how to expose A2A agents as standalone web services

### 3. **Agent Functionality**
- **Dice Rolling**: Can roll dice with configurable number of sides
- **Prime Number Checking**: Can check if numbers are prime
- **State Management**: Maintains roll history in tool context
- **Parallel Tool Execution**: Can use multiple tools in parallel

### 4. **Simple Deployment Pattern**
- Uses the `to_a2a()` utility to convert a standard ADK agent to an A2A service
- Minimal configuration required for remote agent deployment

## Setup and Usage

### Prerequisites

1. **Start the Remote A2A Agent server**:
   ```bash
   # Start the remote agent using uvicorn
   uvicorn contributing.samples.a2a_root.remote_a2a.hello_world.agent:a2a_app --host localhost --port 8001
   ```

2. **Run the Main Agent**:
   ```bash
   # In a separate terminal, run the adk web server
   adk web contributing/samples/
   ```

### Example Interactions

Once both services are running, you can interact with the root agent:

**Simple Dice Rolling:**
```
User: Roll a 6-sided die
Bot: I rolled a 4 for you.
```

**Prime Number Checking:**
```
User: Is 7 a prime number?
Bot: Yes, 7 is a prime number.
```

**Combined Operations:**
```
User: Roll a 10-sided die and check if it's prime
Bot: I rolled an 8 for you.
Bot: 8 is not a prime number.
```

**Multiple Rolls with Prime Checking:**
```
User: Roll a die 3 times and check which results are prime
Bot: I rolled a 3 for you.
Bot: I rolled a 7 for you.
Bot: I rolled a 4 for you.
Bot: 3, 7 are prime numbers.
```

## Code Structure

### Root Agent (`agent.py`)

- **`root_agent`**: A `RemoteA2aAgent` that connects to the remote A2A service
- **Agent Card URL**: Points to the well-known agent card endpoint on the remote server

### Remote Hello World Agent (`remote_a2a/hello_world/agent.py`)

- **`roll_die(sides: int)`**: Function tool for rolling dice with state management
- **`check_prime(nums: list[int])`**: Async function for prime number checking
- **`root_agent`**: The main agent with comprehensive instructions
- **`a2a_app`**: The A2A application created using `to_a2a()` utility



## Troubleshooting

**Connection Issues:**
- Ensure the uvicorn server is running on port 8001
- Check that no firewall is blocking localhost connections
- Verify the agent card URL in the root agent configuration
- Check uvicorn logs for any startup errors

**Agent Not Responding:**
- Check the uvicorn server logs for errors
- Verify the agent instructions are clear and unambiguous
- Ensure the A2A app is properly configured with the correct port

**Uvicorn Issues:**
- Make sure the module path is correct: `contributing.samples.a2a_root.remote_a2a.hello_world.agent:a2a_app`
- Check that all dependencies are installed
