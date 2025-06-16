# Stateful Multi-Agent Systems in ADK

This example demonstrates how to create a stateful multi-agent system in ADK, combining the power of persistent state management with specialized agent delegation. This approach creates intelligent agent systems that remember user information across interactions while leveraging specialized domain expertise.

## What is a Stateful Multi-Agent System?

A Stateful Multi-Agent System combines two powerful patterns:

1. **State Management**: Persisting information about users and conversations across interactions
2. **Multi-Agent Architecture**: Distributing tasks among specialized agents based on their expertise

The result is a sophisticated agent ecosystem that can:
- Remember user information and interaction history
- Route queries to the most appropriate specialized agent
- Provide personalized responses based on past interactions
- Maintain context across multiple agent delegates

This example implements a customer service system for an online course platform, where specialized agents handle different aspects of customer support while sharing a common state.

## Project Structure

```
7-stateful-multi-agent/
│
├── customer_service_agent/         # Main agent package
│   ├── __init__.py                 # Required for ADK discovery
│   ├── agent.py                    # Root agent definition
│   └── sub_agents/                 # Specialized agents
│       ├── course_support_agent/   # Handles course content questions
│       ├── order_agent/            # Manages order history and refunds
│       ├── policy_agent/           # Answers policy questions
│       └── sales_agent/            # Handles course purchases
│
├── main.py                         # Application entry point with session setup
├── utils.py                        # Helper functions for state management
├── .env                            # Environment variables
└── README.md                       # This documentation
```

## Key Components

### 1. Session Management

The example uses `InMemorySessionService` to store session state:

```python
session_service = InMemorySessionService()

def initialize_state():
    """Initialize the session state with default values."""
    return {
        "user_name": "Brandon Hancock",
        "purchased_courses": [""],
        "interaction_history": [],
    }

# Create a new session with initial state
session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID,
    state=initialize_state(),
)
```

### 2. State Sharing Across Agents

All agents in the system can access the same session state, enabling:
- Root agent to track interaction history
- Sales agent to update purchased courses
- Course support agent to check if user has purchased specific courses
- All agents to personalize responses based on user information

### 3. Multi-Agent Delegation

The customer service agent routes queries to specialized sub-agents:

```python
customer_service_agent = Agent(
    name="customer_service",
    model="gemini-2.0-flash",
    description="Customer service agent for AI Developer Accelerator community",
    instruction="""
    You are the primary customer service agent for the AI Developer Accelerator community.
    Your role is to help users with their questions and direct them to the appropriate specialized agent.
    
    # ... detailed instructions ...
    
    """,
    sub_agents=[policy_agent, sales_agent, course_support_agent, order_agent],
    tools=[get_current_time],
)
```

## How It Works

1. **Initial Session Creation**:
   - A new session is created with user information and empty interaction history
   - Session state is initialized with default values

2. **Conversation Tracking**:
   - Each user message is added to `interaction_history` in the state
   - Agents can review past interactions to maintain context

3. **Query Routing**:
   - The root agent analyzes the user query and decides which specialist should handle it
   - Specialized agents receive the full state context when delegated to

4. **State Updates**:
   - When a user purchases a course, the sales agent updates `purchased_courses`
   - These updates are available to all agents for future interactions

5. **Personalized Responses**:
   - Agents tailor responses based on purchase history and previous interactions
   - Different paths are taken based on what the user has already purchased

## Getting Started


### Setup

1. Activate the virtual environment from the root directory:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Make sure your Google API key is set in the `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

### Running the Example

To run the stateful multi-agent example:

```bash
python main.py
```

This will:
1. Initialize a new session with default state
2. Start an interactive conversation with the customer service agent
3. Track all interactions in the session state
4. Allow specialized agents to handle specific queries

### Example Conversation Flow

Try this conversation flow to test the system:

1. **Start with a general query**:
   - "What courses do you offer?"
   - (Root agent will route to sales agent)

2. **Ask about purchasing**:
   - "I want to buy the AI Marketing Platform course"
   - (Sales agent will process the purchase and update state)

3. **Ask about course content**:
   - "Can you tell me about the content in the AI Marketing Platform course?"
   - (Root agent will route to course support agent, which now has access)

4. **Ask about refunds**:
   - "What's your refund policy?"
   - (Root agent will route to policy agent)

Notice how the system remembers your purchase across different specialized agents!

## Advanced Features

### 1. Interaction History Tracking

The system maintains a history of interactions to provide context:

```python
# Update interaction history with the user's query
add_user_query_to_history(
    session_service, APP_NAME, USER_ID, SESSION_ID, user_input
)
```

### 2. Dynamic Access Control

The system implements conditional access to certain agents:

```
3. Course Support Agent
   - For questions about course content
   - Only available for courses the user has purchased
   - Check if "ai_marketing_platform" is in the purchased courses before directing here
```

### 3. State-Based Personalization

All agents tailor responses based on session state:

```
Tailor your responses based on the user's purchase history and previous interactions.
When the user hasn't purchased any courses yet, encourage them to explore the AI Marketing Platform.
When the user has purchased courses, offer support for those specific courses.
```

## Production Considerations

For a production implementation, consider:

1. **Persistent Storage**: Replace `InMemorySessionService` with `DatabaseSessionService` to persist state across application restarts
2. **User Authentication**: Implement proper user authentication to securely identify users
3. **Error Handling**: Add robust error handling for agent failures and state corruption
4. **Monitoring**: Implement logging and monitoring to track system performance

## Additional Resources

- [ADK Sessions Documentation](https://google.github.io/adk-docs/sessions/session/)
- [ADK Multi-Agent Systems Documentation](https://google.github.io/adk-docs/agents/multi-agent-systems/)
- [State Management in ADK](https://google.github.io/adk-docs/sessions/state/)
