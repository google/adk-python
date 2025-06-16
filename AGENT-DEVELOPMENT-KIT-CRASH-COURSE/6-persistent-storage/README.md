# Persistent Storage in ADK

This example demonstrates how to implement persistent storage for your ADK agents, allowing them to remember information and maintain conversation history across multiple sessions, application restarts, and even server deployments.

## What is Persistent Storage in ADK?

In previous examples, we used `InMemorySessionService` which stores session data only in memory - this data is lost when the application stops. For real-world applications, you'll often need your agents to remember user information and conversation history long-term. This is where persistent storage comes in.

ADK provides the `DatabaseSessionService` that allows you to store session data in a SQL database, ensuring:

1. **Long-term Memory**: Information persists across application restarts
2. **Consistent User Experiences**: Users can continue conversations where they left off
3. **Multi-user Support**: Different users' data remains separate and secure
4. **Scalability**: Works with production databases for high-scale deployments

This example shows how to implement a reminder agent that remembers your name and todos across different conversations using an SQLite database.

## Project Structure

```
5-persistent-storage/
│
├── memory_agent/               # Agent package
│   ├── __init__.py             # Required for ADK to discover the agent
│   └── agent.py                # Agent definition with reminder tools
│
├── main.py                     # Application entry point with database session setup
├── utils.py                    # Utility functions for terminal UI and agent interaction
├── .env                        # Environment variables
├── my_agent_data.db            # SQLite database file (created when first run)
└── README.md                   # This documentation
```

## Key Components

### 1. DatabaseSessionService

The core component that provides persistence is the `DatabaseSessionService`, which is initialized with a database URL:

```python
from google.adk.sessions import DatabaseSessionService

db_url = "sqlite:///./my_agent_data.db"
session_service = DatabaseSessionService(db_url=db_url)
```

This service allows ADK to:
- Store session data in a SQLite database file
- Retrieve previous sessions for a user
- Automatically manage database schemas

### 2. Session Management

The example demonstrates proper session management:

```python
# Check for existing sessions for this user
existing_sessions = session_service.list_sessions(
    app_name=APP_NAME,
    user_id=USER_ID,
)

# If there's an existing session, use it, otherwise create a new one
if existing_sessions and len(existing_sessions.sessions) > 0:
    # Use the most recent session
    SESSION_ID = existing_sessions.sessions[0].id
    print(f"Continuing existing session: {SESSION_ID}")
else:
    # Create a new session with initial state
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initialize_state(),
    )
```

### 3. State Management with Tools

The agent includes tools that update the persistent state:

```python
def add_reminder(reminder: str, tool_context: ToolContext) -> dict:
    # Get current reminders from state
    reminders = tool_context.state.get("reminders", [])
    
    # Add the new reminder
    reminders.append(reminder)
    
    # Update state with the new list of reminders
    tool_context.state["reminders"] = reminders
    
    return {
        "action": "add_reminder",
        "reminder": reminder,
        "message": f"Added reminder: {reminder}",
    }
```

Each change to `tool_context.state` is automatically saved to the database.

## Getting Started

### Prerequisites

- Python 3.9+
- Google API Key for Gemini models
- SQLite (included with Python)

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

To run the persistent storage example:

```bash
python main.py
```

This will:
1. Connect to the SQLite database (or create it if it doesn't exist)
2. Check for previous sessions for the user
3. Start a conversation with the memory agent
4. Save all interactions to the database

### Example Interactions

Try these interactions to test the agent's persistent memory:

1. **First run:**
   - "What's my name?"
   - "My name is John"
   - "Add a reminder to buy groceries"
   - "Add another reminder to finish the report"
   - "What are my reminders?"
   - Exit the program with "exit"

2. **Second run:**
   - "What's my name?"
   - "What reminders do I have?"
   - "Update my second reminder to submit the report by Friday"
   - "Delete the first reminder"
   
The agent will remember your name and reminders between runs!

## Using Database Storage in Production

While this example uses SQLite for simplicity, `DatabaseSessionService` supports various database backends through SQLAlchemy:

- PostgreSQL: `postgresql://user:password@localhost/dbname`
- MySQL: `mysql://user:password@localhost/dbname`
- MS SQL Server: `mssql://user:password@localhost/dbname`

For production use:
1. Choose a database system that meets your scalability needs
2. Configure connection pooling for efficiency
3. Implement proper security for database credentials
4. Consider database backups for critical agent data

## Additional Resources

- [ADK Sessions Documentation](https://google.github.io/adk-docs/sessions/session/)
- [Session Service Implementations](https://google.github.io/adk-docs/sessions/session/#sessionservice-implementations)
- [State Management in ADK](https://google.github.io/adk-docs/sessions/state/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/) for advanced database configuration 
