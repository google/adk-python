# Memory Sample

This sample demonstrates memory usage with ADK. The included `main.py` uses
in-memory services for simplicity. For persistent local memory, use SQLite.

## Persistent Local Memory (SQLite)

Programmatic usage:

```python
from google.adk.memory.sqlite_memory_service import SqliteMemoryService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService

memory_service = SqliteMemoryService(db_path="memory.db")
runner = Runner(
    app_name="my_app",
    agent=agent.root_agent,
    session_service=InMemorySessionService(),
    memory_service=memory_service,
)
```

CLI usage (supported for `adk web` and `adk api_server`):

```bash
adk web path/to/agents_dir --memory_service_uri=sqlite:///memory.db
```

Notes:
- `sqlite:///memory.db` uses a relative path.
- `sqlite:////abs/path/memory.db` uses an absolute path.
- `adk run` currently ignores `--memory_service_uri`.
