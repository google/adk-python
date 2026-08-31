# get_fast_api_app

`get_fast_api_app` builds a `FastAPI` application that serves every agent in a
directory over ADK's HTTP API, and hands it back to you so you can add your own
routes, middleware and lifespan around it. Reach for it at the point where
`adk api_server` has stopped being enough. It is the same function that command
calls, exposed so that you can have the application object instead of the
command.

## Introduction

`adk api_server` is a complete server, and that is exactly its limit. You cannot
add a route to it, wrap it in your own authentication middleware, mount it under
an existing application, or run it under a process manager that wants an ASGI
callable. As soon as you self-host, whether that is Cloud Run, GKE, or a
container behind your own gateway, you need the application object rather than
the command.

That is what this function returns. Most of its arguments are wiring decisions
rather than server settings, because the function assembles a whole runtime
before it hands you the app. It resolves the four services an agent run needs,
which are session, artifact, memory and credential, from URI strings. It picks
an agent loader, imports any custom service registrations sitting in your agents
directory, chooses between the production and development server
implementations, and optionally attaches A2A routes for agents that publish an
agent card.

## Get started

Two keyword arguments are required, and every argument is keyword-only.

```python
from google.adk.cli.fast_api import get_fast_api_app

app = get_fast_api_app(agents_dir="./agents", web=False)


@app.get("/build-info")
async def build_info() -> dict[str, str]:
  return {"commit": "abc123", "environment": "staging"}
```

Serve it like any other ASGI application:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

The parameter is `agents_dir`, plural. `agent_dir` raises
`TypeError: get_fast_api_app() got an unexpected keyword argument 'agent_dir'.
Did you mean 'agents_dir'?`

`agents_dir` points at a directory of agents, one importable package per
subdirectory:

```
agents/
  home_automation/
    __init__.py
    agent.py        # defines root_agent
  support_bot/
    __init__.py
    agent.py
```

If you point the function at a single agent's folder, rather than a folder that
holds multiple agents, it detects that automatically. It treats the parent
folder as the agents root and makes that agent the default app, so you can send
requests without naming the app.

You do not need to add `/health` or `/version`; both already exist on the
returned app, returning `{"status": "ok"}` and the ADK version respectively.

## How it works

The call runs through a fixed order, and knowing it explains most of the
surprises.

1.  **Single-agent detection.** `agents_dir` is resolved and tested. If it is
    itself an agent directory, the effective agents root becomes its parent and
    its name becomes the server's default app name.
2.  **The agent loader.** With no `agent_loader` of your own, `web=True` gets a
    `NestedAgentLoader` and `web=False` gets an `AgentLoader`; both read agents
    from the directory.
3.  **`services.py` and `services.yaml` are imported** from the agents
    directory. The import happens *before* any service is constructed, which is
    what makes a custom URI scheme declared in those files usable in the
    arguments you passed. Registration is an import side effect and the registry
    is process-global.
4.  **The four services are built.** Session, artifact and memory each come from
    their URI argument, or from a local default when the argument is `None`. The
    credential service is always `InMemoryCredentialService`; there is no
    argument for it.
5.  **The server class is chosen.** `web=False` gives `ApiServer` and the
    production-safe routes only. `web=True` gives `DevServer`, which adds the
    Angular UI and the `/dev/...` endpoints for tracing, evaluation and the
    agent builder on top of those, roughly tripling the route count, and also
    switches on a denylist of YAML keys for config-defined agents. Some
    production packages ship without the development server at all; there,
    `web=True` logs a warning and falls back to `ApiServer`, so the UI and eval
    endpoints quietly disappear.
6.  **A2A routes are mounted last.** With `a2a=True`, every subdirectory of the
    agents root that contains an `agent.json` gets A2A routes mounted at
    `/a2a/<agent_name>`. A failure setting up one agent is logged and skipped
    rather than raised, so the other agents still come up.

The production-safe set that `ApiServer` serves is four groups of routes:

*   The three ways to run an agent, at `/run`, `/run_sse` and `/run_live`.
*   The session and artifact CRUD endpoints.
*   `/list-apps`.
*   The two status endpoints, `/health` and `/version`.

The JSON wire format is camelCase throughout.

## Configuration options

The function takes 28 keyword arguments. The five groups below are what a
self-hosted deployment actually sets; the ones missing from them exist for the
CLI's own plumbing.

### Where the agents come from

Set these when your agents are somewhere other than a plain directory of
packages, or when you want an edit to take effect without a restart.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `agents_dir` | `str` | required | Directory of agent packages, or a single agent directory. |
| `agent_loader` | `BaseAgentLoader \| None` | `None` | Load agents from somewhere other than a directory. |
| `reload_agents` | `bool` | `False` | Watch the agents directory and reload on change. |

### Where state is kept

Set these the moment sessions and artifacts have to outlive one process, which
is every deployment running more than one replica.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `session_service_uri` | `str \| None` | `None` | Backend for sessions. |
| `artifact_service_uri` | `str \| None` | `None` | Backend for artifacts. |
| `memory_service_uri` | `str \| None` | `None` | Backend for memory. |
| `task_store_uri` | `str \| None` | `None` | A2A task store. In-memory when unset. Only read when `a2a=True`. |
| `use_local_storage` | `bool` | `True` | Use on-disk defaults for sessions and artifacts when no URI is given. |
| `auto_create_session` | `bool` | `False` | Create a session on a request that names one that does not exist. |

### The network address and who may call

Set these once the server is reachable by anything other than you, since they
decide which `Host` headers and which browser origins are accepted.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bind_host` | `str \| None` | `None` | The address you will bind to. A loopback value turns on DNS-rebinding protection. |
| `host` | `str` | `"127.0.0.1"` | Advertised host. Binds nothing. |
| `port` | `int` | `8000` | Advertised port. Binds nothing. |
| `allow_origins` | `list[str] \| None` | `None` | CORS allow-list. |

### What the app exposes, and what you wrap around it

Set these to decide which routes exist at all, and to attach your own startup
work and plugins to the app the function returns.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `web` | `bool` | required | Serve the development UI and `/dev/...` endpoints as well as the API. |
| `a2a` | `bool` | `False` | Attach A2A routes for agents that ship an `agent.json`. |
| `url_prefix` | `str \| None` | `None` | Path prefix the app is served under, for the bundled UI's benefit. |
| `lifespan` | `Lifespan[FastAPI] \| None` | `None` | Your own startup and shutdown context manager. |
| `extra_plugins` | `list[str] \| None` | `None` | Fully qualified names of plugins to load into every runner. |

### Where traces go

Set one of these when you run on Google Cloud and want the server's spans in
Cloud Trace rather than dropped.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `trace_to_cloud` | `bool` | `False` | Export traces to Cloud Trace. |
| `otel_to_cloud` | `bool` | `False` | Export OpenTelemetry data to Google Cloud. |

### The service URIs

Each URI is dispatched by scheme through the service registry. The built-in
schemes are:

*   **Sessions:** `memory://`, `sqlite://`, `postgresql://`, `mysql://`,
    `agentengine://`.
*   **Artifacts:** `memory://`, `gs://` for Cloud Storage, `file://`.
*   **Memory:** `memory://`, `rag://` for a Vertex RAG corpus, `agentengine://`.
*   **A2A task stores**, for `task_store_uri`: `memory://`,
    `postgresql+asyncpg://`, `mysql+aiomysql://`, `sqlite+aiosqlite://`.

Sessions have one extra behavior: a scheme nobody registered falls through to
`DatabaseSessionService` with the URI used as a SQLAlchemy URL, which is how
AlloyDB and Cloud Spanner work without any explicit registration. Artifacts and
memory have no such fallback, so an unrecognized scheme raises there. Be aware
that the exception is a `click.ClickException`, not a `ValueError`, even though
nothing about your call involved the command line.

Leave the URIs unset and `use_local_storage` decides. At its default of `True`,
sessions go to a per-agent SQLite file under `<agents_dir>/<agent>/.adk/` and
artifacts to local files; memory is always in-memory when no URI is given. Set
`use_local_storage=False` for in-memory sessions and artifacts instead. A
directory that does not exist, or is not writable, is not an error: the function
logs a warning and falls back to in-memory services, which means a typo in
`agents_dir` produces a server that starts cleanly and loses everything on
restart.

### `bind_host`, and why `host` is not it

`host` and `port` bind nothing. They are advertised values, printed in the CLI
banner; the actual binding is done by whatever serves the app, which is uvicorn
in every example here.

`bind_host` is the security-relevant one. Tell the function the address you are
going to bind to, and if that address is loopback, the app switches on
DNS-rebinding protection: any request whose `Host` header names something other
than a loopback address, or a host vouched for by `allow_origins`, is rejected
with `403 Forbidden: host not allowed`. The check defends a local development
server against a web page that resolves its own domain to `127.0.0.1` and then
talks to your agent from the browser.

```python
app = get_fast_api_app(
    agents_dir="./agents", web=True, bind_host="127.0.0.1", port=8000
)
```

Three things switch the guard off again.

*   Leaving `bind_host` as `None` disables it entirely, which is the right
    default for an app served behind a reverse proxy, since the proxy's hostname
    would otherwise be rejected.
*   Passing `allow_origins=["*"]` disables it too, because a literal `*` is read
    as opting out.
*   A non-loopback `bind_host` such as `0.0.0.0` leaves it off, because a server
    reachable from the network cannot use "you must have reached me over
    loopback" as a signal.

### Compose with your own application

`lifespan` is your hook for anything that has to open at startup and close at
shutdown, such as a database pool or a background task. Pass an async context
manager and it is used as the app's lifespan. When `a2a=True` and the task store
owns a database engine, ADK wraps your lifespan in its own so that the engine is
disposed after yours exits.

`extra_plugins` takes fully qualified names, not plugin objects, and those
plugins are loaded into every runner the server creates.

`url_prefix` does not re-prefix the routes. It tells the bundled web UI where
the backend lives when the whole app is served under a path prefix, and adjusts
the `/dev-ui/` redirect. To actually serve the API under a prefix, mount the
returned app on a parent application.

## Advanced applications

Two of the decisions the function makes for you can be replaced: where agents
are read from, and which class a service URI resolves to.

### Load agents from somewhere other than a directory

The default loaders read agents from the filesystem. When your agents live in a
database, a package, or a config service, implement `BaseAgentLoader` and pass
it as `agent_loader`. It is a two-method abstract base class:

```python
from google.adk.agents import LlmAgent
from google.adk.apps.app import App
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader


class RegistryAgentLoader(BaseAgentLoader):

  def __init__(self, agents: dict[str, LlmAgent]):
    self._agents = agents

  def load_agent(self, agent_name: str) -> LlmAgent | App:
    return self._agents[agent_name]

  def list_agents(self) -> list[str]:
    return sorted(self._agents)


app = get_fast_api_app(
    agents_dir="./agents",
    web=False,
    agent_loader=RegistryAgentLoader(agents_by_name),
)
```

`agents_by_name` there is your own mapping from app name to root agent.

`list_agents` is expected to return names in alphabetical order, since it backs
`/list-apps`. There is a third, optional method, `list_agents_detailed`, whose
default implementation returns each name with empty display metadata; override
it to give the UI descriptions.

`agents_dir` is still required even with a custom loader, because it is where
service registrations and local storage are looked for. Point it at a real
directory you control.

### Register your own service backend

A custom session, artifact, memory or A2A task store becomes usable through
`session_service_uri` and friends once you register it against a URI scheme.
The registration goes in a `services.py` or a `services.yaml` inside the agents
directory, which `get_fast_api_app` imports for you before it builds anything.
Use the YAML form when the class can be built as `MyService(uri=..., **kwargs)`,
declaring its kind under a `type` key of `session`, `artifact`, `memory` or
`task_store`. Use Python for anything needing real construction logic. When
both files are present, both load, YAML first, and `services.py` wins on a
scheme collision. The
[`services.py`](../../../../contributing/samples/services.py) and
[`services.yaml`](../../../../contributing/samples/services.yaml) samples show
both styles, and the [ServiceRegistry guide](../service_registry/index.md)
covers the factory contract and the loading rules in full.

## Limitations

*   **`host` and `port` bind nothing**, and neither does the returned app. It is
    an ASGI application; something else has to serve it.
*   **The credential service cannot be replaced.** It is always
    `InMemoryCredentialService`, so tool credentials do not survive a restart
    and are not shared between processes.
*   **A bad `agents_dir` is silent.** A missing or read-only directory produces
    a working server on in-memory storage rather than an error.
*   **An unsupported artifact or memory URI raises `click.ClickException`,** a
    command-line exception type leaking into a library call.
*   **A2A setup failures are swallowed.** An agent whose `agent.json` is
    malformed is logged and skipped; the server starts without it.

## Related samples

*   [services.py](../../../../contributing/samples/services.py) registers a
    custom service backend against a URI scheme in Python.
*   [services.yaml](../../../../contributing/samples/services.yaml) does the
    same thing declaratively, for services that need no construction logic.
*   [dummy_services.py](../../../../contributing/samples/dummy_services.py)
    holds the throwaway service implementations those two register.
