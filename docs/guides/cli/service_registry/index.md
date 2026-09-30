# ServiceRegistry

`ServiceRegistry` maps a URI scheme to a factory function, so that
`--session_service_uri=mystore://...` on the command line, or
`session_service_uri="mystore://..."` in
[`get_fast_api_app`](../fast_api/index.md), builds your class instead of one of
ADK's. The same applies to an A2A task store.

## Introduction

Writing a `BaseSessionService` of your own is not the hard part, and using it
from a `Runner` you construct yourself takes one constructor argument. The
problem is everything that constructs the runner for you: `adk web`,
`adk api_server`, `get_fast_api_app`. Those take a URI string and resolve it for
you, and until your scheme is in the registry there is no string that names your
class.

The registry is the seam. It keeps four separate sets of schemes, one each for
sessions, artifacts, memory and A2A task stores, and in each of them a scheme
names a callable that takes `(uri, **kwargs)` and returns a service. ADK fills
them with its own built-ins at first use, and you add to them.

You never construct a `ServiceRegistry`. There is one process-wide instance
behind `get_service_registry()`.

## Get started

Put a `services.py` in your agents directory and register a factory in it:

```python
# my_agents/services.py
from google.adk.cli.service_registry import get_service_registry

from my_package.stores import DynamoSessionService


def dynamo_session_factory(uri: str, **kwargs) -> DynamoSessionService:
  """Builds a session service from dynamo://<table-name>."""
  table = uri.removeprefix("dynamo://")
  return DynamoSessionService(table_name=table)


get_service_registry().register_session_service("dynamo", dynamo_session_factory)
```

Then name the scheme wherever a service URI is accepted:

```bash
adk api_server my_agents --session_service_uri=dynamo://agent-sessions
```

The file has to be called exactly `services.py` and sit at the top of the
directory you pass as the agents directory. ADK inserts that directory on
`sys.path` and imports it before it builds any service, so a scheme registered
there is available by the time the URI is resolved.

When the class can be built as `MyService(uri=..., **kwargs)` and needs no
logic of its own, skip Python and declare it in `services.yaml` beside your
agents:

```yaml
services:
  - scheme: dynamo
    type: session
    class: my_package.stores.DynamoSessionService
```

`type` is one of `session`, `artifact`, `memory`, or `task_store`.

## How it works

Two moments matter, and they happen far apart. Your registrations run once at
startup, and a URI is resolved against them later, when a service is built.

### The load step

`load_services_module(agents_dir)` runs once during server startup, before any
service is constructed. It does three things:

1.  Inserts `agents_dir` at the front of `sys.path`, if it is not already
    there.
2.  Loads `services.yaml`, or `services.yml` if there is no `.yaml`, and
    registers everything in it.
3.  Imports the top-level module named `services`.

Both files may be present, and then both load. YAML goes first, so a scheme
declared in both ends up with the `services.py` definition, and more generally
a later registration of the same scheme silently replaces an earlier one.

A YAML file that fails to parse aborts the whole step with a warning and
`services.py` is never imported. A `services.py` that raises is logged at
warning level and startup continues, so a broken registration produces a server
that comes up and then rejects your URI. Neither case stops the server.

### Resolution

Pass a URI to one of the `create_*_service` methods and it takes the scheme off
the front, looks that scheme up among the ones registered for its kind, and
calls your factory with the full URI plus whatever keyword arguments the caller
supplied. **If the scheme is not registered you get `None` back, not an
error**, and that holds for all three kinds. The registry never raises and never
warns, so a misspelled scheme is invisible at this layer.

What happens to that `None` is the caller's business, and the CLI does something
different for each kind:

*   **Sessions** fall back to `DatabaseSessionService`, treating the URI as a
    SQLAlchemy URL. That is how AlloyDB and Spanner work with no registration at
    all. An unusable string then fails with SQLAlchemy's own message about the
    URL format, which does not mention schemes.
*   **Memory** raises `ValueError("Unsupported memory service URI: ...")`.
    `get_fast_api_app` catches it and re-raises it as a `click.ClickException`.
*   **Artifacts** depend on who is asking. `get_fast_api_app` raises, the way
    memory does. The default behavior everywhere else is to log a warning and
    substitute an in-memory artifact service, so a typo costs you every artifact
    silently.

A `kwargs` entry named `agents_dir` is passed to every factory, so yours should
accept `**kwargs` and ignore what it does not recognize.

The A2A task store is the exception: an unregistered task-store scheme raises
`ValueError` listing the supported schemes rather than returning `None`.

### The built-in schemes

`get_service_registry()` registers these on first call.

| Kind | Schemes |
| :--- | :--- |
| Session | `memory`, `sqlite`, `postgresql`, `mysql`, `agentengine` |
| Artifact | `memory`, `gs`, `file` |
| Memory | `memory`, `rag`, `agentengine` |
| A2A task store | `memory`, `postgresql+asyncpg`, `mysql+aiomysql`, `sqlite+aiosqlite` |

Registering one of those names replaces the built-in for the whole process.
That is a legitimate way to swap in your own SQLite implementation, and also a
reliable way to break something by accident, since nothing warns.

## Functions and types

The registry is reached through one accessor, one loader, and the methods on the
object they hand back.

| Symbol | Signature | Description |
| :--- | :--- | :--- |
| `get_service_registry` | `() -> ServiceRegistry` | The process-wide singleton, built and populated on first call. |
| `load_services_module` | `(agents_dir: str) -> None` | Loads `services.yaml` and `services.py` from a directory. |
| `ServiceFactory` | `Protocol` | `(uri: str, **kwargs) -> BaseSessionService \| BaseArtifactService \| BaseMemoryService` |
| `register_session_service` | `(scheme: str, factory: ServiceFactory) -> None` | |
| `register_artifact_service` | `(scheme: str, factory: ServiceFactory) -> None` | |
| `register_memory_service` | `(scheme: str, factory: ServiceFactory) -> None` | |
| `create_session_service` | `(uri: str, **kwargs) -> BaseSessionService \| None` | |
| `create_artifact_service` | `(uri: str, **kwargs) -> BaseArtifactService \| None` | |
| `create_memory_service` | `(uri: str, **kwargs) -> BaseMemoryService \| None` | |

The factory is any callable matching the protocol; it does not need to be a
`ServiceFactory` subclass, and a lambda or a class works as well as a function.

**A2A task stores have no registration method of their own in that table.** The
supported way to add a task-store scheme is the YAML form with
`type: task_store`.

## Advanced applications

Two questions come up once the basic registration works: how settings reach a
factory, and how to register when there is no `services.py` to put the call in.

### Read configuration out of the URI

A factory receives the whole URI, so it can carry more than a name. Parse it
rather than string-slicing:

```python
from urllib.parse import parse_qs
from urllib.parse import urlparse


def dynamo_session_factory(uri: str, **kwargs) -> DynamoSessionService:
  parsed = urlparse(uri)
  options = parse_qs(parsed.query)
  return DynamoSessionService(
      table_name=parsed.netloc,
      region=options.get("region", ["us-east-1"])[0],
  )
```

That accepts `dynamo://agent-sessions?region=eu-west-1`. Keep credentials out of
it, because the URI is passed on a command line and shows up in process listings
and shell history.

### Register from an application instead of a file

`services.py` exists for servers you do not control the startup of. When you do
control it, call the registry directly before building the app:

```python
get_service_registry().register_memory_service("dynamo", dynamo_memory_factory)

app = get_fast_api_app(
    agents_dir="./agents", web=False, memory_service_uri="dynamo://memories"
)
```

The registry is a module-level singleton, so a registration made anywhere in
the process is visible everywhere in it. That is also the argument for doing it
once at startup rather than inside a request handler.

## Limitations

*   **`services.py` is imported once per process, under the fixed module name
    `services`.** A second call to `load_services_module` with a different
    directory does nothing, because `services` is already in `sys.modules`. The
    second file is never executed and its schemes are never registered, and
    nothing says so. A process serving two agents directories gets only the
    first one's registrations. The same collision applies to any unrelated
    module named `services` that was imported earlier.
*   **Registration is process-global and last-write-wins.** No warning on
    replacing a scheme, including a built-in one.
*   **A failure to load is a warning, not an error.** A broken `services.yaml`
    also prevents `services.py` from being imported, and the server still
    starts.
*   **An unknown scheme returns `None`.** The registry itself never tells you
    that you misspelled a scheme; you learn it from whatever the caller does
    with `None`, which for sessions is to hand the string to SQLAlchemy and fail
    with a message about database drivers, and for artifacts outside
    `get_fast_api_app` is to fall back to in-memory storage.
*   **A task-store scheme can only be registered from YAML.** There is no
    supported Python call for it, unlike sessions, artifacts and memory.
*   **The YAML form cannot pass constructor arguments.** It builds
    `cls(uri=uri, **kwargs)` and nothing else, so anything needing a client, a
    pool or a credential belongs in `services.py`.

## Related samples

*   [services.py](../../../../contributing/samples/services.py) registers a
    memory service against the scheme `foo` in Python.
*   [services.yaml](../../../../contributing/samples/services.yaml) does the
    same thing declaratively, against the scheme `bar`.
*   [dummy_services.py](../../../../contributing/samples/dummy_services.py)
    holds the two throwaway memory services those files register.

## Related guides

*   [get_fast_api_app](../fast_api/index.md) is the function that calls
    `load_services_module` and then resolves your URIs.
*   [Session and BaseSessionService](../../sessions/session/index.md) covers the
    interface a custom session backend implements.
*   [BaseMemoryService](../../memory/memory_service/index.md) and
    [BaseArtifactService](../../artifacts/artifact_service/index.md) cover the
    other two.
