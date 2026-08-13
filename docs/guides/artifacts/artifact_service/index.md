# BaseArtifactService

`BaseArtifactService` is the storage interface ADK uses for named binary blobs —
a generated report, a chart, an uploaded PDF. Tools and callbacks reach it
through the `Context` they already receive, and every write produces a new
numbered version.

## Introduction

A tool that produces a 200 KB report has nowhere good to put it. Returning it as
the tool result pushes the whole thing into the model's context on this turn and
every turn after it, and writing it to local disk breaks as soon as the agent is
served from more than one machine.

The artifact service is the answer to that. It stores a `types.Part` under a
filename scoped to the app, the user, and (usually) the session, and hands back
an integer version. The tool returns the filename and version — a few dozen
bytes — and the payload is fetched only when something actually needs it. Three
implementations ship with ADK behind the same interface, and an agent's code
does not change when you swap one for another; only the `artifact_service` you
hand to the `Runner` does.

## Get started

The service is wired once, on the `Runner`. A tool then saves through its
`Context` parameter, which ADK injects by type annotation.

The example below runs that end to end. The model calls `save_report`, the tool
writes `report.md` into the artifact service and returns only the filename and
the version it got back, and after the run finishes the caller loads those bytes
out of the same service.

```python
import asyncio

from google.adk import Context
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def save_report(topic: str, ctx: Context) -> dict[str, str | int]:
  """Writes a short report on a topic and stores it as an artifact.

  Args:
    topic: What the report should be about.
  """
  body = f"# {topic}\n\nEverything worth knowing about {topic}."
  version = await ctx.save_artifact(
      "report.md",
      types.Part.from_bytes(data=body.encode(), mime_type="text/markdown"),
  )
  return {"filename": "report.md", "version": version}


agent = LlmAgent(
    name="report_agent",
    model="gemini-2.5-flash",
    instruction="Use save_report to write the report the user asks for.",
    tools=[save_report],
)

runner = Runner(
    app_name="report_app",
    agent=agent,
    session_service=InMemorySessionService(),
    artifact_service=InMemoryArtifactService(),
)


async def main() -> None:
  await runner.session_service.create_session(
      app_name="report_app", user_id="u1", session_id="s1"
  )
  async for _ in runner.run_async(
      user_id="u1",
      session_id="s1",
      new_message=types.Content(
          role="user", parts=[types.Part(text="Write a report on otters.")]
      ),
  ):
    pass

  report = await runner.artifact_service.load_artifact(
      app_name="report_app", user_id="u1", session_id="s1", filename="report.md"
  )
  print(report.inline_data.data.decode())


if __name__ == "__main__":
  asyncio.run(main())
```

The `ctx: Context` parameter is found by its annotation rather than its name, and
it is stripped from the declaration the model sees — the model only knows about
`topic`. Callbacks get the same object, since `CallbackContext` and `ToolContext`
are both aliases of `Context`.

Reading it back inside another tool is one call, and returns `None` when nothing
was ever saved under that name:

```python
part = await ctx.load_artifact("report.md")
```

## Versioning

Artifacts are append-only. `save_artifact` never overwrites: it appends a new
version and returns its number. The first save of a filename returns `0`, and
each subsequent save of that same filename returns one more than the last.
`load_artifact` returns the highest version when `version` is not given, and an
exact version when it is:

```python
latest = await ctx.load_artifact("report.md")
original = await ctx.load_artifact("report.md", version=0)
```

Version metadata is available separately from the payload, so you can read the
MIME type or your own bookkeeping without fetching the bytes.
`ctx.get_artifact_version(filename)` returns an `ArtifactVersion`, carrying
`version`, `canonical_uri`, `mime_type`, `create_time`, and whatever
`custom_metadata` dict you passed to `save_artifact`:

```python
await ctx.save_artifact(
    "report.md",
    types.Part.from_bytes(data=b"# otters", mime_type="text/markdown"),
    custom_metadata={"topic": "otters"},
)

info = await ctx.get_artifact_version("report.md")
print(info.mime_type, info.custom_metadata)
# text/markdown {'topic': 'otters'}
```

Each write is also recorded on the event ADK emits for that turn, in
`ctx.actions.artifact_delta`, as a `{filename: version}` mapping. That is how the
session history knows which artifacts a turn produced.

## The `user:` filename prefix

By default an artifact belongs to one session, so two sessions for the same user
do not see each other's `report.md`. Prefixing the filename with `user:` changes
that. The service inspects the filename, and when it starts with `user:` it
stores the artifact under the user rather than the session — every session
belonging to that user, now or later, reads and writes the same file.

```python
# Session-scoped: only this conversation sees it.
await ctx.save_artifact("draft.md", types.Part.from_text(text="a draft"))

# User-scoped: every session for this user sees it.
await ctx.save_artifact(
    "user:preferences.json", types.Part.from_text(text='{"theme": "dark"}')
)

await ctx.list_artifacts()
# ['draft.md', 'user:preferences.json']
```

Two things about the prefix are easy to get wrong. It is part of the name, not a
flag that gets consumed: you load it back with the same `"user:preferences.json"`
string, and `list_artifacts()` returns it with the prefix still attached, as
above. And it is the *only* lever a tool has, because `Context` always passes the
current session to the service — there is no `scope=` argument to reach for.

Calling `list_artifacts()` from a *different* session of the same user returns
`['user:preferences.json']` alone.

There is no matching `app:` prefix. Every artifact is already stored under a
path that starts with the app and the user, as
`apps/<app>/users/<user>/…`, so the prefix only chooses between session scope
and user scope inside one app. An artifact shared by every user of an app is
not something the interface can express.

## Choosing an implementation

| Implementation | Constructor | Use it for |
| --- | --- | --- |
| `InMemoryArtifactService` | `InMemoryArtifactService()` | Tests and local development. Everything is lost when the process exits, and it is not safe under concurrent writers. |
| `FileArtifactService` | `FileArtifactService(root_dir)` | A single machine that needs artifacts to survive a restart. Filenames may contain `/` and become nested directories. |
| `GcsArtifactService` | `GcsArtifactService(bucket_name)` | Serving from more than one process or machine. Extra keyword arguments are forwarded to the Cloud Storage client. |

`InMemoryRunner` picks `InMemoryArtifactService` for you, which is why artifact
code works in tests without any wiring.

Deploying to Agent Engine does not pick one for you. `adk deploy agent_engine`
points sessions and memory at the managed Agent Engine services, but there is
no managed artifact service behind it. Pass
`--artifact_service_uri=gs://<bucket>` to get `GcsArtifactService`; leave it off
and the deployed agent falls back to `InMemoryArtifactService` and loses every
artifact when it restarts.

Writing your own means subclassing `BaseArtifactService` and implementing its
seven abstract methods: `save_artifact`, `load_artifact`, `list_artifact_keys`,
`delete_artifact`, `list_versions`, `list_artifact_versions`, and
`get_artifact_version`. All are keyword-only and take `app_name`, `user_id`, and
an optional `session_id`, where `None` means the user-scoped namespace. Your
implementation is responsible for honoring the `user:` prefix, since the routing
lives in the service and not above it.

## Letting the model fetch an artifact

Tools decide for themselves what to load. To let the *model* decide, add the
built-in `load_artifacts` tool:

```python
from google.adk.tools import load_artifacts

agent = LlmAgent(name="report_agent", tools=[save_report, load_artifacts])
```

The model is told which filenames exist and can call `load_artifacts` with the
ones it wants; their content is attached to the next request only, not written
into the session. This is what keeps a large payload out of the context window
until the turn that genuinely needs it.

## Limitations

*   **A `Runner` has no artifact service unless you give it one.** Calling
    `ctx.save_artifact` when `artifact_service` is `None` raises `ValueError`
    rather than silently doing nothing.
*   **Deletion is all-or-nothing.** `delete_artifact` drops every version of a
    filename; there is no API for removing one version or trimming history, so a
    frequently-rewritten artifact grows without bound.
*   **No cross-user or cross-app access.** Artifact reference URIs are validated
    against the caller's app, user, and session, so an artifact saved under one
    user cannot be read from another.

## Related samples

*   [Artifacts](../../../../contributing/samples/core/artifacts) — saving text,
    HTML, image, audio, and video artifacts, and loading them back by version.
*   [Context offloading with artifacts](../../../../contributing/samples/patterns/context_offloading_with_artifact)
    — keeping large tool output out of the context window until it is needed.
