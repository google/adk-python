# ADK exceptions

`google.adk.errors` holds the six exception types ADK raises on its own behalf
rather than passing along from a dependency. They come from four parts of the
framework: the session services, the artifact services, the evaluation
subsystem, and tool code.

## Introduction

Most of the failures you see from ADK come from somewhere else, whether that is
a `ValidationError` from Pydantic, an `APIError` from the GenAI SDK, or a
`ModuleNotFoundError` from an extra you have not installed. The six types in
`google.adk.errors` are the ones ADK defines itself, and each of them marks a
place where the framework has a specific condition to report and expects you to
be able to do something about it.

Three facts shape how you work with them, and all three catch people out.

*   **There is no common base class.** These six do not descend from a shared
    `AdkError`, so there is no single `except` clause that means "an ADK error".
    Three of them subclass `ValueError` and three subclass `Exception` directly.
*   **Three of them subclass `ValueError`.** If you already have an
    `except ValueError` anywhere near session or artifact code, it is catching
    `StaleSessionError`, `SessionNotFoundError` and `InputValidationError` right
    now, almost certainly without you intending it to. That inheritance is
    deliberate backward compatibility, and of everything these six types do it
    is the most likely to cost you data.
*   **Only one is re-exported from the package.** You can write
    `from google.adk.errors import StaleSessionError`, and the other five have
    to be reached by module path. `StaleSessionError` has no public module of
    its own, so the package import is the only route to it.

## The six exceptions

| Exception | Base | Import from |
| :--- | :--- | :--- |
| `StaleSessionError` | `ValueError` | `google.adk.errors` |
| `SessionNotFoundError` | `ValueError` | `google.adk.errors.session_not_found_error` |
| `InputValidationError` | `ValueError` | `google.adk.errors.input_validation_error` |
| `AlreadyExistsError` | `Exception` | `google.adk.errors.already_exists_error` |
| `NotFoundError` | `Exception` | `google.adk.errors.not_found_error` |
| `ToolExecutionError` | `Exception` | `google.adk.errors.tool_execution_error` |

### `StaleSessionError`

**Raised by** `append_event` on the session services that check for concurrent
writes: `DatabaseSessionService`, `SqliteSessionService`, and
`FirestoreSessionService`.

**Means** the in-memory `Session` you are holding has fallen behind the stored
copy, because somebody else wrote to that session after you read it. Without
that check your append would have silently overwritten the other writer's
history.

**Catch it.** This is the one exception on the page with a real recovery: re-read
the session with `get_session` and replay your append against the fresh copy.

It defines no constructor of its own and inherits `ValueError`'s. A message you
pass will show up in `str()`, but there is no `.message` attribute to read it
back from and no default text either, so `str(StaleSessionError())` gives you the
empty string.

### `SessionNotFoundError`

**Raised by** `Runner.run_async` and `run_live` when the session ID you passed
does not exist and `auto_create_session` is `False`, which is the default. Also
by `append_event` on the database, SQLite and Firestore services when the
session is not in storage.

**Means** you referenced a session that was never created, or that has been
deleted. It is not a transient condition.

**Do not catch it as flow control.** Seeing it means your application skipped
`create_session`, so the fix is to create the session first. If starting a new
conversation on an unknown ID is genuinely what you want, set
`auto_create_session=True` on the `Runner` instead. The API server maps this
error to HTTP 404.

One thing to watch for is that `get_session` does *not* raise it. A missing
session comes back from there as `None`.

### `InputValidationError`

**Raised by** the artifact services, on an identifier or a payload that fails
validation. Every artifact service rejects an `app_name`, `user_id` or
`session_id` that is empty, holds a null byte, is an absolute or
drive-qualified path, or carries a `..` traversal segment.
`FileArtifactService` applies the same rules to the *filename*, and also
confirms that the resolved path stayed inside the storage directory.

Payload validation is where the three services part company, and the
differences tend to surface as a surprise when you switch backends:

| Payload | `InMemory` | `File` | `Gcs` |
| :--- | :--- | :--- | :--- |
| `inline_data` with no bytes | stored | rejected | rejected |
| `file_data` with no URI | stored | rejected | rejected |
| Any `file_data` at all | stored | rejected | stored |
| Empty `Part` | rejected | rejected | rejected |

`InMemoryArtifactService` and `GcsArtifactService` also reject a malformed
`artifact://` reference URI, along with a reference that points outside the
caller's own app, user or session scope. `FileArtifactService` stores no
references at all, so it turns away every `file_data` part long before that
question comes up. It has one guard the others do not: it rejects a filename that
would collide with the metadata document it writes beside each version.

**Means** a value that arrived from outside your code is either unsafe or
unusable, and in practice it usually arrived from a model-generated tool call.
Several of these checks are path-traversal guards, so read one as a signal about
untrusted input rather than as a bug in your own code. Only `FileArtifactService`
guards the filename, and on `InMemoryArtifactService` a filename containing `..`
is stored under that literal name without complaint, because there is no
filesystem there to escape from.

**Catch it at the boundary** where you accept artifact names, and turn it into an
error response of your own. The API server maps it to HTTP 400.

### `AlreadyExistsError`

**Raised by** `create_session` on every session service, including
`InMemorySessionService`, when you pass a `session_id` that is already taken.

**Means** exactly that, and nothing more.

**Catch it** if your application generates session IDs itself and a collision is
possible, which happens most often when a retried request replays a session
creation. If you let ADK generate the ID by leaving `session_id` out, you will
never see this one. The API server maps it to HTTP 409.

### `NotFoundError`

**Raised by** the evaluation subsystem, and nowhere else. The eval-set managers
raise it for an unknown eval set, eval case, or eval-set result. The metric
evaluator registry raises it for a metric with no registered evaluator, and the
persona registry raises it for an unknown simulator persona.

**Means** a named evaluation resource does not exist. The generic name oversells
it, because this is not a general-purpose "not found". A missing *session* raises
`SessionNotFoundError`, and a missing *file* raises a plain `FileNotFoundError`.

**Catch it** when you are driving evaluation programmatically and want to tell
"no such eval set" apart from a real failure. The dev server maps it to HTTP 404
on most endpoints, and downgrades it to a logged warning when it is listing eval
sets for an app that has none.

### `ToolExecutionError`

**Raised by** tool code. ADK itself raises it only from
`ExampleTool.from_config`, on an examples value it cannot use, so in practice
this type exists for *you* to raise from your own tools.

**Means** a tool failed in a way worth classifying. It takes an optional
`error_type`, which is either a `ToolErrorType` enum member or a raw string such
as `"500"`. An enum member is normalized to its string value, and the result is
readable back off the exception as `error_type`.

**Raise it rather than catching it.** The reason to reach for it over a bare
`Exception` is telemetry. Whatever you put in `error_type` becomes the
OpenTelemetry `error.type` span attribute, and the exception's class name is
used only when there is nothing there. That means
`ToolExecutionError("timed out", ToolErrorType.REQUEST_TIMEOUT)` records
`REQUEST_TIMEOUT` in your traces, where a bare `TimeoutError` would only ever
record `TimeoutError`.

`ToolErrorType` is a `str` enum of nine HTTP-shaped values that follow
OpenTelemetry semantics: `BAD_REQUEST`, `UNAUTHORIZED`, `FORBIDDEN`,
`NOT_FOUND`, `REQUEST_TIMEOUT`, `INTERNAL_SERVER_ERROR`, `BAD_GATEWAY`,
`SERVICE_UNAVAILABLE`, and `GATEWAY_TIMEOUT`.

```python
raise ToolExecutionError(
    f"Inventory service did not respond for sku={sku}",
    ToolErrorType.GATEWAY_TIMEOUT,
)
```

## The `ValueError` trap

`StaleSessionError`, `SessionNotFoundError` and `InputValidationError` all
subclass `ValueError`, and each of them says so for backward compatibility.
Callers were catching `ValueError` in these places before the specific types
existed, and narrowing the base class afterwards would have broken all of that
code at once.

The bill for that decision is paid by new code. An `except ValueError` anywhere
near a session or artifact call now swallows three conditions you almost
certainly wanted to hear about, and it does so without a word. The code below
looks like careful error handling and behaves like a hole in the floor:

```python
try:
  await session_service.append_event(session, event)
except ValueError:
  logger.warning("bad event, skipping")
```

A `StaleSessionError` there means another writer reached the session first and
your append needs replaying. What that handler does instead is log a line about a
bad event and carry on, so the turn's history is thrown away and nobody finds out
until a user notices the conversation is missing a chunk of itself. Catch the
specific type, and put its clause above any `ValueError` clause:

```python
try:
  await session_service.append_event(session, event)
except StaleSessionError:
  session = await session_service.get_session(
      app_name=session.app_name,
      user_id=session.user_id,
      session_id=session.id,
  )
  await session_service.append_event(session, event)
```

When one clause should genuinely cover several of these conditions, name them in
a tuple, because there is no base class to name instead:
`except (StaleSessionError, SessionNotFoundError)`. And where the surrounding
code has a reason of its own to catch `ValueError`, keep that clause and put the
ADK types above it. Python takes the first clause that matches, so a broad one
placed first hides every narrower one below it.

The mistake in the other direction is worth knowing about too. Because
`AlreadyExistsError`, `NotFoundError` and `ToolExecutionError` derive from
`Exception` and *not* from `ValueError`, an `except ValueError` wrapped around
session creation or an eval-set lookup catches nothing whatsoever, and the
exception sails straight past it.

## Limitations

*   **No shared base class.** You cannot write one `except` for "any ADK error",
    and adding a base class later would itself be a breaking change for anyone
    relying on the current ones.
*   **The package re-exports only `StaleSessionError`.** The other five need
    their module path, which means a longer import and a path with no obvious
    guarantee of stability.
*   **The constructors are inconsistent.** `NotFoundError`,
    `AlreadyExistsError` and `InputValidationError` take an optional message
    with a sensible default and expose it as `.message`. `ToolExecutionError`
    also exposes `.message`, but requires it. `SessionNotFoundError` has a
    default of its own but exposes no `.message`. `StaleSessionError` defines no
    constructor at all, so `str(StaleSessionError())` is the empty string while
    every other default-constructed type here gives you a sentence.
*   **`NotFoundError` is named more generally than it behaves.** It belongs to
    the evaluation subsystem, and a missing session or file raises something
    else entirely.
*   **Nothing carries structured detail.** Apart from `ToolExecutionError` and
    its `error_type`, you get no error code, no offending identifier, and no
    cause field. The message string is all there is.

## Related guides

*   [Session and BaseSessionService](../sessions/session/index.md) covers the
    lifecycle that raises `AlreadyExistsError`, `SessionNotFoundError` and
    `StaleSessionError`.
*   `DatabaseSessionService`
    explains the revision checking behind `StaleSessionError`.
*   [BaseArtifactService](../artifacts/artifact_service/index.md) has the
    identifier and payload rules that raise `InputValidationError`.
*   [Runner and InMemoryRunner](../runners/runner/index.md) documents
    `auto_create_session` and when `SessionNotFoundError` reaches you.
*   `Evaluator` describes the registry lookups
    that raise `NotFoundError`.
