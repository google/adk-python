# inject_session_state

`inject_session_state` is what turns the `{user_name}` in
`instruction="Hello {user_name}."` into a name. It substitutes session state
values and artifact contents into an instruction string, resolving the `{var}`,
`{var?}` and `{artifact.name}` placeholders. ADK already
runs it over every string `instruction=` you write, and you call it yourself in
the one case where the framework does not, which is when your instruction is a
callable.

## Introduction

An agent instruction is usually a constant string. As soon as it has to mention
something from the current conversation, though, it needs to be assembled from
session state. That might be the user's name, the file they uploaded a moment
ago, or a preference they set three turns back.

ADK does that assembly for you, but only in one of the two ways you can supply
an instruction, and the difference catches people out:

*   Give a plain string, as in `instruction="Hello {user_name}."`, and ADK runs
    `inject_session_state` over it before every model call, so `{user_name}` is
    replaced with the value from session state.
*   Give a callable, as in `instruction=build_instruction`, and ADK calls your
    function and uses what it returns **verbatim**. State injection is
    deliberately skipped here, on the assumption that a function which can read
    the context can do its own interpolation.

That second case is why this function is public. If your provider returns a
string containing `{user_name}`, nothing substitutes it and the model sees the
braces. Calling `inject_session_state` yourself is how you get the placeholder
behavior back.

`InstructionProvider` is the type alias for such a callable, spelled
`Callable[[ReadonlyContext], str | Awaitable[str]]`. It is declared alongside
`inject_session_state` and exported from the same module.

One thing to weigh before you reach for placeholders at all: a resolved
instruction becomes the system instruction, and a context cache is keyed on a
prefix that includes it. An instruction whose text changes between requests
therefore costs you the cache. See
[What a dynamic instruction costs a context cache](#what-a-dynamic-instruction-costs-a-context-cache).

## Get started

Write an instruction provider that does some work of its own and still resolves
placeholders.

```python
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.utils.instructions_utils import inject_session_state


async def build_instruction(readonly_context: ReadonlyContext) -> str:
  base = "You are a support agent."
  if readonly_context.state.get("escalated"):
    base += " This case has been escalated; be brief and precise."
  return await inject_session_state(
      base + " The customer is {customer_name} on plan {plan_tier?}.",
      readonly_context,
  )


root_agent = Agent(
    name="support_agent",
    description="Answers customer support questions.",
    instruction=build_instruction,
)
```

With `{"customer_name": "Ada"}` in session state and no `plan_tier`, the model
receives `You are a support agent. The customer is Ada on plan .`, because the
question mark on `plan_tier?` turns a missing value into an empty string
instead of an error.

## Placeholder syntax

The default engine recognizes three forms. The difference that matters between
them is what each one does when the value is not there, because that is the case
the syntax does not make obvious.

*   **`{name}` is the required form.** The engine looks up `name` in session
    state and substitutes `str()` of the value. A key that is not in state
    raises `KeyError` in the middle of the request, which is the behavior you
    want when the instruction makes no sense without the value. An agent whose
    instruction reads `You are helping {customer_name}.` is better off failing
    than telling the model it is helping nobody in particular.
*   **`{name?}` is the optional form.** The trailing question mark is stripped
    from the key before the lookup, and a missing key substitutes an empty
    string rather than raising. Reach for it whenever the key is only sometimes
    set. The cost is that the sentence around the placeholder has to still read
    sensibly when the value is gone: `The customer is on plan {plan_tier?}.`
    renders as `The customer is on plan .`, so a phrasing that tolerates the
    empty case, such as putting the value on a line of its own, is worth the
    extra thought.
*   **`{artifact.filename}` loads a file instead of a state value.** The engine
    asks the artifact service for that artifact in the current session and
    substitutes `str()` of what comes back. A missing artifact raises
    `KeyError`, and `{artifact.filename?}` substitutes an empty string in the
    same way the optional state form does. Check what that `str()` produces
    before relying on the form: an artifact fetched from an artifact service is
    a `types.Part`, and its `str()` is the whole Pydantic field dump rather than
    the text you had in mind.

A state key may carry one of the state prefixes, so `{app:theme}`,
`{user:locale}`, and `{temp:draft}` all work, and read from the corresponding
scope. See [the State guide](../../sessions/state/index.md) for what those
prefixes mean. One catch when you are testing: a `temp:` key passed to
`create_session(state=...)` is dropped, because temp state is never persisted.
`{temp:draft}` resolves only once something in the invocation has written it.

A state value of `None` renders as an empty string, not as the text `None`.
Every other value is rendered with `str()`.

**Text that merely contains braces is left alone.** Before substituting, the
engine checks whether the contents look like a state name: a Python identifier,
optionally prefixed with `app:`, `user:`, or `temp:`. Anything else is returned
unchanged, so an instruction that includes `{"role": "user"}` as an example of
JSON, or `{}` as an empty object, survives intact. The check is on shape, not on
presence, so `{customer_name}` with no `customer_name` in state is a valid name
that is missing, and raises.

## How it works

`inject_session_state` reaches the session and the artifact service through the
`ReadonlyContext` you pass it, so you can only call it somewhere a context
exists, such as an instruction provider, a plugin callback, or a tool.

The default engine matches a run of opening braces, then text containing no
braces of its own, then a run of closing braces. The braces are stripped from
both ends and the remainder is trimmed. A template with no `{` in it at all is
returned as it stands, which is the common case for a static instruction and
costs nothing on each model call.

For each match the engine decides between three cases.

*   A name starting with `artifact.` triggers `artifact_service.load_artifact`
    for the session, and raises
    `ValueError("Artifact service is not initialized.")` if the runner has no
    artifact service.
*   A name that is not a valid state name is left as it was written.
*   Anything else is looked up in session state.

Substitution is a single left-to-right pass, so a value that itself contains
braces is inserted literally and never re-scanned.

### Where the framework already calls it

Knowing these four places tells you where placeholders do and do not work.

*   Every string `instruction`, and on the root agent every string
    `global_instruction`, before each model call. That first call site is the one
    nearly all agents rely on.
*   A string global instruction supplied at the app level.
*   `ManagedAgent`'s system instruction.
*   The body of a skill whose frontmatter sets
    `metadata.adk_inject_state: true`, at the moment the model loads it. See
    the Skill guide.

In each of the first three, injection is skipped precisely when the instruction
came from a callable. An `LlmAgent`'s `static_instruction` is never passed
through this function at all.

### What a dynamic instruction costs a context cache

A resolved instruction is the agent's system instruction, and a context cache is
keyed on a prefix that includes it. So the question is not whether your
instruction has placeholders. It is whether the values behind them change.

If a placeholder resolves to the same text on every request, such as a
`{user:locale}` that was set once and left alone, the system instruction is
stable and the cache keeps matching. If it resolves to something that varies,
such as a customer name that differs per user or a value that moves each turn,
the system instruction no longer matches the one the cache was built on, and
that request pays for the whole prefix again at full price. The same holds for
anything else that rewrites system content per request, whether that is a
template, a callback, a plugin, or a tool such as `ExampleTool`.

You can keep the cache by splitting the instruction in two. Put the text that
never changes in `static_instruction`, which is sent as the system instruction
with no substitution at all, and leave the placeholders in `instruction`, which
then travels as ordinary user content rather than as system instruction. The
varying text still reaches the model, and it is no longer inside the cached
prefix. Caching itself is turned on through `App.context_cache_config`, covered
in [the App guide](../../apps/app/index.md).

## Configuration options

The function takes two required arguments and one switch that changes the
template language entirely.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `template` | `str` | *required* | The instruction text containing placeholders. |
| `readonly_context` | `ReadonlyContext` | *required* | Supplies the session state and the artifact service. |
| `use_jinja2` | `bool` | `False` | Render with Jinja2 instead of the brace engine. |

## Advanced applications

Two things push past the ordinary case. An instruction whose shape, rather than
whose wording, depends on state needs a real template language, and a string
that is not an instruction at all can still go through the same substitution.

### Conditionals and loops with Jinja2

The brace engine can only substitute. It cannot include a paragraph
conditionally, or iterate over a list in state, so an instruction that has to
change shape rather than change a word needs a real template language. Pass
`use_jinja2=True` and session state keys become top-level template variables,
while artifacts are loaded with an async `artifact()` helper:

```python
async def build_instruction(readonly_context: ReadonlyContext) -> str:
  return await inject_session_state(
      "{% if show_hint is defined and show_hint %}Hint: read the docs.{% endif %}"
      "{% for item in items %}{{ item }} {% endfor %}",
      readonly_context,
      use_jinja2=True,
  )
```

The two engines share nothing but this function. Their syntax differs, with
`{{ var }}` against `{var}` and `{{ artifact('report.md') }}` against
`{artifact.report.md}`, and so does their behavior on a missing value. ADK
builds the Jinja2 environment with `undefined=jinja2.StrictUndefined`, so a
variable that is not in session state raises rather than rendering empty, and
there is no `?` equivalent. A missing artifact raises `KeyError` in both
engines. Autoescaping is off, which is right for prompt text and would be wrong
for HTML.

`StrictUndefined` is why `is defined` appears in the example above. **A bare
`{% if show_hint %}` also raises** when `show_hint` is absent, because testing
an undefined name for truthiness is enough to trigger it:
`jinja2.exceptions.UndefinedError: 'show_hint' is undefined`. That is the
opposite of the usual Jinja2 behavior and it catches people, because "guard the
optional key with an `if`" is exactly the thing that does not work. Write
`{% if x is defined and x %}`, or guarantee the key is in state.

Jinja2 is an optional dependency. It ships with ADK's evaluation and testing
extras but not with the base install, and calling with `use_jinja2=True` without
it raises `ImportError` telling you to `pip install jinja2`. The import happens
inside the function rather than at module scope, so merely importing ADK never
requires it.

### Render something that is not an agent instruction

The same placeholder syntax is often wanted in a tool description, a prompt
fragment, or text you are about to write to an artifact. Nothing ties the
function to instructions. Anywhere you hold a `ReadonlyContext`, and a
`ToolContext` is one, you can call it on an arbitrary string.

## Limitations

*   **A missing key raises by default.** `{name}` with no `name` in state raises
    `KeyError` mid-request. If the key is only sometimes present, `{name?}` is
    almost always what you want.
*   **Nested braces are not supported, and fail quietly.** Inner braces are
    never matched, so in `{outer{inner}}` the only match is `{inner}}`, because
    the run of closing braces is consumed with it. With `inner` set to `I`, the
    whole thing renders as `{outerI`. No error, and the opening brace is still
    in the prompt.
*   **No recursion.** A substituted value containing `{other}` is inserted
    literally.
*   **Silence on typos.** `{costumer_name}` is a valid state name that is
    missing, so it raises. `{customer name}` is not a valid name, so it is left
    in the prompt untouched and the model sees the braces. The two typos fail in
    completely different ways.
*   **Artifacts are stringified, and that is almost never what you want.** Both
    engines insert `str(artifact)`, and an artifact loaded from an artifact
    service is a `types.Part`, whose `str()` is the Pydantic field dump. A
    plain-text artifact reading "hello world" reaches the prompt as
    `media_resolution=None code_execution_result=None ... text='hello world'
    thought=None ...`, which is every field of the Part when only one of them
    was what you wanted. Binary data is worse still. There is no option to
    extract `.text`, so an
    instruction that needs an artifact's contents should load it in Python and
    interpolate the text itself rather than using `{artifact.name}`.
*   **A callable instruction gets no injection.** The omission is deliberate,
    on the assumption that a function with access to the context can interpolate
    for itself, and it is the reason to call this function yourself.
*   **A value that changes between requests defeats context caching.** The
    resolved instruction is the system instruction, so a placeholder whose value
    moves changes the cached prefix and the next request misses. Move the
    unchanging text into `static_instruction`.

## Related samples

*   [Skills with state injection](../../../../contributing/samples/environment_and_skills/skills_inject_state)
    routes a `SKILL.md` body through this function, using `{dev_name?}`-style
    optional placeholders.

## Related guides

*   [State](../../sessions/state/index.md) covers what goes in session state and
    what the `app:`, `user:`, and `temp:` prefixes mean.
*   [BaseArtifactService](../../artifacts/artifact_service/index.md) is what
    `{artifact.name}` loads from.
*   [ManagedAgent](../../agents/managed_agent/index.md) is one of the framework
    call sites.
