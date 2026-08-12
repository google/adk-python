# BaseLlm and LLMRegistry

`BaseLlm` is the interface that every model implementation in ADK satisfies.
`LLMRegistry` is the lookup that turns a model name such as
`"gemini-3.5-flash"` into an instance of one.

## Introduction

An agent names the model it wants as a plain string. Something has to decide
which class serves that string, and it has to do so without importing every
backend ADK can talk to. That is the job of the model layer. `BaseLlm` defines
the contract — accept an `LlmRequest`, yield `LlmResponse` objects — and
`LLMRegistry` maps model-name regexes to the classes that implement it.

`LlmAgent.model` accepts either a string or a `BaseLlm` instance. Given a
string, `LlmAgent.canonical_model` calls `LLMRegistry.new_llm` once and caches
the instance, resolving again only if `model` is reassigned. An agent with no
model of its own inherits from the nearest `LlmAgent` ancestor, and failing that
gets `LlmAgent.DEFAULT_MODEL` (currently `gemini-3.5-flash`) unless
`LlmAgent.set_default_model` has overridden it. Live mode resolves separately,
through `canonical_live_model` and `LlmAgent.DEFAULT_LIVE_MODEL`.

Plugging in a model ADK does not ship therefore has two forms. Pass an instance
and the registry is never consulted. Register the class and a plain model name
resolves to it.

Subclassing `BaseLlm` is the ordinary way to add a backend, not an escape hatch.
Every non-Gemini provider ADK ships is built that way: `LiteLlm`, `Claude`,
`OpenAILlm`, and `OCIGenAILlm` all subclass it and are registered against their
own model-name patterns, exactly as the example below registers `EchoLlm`.

## Get started

A complete model implementation. It answers with the text it was sent, so it
runs with no credentials and no network.

```python
import asyncio
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.models import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.runners import InMemoryRunner
from google.genai import types


class EchoLlm(BaseLlm):
  """A stand-in model that answers with the text it was sent."""

  @classmethod
  def supported_models(cls) -> list[str]:
    # Any model name fully matching one of these regexes resolves to this class.
    return [r'echo-.*']

  @property
  def capabilities(self) -> LlmCapabilities:
    # Declare capabilities outright in a direct BaseLlm subclass.
    return LlmCapabilities(output_schema_and_tools=False)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    prompt = llm_request.contents[-1].parts[0].text
    yield LlmResponse(
        content=types.Content(
            role='model',
            parts=[types.Part(text=f'{self.model} heard: {prompt}')],
        )
    )


LLMRegistry.register(EchoLlm)

agent = LlmAgent(name='echo_agent', model='echo-v1')

asyncio.run(InMemoryRunner(agent=agent).run_debug('hello'))
```

`LLMRegistry.register` reads `supported_models()` and files the class under
each regex it returns, so `"echo-v1"` now resolves the way `"gemini-3.5-flash"`
does. To skip the registry, hand the agent an instance:
`LlmAgent(name='echo_agent', model=EchoLlm(model='echo-v1'))`.

## How a name is resolved

`LLMRegistry.resolve` returns the class for a name and `LLMRegistry.new_llm`
resolves and then constructs it. Resolution tries the following, in order.

1.  **An explicit class override.** A name shaped like `prefix:model` treats the
    prefix as a class name and skips regex matching. The comparison is
    case-insensitive and ignores a trailing `Llm`, so `lite:openai/gpt-4o` and
    `LiteLlm:openai/gpt-4o` both select `LiteLlm`. `new_llm` strips the prefix
    before construction, giving `LiteLlm(model='openai/gpt-4o')`. A prefix
    matching no class name is left in the model string.
2.  **A regex match.** Registered patterns are tried in registration order and
    the first one matching the *whole* name wins. Order is load-bearing:
    `gemma-4.*` is registered alongside the Gemini patterns, which come first,
    so `gemma-4-1b` resolves to `Gemini` while `gemma-3-1b` resolves to `Gemma`.
3.  **A LiteLLM provider.** If nothing matched and the name contains a slash,
    the text before it is checked against LiteLLM's own provider list. That is
    why `xai/grok-4`, which the registry never spells out, still resolves to
    `LiteLlm` when LiteLLM is installed.
4.  **Failure.** Otherwise `resolve` raises `ValueError`, naming the optional
    package to install for a `claude-` or `provider/model` name.

`resolve` is memoized, and `register` clears that cache, so registering a class
over a name that has already been resolved does take effect.

### Lazy entries

A registry entry holds either a class or the module path and class name to
import it from. ADK's built-in providers are filed as the latter, so importing
`google.adk.models` pulls in neither `anthropic` nor `litellm` nor any other
optional dependency. The first time such an entry matches, its module is
imported and the entry is replaced by the class. If that import fails the entry
is discarded and matching continues with the next pattern.

## The request and the response

`LlmRequest` is what the framework hands a model. It is a Pydantic model, and a
`before_model_callback` receives the same object.

*   `model` is the resolved model's own name, which the flow copies from
    `canonical_model`. Built-in implementations read it in preference to
    `self.model`.
*   `contents` is the conversation as a `list[types.Content]`.
*   `config` is a `types.GenerateContentConfig` carrying the system
    instruction, the tool declarations, the generation parameters, and any
    response schema. `live_connect_config` is its counterpart for live mode.
*   `tools_dict` maps a declared tool name to the `BaseTool` behind it.
*   `cache_config` and `cache_metadata` carry context caching state.

Build a request with `append_instructions`, `append_tools`, and
`set_output_schema` rather than by mutating `config` directly.

`LlmResponse` is what comes back. `content` holds the generated
`types.Content`, and `get_function_calls` and `get_function_responses` pull the
function-call parts out of it. `usage_metadata`, `grounding_metadata`,
`citation_metadata`, and `finish_reason` carry the rest of the turn's metadata.
An error is reported in-band through `error_code` and `error_message` rather
than as an exception. A backend whose wire format is already a
`types.GenerateContentResponse` should use the `LlmResponse.create` static
method, which performs that mapping including the error cases.

Streaming has a contract worth restating. With `stream=True` a model yields
chunks with `partial=True` and then exactly one response with `partial=False`
holding the whole turn, identical to what `stream=False` would have yielded
once. Callers depend on that last response.

## Capabilities

`BaseLlm.capabilities` returns an `LlmCapabilities`, a frozen Pydantic model
whose fields answer what the model supports. Callers read it instead of
re-deriving support from the model name. A direct subclass of `BaseLlm` declares
its capabilities outright, as in the example above. A subclass of an existing
model builds on the parent's report instead, so that capabilities it does not
name keep the parent's value:

```python
from google.adk.models import Gemini


class MyGemini(Gemini):

  @property
  def capabilities(self) -> LlmCapabilities:
    return LlmCapabilities(
        **super().capabilities.model_dump() | {'output_schema_and_tools': True}
    )
```

Keep the override a plain property, not a cached one: a capability may depend on
state that changes after construction.

## Limitations

*   **Only `generate_content_async` is required.** `BaseLlm.connect` opens a
    live `BaseLlmConnection` for bidirectional streaming and raises
    `NotImplementedError` by default, so a model that does not override it
    cannot be used in live mode.
*   **A model that does not report capabilities gets a deprecated fallback.** A
    subclass that leaves `capabilities` alone falls back to inferring
    `output_schema_and_tools` from the model name, and emits a `FutureWarning`
    when that inference grants the capability. The fallback will be removed.
*   **`canonical_model` is framework API.** It is the agent's resolution entry
    point, documented for ADK's own use. Application code should read
    `LlmAgent.model` or hold its own `BaseLlm` instance.

## Related samples

*   [Model backends](../../../../contributing/samples/models)
