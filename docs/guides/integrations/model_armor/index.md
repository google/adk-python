# Model Armor

`ModelArmorPlugin` screens user input and model output against [Google Cloud Model Armor](https://cloud.google.com/security-command-center/docs/model-armor-overview) templates. When a filter matches, or when screening cannot complete, the offending content is replaced with a safe message before it reaches the model or the user.

## Introduction

Model Armor is a Google Cloud service that inspects text for prompt injection and jailbreak attempts, harmful content, and sensitive data. You define what to look for in a *template* — a named, server-side policy — and the service returns a verdict for each piece of text you send it.

The integration is two public types: `ModelArmorPlugin`, a `BasePlugin` subclass driven by `PluginManager`, and `ModelArmorConfig`, which says which templates to screen against and what to do about a match. The plugin reads text off the `LlmRequest` and `LlmResponse`, calls Model Armor, and returns a replacement `LlmResponse` when content should be blocked.

Key features:

- **Input and output screening**, each governed by its own template, and each optional.
- **Block screening failures by default**: by default screening failures are blocked rather than delivered.
- **Blocked responses are marked** with `custom_metadata['model_armor_blocked']` so your application can detect them.

## Get started

Install the dependency:

```shell
pip install 'google-adk[gcp]'
```

Create your templates in the Google Cloud console, then register the plugin on an `App`:

```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.integrations.model_armor import ModelArmorPlugin

agent = LlmAgent(
    name="screened_agent",
    description="Assistant whose input and output are screened.",
    instruction="You are a helpful assistant.",
)

app = App(
    name="model_armor_demo",
    root_agent=agent,
    plugins=[
        ModelArmorPlugin(
            config=ModelArmorConfig(
                prompt_template_name="projects/my-project/locations/us-central1/templates/my-prompt-template",
                response_template_name="projects/my-project/locations/us-central1/templates/my-response-template",
            )
        )
    ],
)
```

The plugin screens user inputs against `prompt_template_name` and model outputs against `response_template_name`. Blocked turns are substituted with `input_blocked_message` or `output_blocked_message` respectively.

Credentials come from Application Default Credentials.

## How it works

### `before_model_callback` - screening input

`before_model_callback` runs before each model call:

1. If `prompt_template_name` is unset, it returns immediately and nothing is screened.
2. It walks `llm_request.contents` backwards for the most recent `user` content with text parts.
3. It sends that text to Model Armor's `SanitizeUserPrompt` method.
4. It acts on the result (below).

### `after_model_callback` - screening output

`after_model_callback` runs after each model response:

1. If `response_template_name` is unset, it returns immediately and nothing is screened.
2. Otherwise it reads whichever form the text arrived in:
   - **Unary**: text parts on `llm_response.content`.
   - **Live**: `llm_response.output_transcription`, this is checked first.
3. The text is sent to Model Armor's `SanitizeModelResponse` method.
4. It acts on the result (below).

### Acting on a result

| `invocation_result` | Meaning | Plugin behavior |
| :--- | :--- | :--- |
| `SUCCESS` | Every filter ran. | Check `filter_match_state`. |
| Anything else | Some or all filters were skipped, failed, or the field was unset. | Screening failure. |

When screening completes successfully, a `filter_match_state` of `MATCH_FOUND` means at least one filter tripped, and the content is blocked. Anything else passes through untouched.

A screening failure is routed through `block_on_screening_failure` and blocked by default.

### The blocked response

Blocking returns an `LlmResponse` carrying the message for the direction that
was screened: `input_blocked_message` for user input, `output_blocked_message`
for model output.

### Template paths and regional endpoints

Templates must be specified using their full resource paths:
`projects/{project}/locations/{location}/templates/{template}`

The plugin parses the `{location}` segment from the configured template names to target the appropriate regional service endpoint (e.g. `modelarmor.us-central1.rep.googleapis.com`). Because one plugin instance talks to one regional endpoint, all of its configured templates must belong to the same region.

## Configuration options

### Plugin options

Options introduced by `ModelArmorPlugin` (those inherited from `BasePlugin` are omitted):

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `config` | `ModelArmorConfig` | required | Templates and enforcement behavior. |
| `name` | `str` | `"model_armor_plugin"` | Plugin instance identifier. |
| `client` | `ModelArmorAsyncClient \| None` | `None` | A pre-built SDK client, mainly for tests. Built from `config` when omitted. |
| `credentials` | `Credentials \| None` | `None` | Credentials used when building the client. Defaults to Application Default Credentials. |

- **`config`** carries everything that decides what gets screened and what happens on a match. See [ModelArmorConfig fields](#modelarmorconfig-fields) below.
- **`name`** matters when you register more than one instance, for example a strict template on one agent and a permissive one on another.
- **`client`** lets you inject a double in tests, or an SDK client you configured yourself.
- **`credentials`** custom credentials can be provided that override Application Default Credentials.

### `ModelArmorConfig` fields

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `prompt_template_name` | `str \| None` | `None` | Template used to screen user input. Unset means input is not screened. |
| `response_template_name` | `str \| None` | `None` | Template used to screen model output. Unset means output is not screened. |
| `input_blocked_message` | `str` | `"I'm sorry, but I can't help with that request."` | Replacement text shown when user input is blocked. |
| `output_blocked_message` | `str` | `"I'm sorry, but I can't help with that request."` | Replacement text shown when model output is blocked. |
| `block_on_screening_failure` | `bool` | `True` | Whether to block content that could not be screened. |

At least one of the two template names must be set.

#### `prompt_template_name` and `response_template_name`

Both fields require fully-qualified resource paths formatted as:
`projects/{project}/locations/{location}/templates/{template}`

You can configure either or both:

- `prompt_template_name`: Screens user input prompts before forwarding to the model.
- `response_template_name`: Screens model responses before delivering to the user.

If both are set they must reside in the same GCP location — see [Template paths and regional endpoints](#template-paths-and-regional-endpoints).

#### `input_blocked_message` and `output_blocked_message`

Defines the replacement text returned to the user when a prompt or response is blocked. Screening failures reuse the message for the direction that failed.

#### `block_on_screening_failure`

Controls how the plugin behaves when Model Armor cannot return a definitive `SUCCESS` verdict.

- **`True` (default)**: Blocks the content. Unscreened content is treated as unsafe.
- **`False`**: Delivers the content.

## Advanced applications

### Screening one direction only

```python
config = ModelArmorConfig(
    prompt_template_name="projects/my-project/locations/us-central1/templates/my-prompt-template",
)
```

### Staying available when Model Armor is down

```python
config = ModelArmorConfig(
    prompt_template_name="projects/my-project/locations/us-central1/templates/my-prompt-template",
    block_on_screening_failure=False,
)
```

### Detecting blocks in your application

Blocked responses carry a marker, so a UI can render them differently from a real answer:

```python
async for event in runner.run_async(...):
    if (event.custom_metadata or {}).get("model_armor_blocked"):
        ...  # show a policy notice rather than a model reply
```

## Limitations

- **Tool output is not screened.** Only the most recent `user` content with text parts is sent for screening. Tool results are added to the request as `user` content whose only part is a `function_response` and doesn't reach Model Armor.

- **Enforcement mode is limited.** The Model Armor plugin is currently limited to logging detection results and blocking content. Future extensions could include replacing or redacting text.

- **Live audio screening uses transcriptions.** The Model Armor plugin currently screens audio via input and output transcriptions, which relies on their accuracy.
