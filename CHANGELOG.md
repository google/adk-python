# Changelog

## 1.1.1

### Features
* Add BigQuery first-party tools. See [here](https://github.com/google/adk-python/commit/d6c6bb4b2489a8b7a4713e4747c30d6df0c07961) for more details.


## 1.1.0

### Features

* Extract agent loading logic from fast_api.py to a separate AgentLoader class and support more agent definition folder/file structure.
* Added audio play in web UI.
* Added input transcription support for live/streaming.
* Added support for storing eval run history locally in adk eval cli.
* Image artifacts can now be clicked directly in chat message to view.
* Left side panel can now be resized.

### Bug Fixes

* Avoid duplicating log in stderr.
* Align event filtering and ordering logic.
* Add handling for None param.annotation.
* Fixed several minor bugs regarding eval tab in web UI.

### Miscellaneous Chores

* Updates mypy config in pyproject.toml.
* Add google search agent in samples.
* Update filtered schema parameters for Gemini API.
* Adds autoformat.sh for formatting codebase.

## 1.0.0

### ⚠ BREAKING CHANGES

* Evaluation dataset schema is finalized with strong-type pydantic models.
  (previously saved eval file needs re-generation, for both adk eval cli and
  the eval tab in adk web UI).
* `BuiltInCodeExecutor` (in code_executors package) replaces
  `BuiltInCodeExecutionTool` (previously in tools package).
* All methods in services are now async, including session service, artifact
  service and memory service.
  * `list_events` and `close_session` methods are removed from session service.
* agent.py file structure with MCP tools are now easier and simpler ([now](https://github.com/google/adk-python/blob/3b5232c14f48e1d5b170f3698d91639b079722c8/contributing/samples/mcp_stdio_server_agent/agent.py#L33) vs [before](https://github.com/google/adk-python/blob/a4adb739c0d86b9ae4587547d2653d568f6567f2/contributing/samples/mcp_agent/agent.py#L41)).
  Old format is not working anymore.
* `Memory` schema and `MemoryService` is redesigned.
* Mark various class attributes as private in the classes in the `tools` package.
* Disabled session state injection if instruction provider is used.
  (so that you can have `{var_name}` in the instruction, which is required for code snippets)
* Toolbox integration is revamped: tools/toolbox_tool.py → tools/toolbox_toolset.py.
* Removes the experimental `remote_agent.py`. We'll redesign it and bring it back.

### Features

* Dev UI:
  * A brand new trace view for overall agent invocation.
  * A revamped evaluation tab and comparison view for checking eval results.
* Introduced `BaseToolset` to allow dynamically add/remove tools for agents.
  * Revamped MCPToolset with the new BaseToolset interface.
  * Revamped GoogleApiTool, GoogleApiToolset and ApplicationIntegrationToolset with the new BaseToolset interface.
  * Resigned agent.py file structure when needing MCPToolset.
  * Added ToolboxToolset.
* Redesigned strong-typed agent evaluation schema.
  * Allows users to create more cohesive eval sets.
  * Allows evals to be extended for non-text modality.
  * Allows for a structured interaction with the uber eval system.
* Redesigned Memory schema and MemoryService interfaces.
* Added token usage to LlmResponse.
* Allowed specifying `--adk_version` in `adk deploy cloud_run` cli. Default is the current version.

### Bug Fixes

* Fixed `adk deploy cloud_run` failing bug.
* Fixed logs not being printed due to `google-auth` library.

### Miscellaneous Chores

* Display full help text when adk cli receives invalid arguments.
* `adk web` now binds `127.0.0.1` by default, instead of 0.0.0.0.
* `InMemoryRunner` now takes `BaseAgent` in constructor.
* Various docstring improvements.
* Various UI tweaks.
* Various bug fixes.
* Update various contributing/samples for contributors to validate the implementation.


## 0.5.0

### ⚠ BREAKING CHANGES

* Updated artifact and memory service interface to be async. Agents that
  interact with these services through callbacks or tools will now need to
  adjust their invocation methods to be async (using await), or ensure calls
  are wrapped in an asynchronous executor like asyncio.run(). Any service that
  extends the base interface must also be updated.

### Features

* Introduced the ability to chain model callbacks.
* Added support for async agent and model callbacks.
* Added input transcription support for live/streaming.
* Captured all agent code error and display on UI.
* Set param required tag to False by default in openapi_tool.
* Updated evaluation functions to be asynchronous.

### Bug Fixes

* Ensured a unique ID is generated for every event.
* Fixed the issue when openapi_specparser has parameter.required as None.
* Updated the 'type' value on the items/properties nested structures for Anthropic models to adhere to JSON schema.
* Fix litellm error issues.

### Miscellaneous Chores

* Regenerated API docs.
* Created a `developer` folder and added samples.
* Updated the contributing guide.
* Docstring improvements, typo fixings, GitHub action to enforce code styles on formatting and imports, etc.

## 0.4.0

### ⚠ BREAKING CHANGES
* Set the max size of strings in database columns. MySQL mandates that all VARCHAR-type fields must specify their lengths.
* Extract content encode/decode logic to a shared util, resolve issues with JSON serialization, and update key length for DB table to avoid key too long issue in mysql.
* Enhance `FunctionTool` to verify if the model is providing all the mandatory arguments.

### Features
* Update ADK setup guide to improve onboarding experience.
* feat: add ordering to recent events in database session service.
* feat(llm_flows): support async before/after tool callbacks.
* feat: Added --replay and --resume options to adk run cli. Check adk run --help for more details.
* Created a new Integration Connector Tool (underlying of the ApplicationIntegrationToolSet) so that we do not force LLM to provide default value.

### Bug Fixes

* Don't send content with empty text to LLM.
* Fix google search reading undefined for `renderedContent`.

### Miscellaneous Chores
* Docstring improvements, typo fixings, github action to enfore code styles on formatting and imports, etc.

## 0.3.0

### ⚠ BREAKING CHANGES

* Auth: expose `access_token` and `refresh_token` at top level of auth
  credentials, instead of a `dict`
  ([commit](https://github.com/google/adk-python/commit/956fb912e8851b139668b1ccb8db10fd252a6990)).

### Features

* Added support for running agents with MCPToolset easily on `adk web`.
* Added `custom_metadata` field to `LlmResponse`, which can be used to tag
  LlmResponse via `after_model_callback`.
* Added `--session_db_url` to `adk deploy cloud_run` option.
* Many Dev UI improvements:
  * Better google search result rendering.
  * Show websocket close reason in Dev UI.
  * Better error message showing for audio/video.

### Bug Fixes

* Fixed MCP tool json schema parsing issue.
* Fixed issues in DatabaseSessionService that leads to crash.
* Fixed functions.py.
* Fixed `skip_summarization` behavior in `AgentTool`.

### Miscellaneous Chores

* README.md improvements.
* Various code improvements.
* Various typo fixes.
* Bump min version of google-genai to 1.11.0.

## 0.2.0

### ⚠ BREAKING CHANGES

* Fix typo in method name in `Event`: has_trailing_code_exeuction_result --> has_trailing_code_execution_result.

### Features

* `adk` CLI:
  * Introduce `adk create` cli tool to help creating agents.
  * Adds `--verbosity` option to `adk deploy cloud_run` to show detailed cloud
    run deploy logging.
* Improve the initialization error message for `DatabaseSessionService`.
* Lazy loading for Google 1P tools to minimize the initial latency.
* Support emitting state-change-only events from planners.
* Lots of Dev UI updates, including:
  * Show planner thoughts and actions in the Dev UI.
  * Support MCP tools in Dev UI.
    (NOTE: `agent.py` interface is temp solution and is subject to change)
  * Auto-select the only app if only one app is available.
  * Show grounding links generated by Google Search Tool.
* `.env` file is reloaded on every agent run.

### Bug Fixes

* `LiteLlm`: arg parsing error and python 3.9 compatibility.
* `DatabaseSessionService`: adds the missing fields; fixes event with empty
  content not being persisted.
* Google API Discovery response parsing issue.
* `load_memory_tool` rendering issue in Dev UI.
* Markdown text overflows in Dev UI.

### Miscellaneous Chores

* Adds unit tests in Github action.
* Improves test coverage.
* Various typo fixes.

## 0.1.0

### Features

* Initial release of the Agent Development Kit (ADK).
* Multi-agent, agent-as-workflow, and custom agent support
* Tool authentication support
* Rich tool support, e.g. built-in tools, google-cloud tools, third-party tools, and MCP tools
* Rich callback support
* Built-in code execution capability
* Asynchronous runtime and execution
* Session, and memory support
* Built-in evaluation support
* Development UI that makes local development easy
* Deploy to Google Cloud Run, Agent Engine
* (Experimental) Live(Bidi) audio/video agent support and Compositional Function Calling(CFC) support
