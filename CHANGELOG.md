# Changelog

## [1.3.0](https://github.com/google/adk-python/compare/v1.2.1...v1.3.0) (2025-06-11)


### Features

* Add memory_service option to CLI ([416dc6f](https://github.com/google/adk-python/commit/416dc6feed26e55586d28f8c5132b31413834c88))
* Add support for display_name and description when deploying to agent engine ([aaf1f9b](https://github.com/google/adk-python/commit/aaf1f9b930d12657bfc9b9d0abd8e2248c1fc469))
* Dev UI: Trace View
  * New trace tab which contains all traces grouped by user messages
  * Click each row will open corresponding event details
  * Hover each row will highlight the corresponding message in dialog
* Dev UI: Evaluation
  * Evaluation Configuration: users can now configure custom threshold for the metrics used for each eval run ([d1b0587](https://github.com/google/adk-python/commit/d1b058707eed72fd4987d8ec8f3b47941a9f7d64))
  * Each eval case added can now be viewed and edited. Right now we only support edit of text.
  * Show the used metric in evaluation history ([6ed6351](https://github.com/google/adk-python/commit/6ed635190c86d5b2ba0409064cf7bcd797fd08da))
* Tool enhancements:
  * Add url_context_tool ([fe1de7b](https://github.com/google/adk-python/commit/fe1de7b10326a38e0d5943d7002ac7889c161826))
  * Support to customize timeout for mcpstdio connections ([54367dc](https://github.com/google/adk-python/commit/54367dcc567a2b00e80368ea753a4fc0550e5b57))
  * Introduce write protected mode to BigQuery tools ([6c999ca](https://github.com/google/adk-python/commit/6c999caa41dca3a6ec146ea42b0a794b14238ec2))



### Bug Fixes

* Agent Engine deployment:
  * Correct help text formatting for `adk deploy agent_engine` ([13f98c3](https://github.com/google/adk-python/commit/13f98c396a2fa21747e455bb5eed503a553b5b22))
  * Handle project and location in the .env properly when deploying to Agent Engine ([0c40542](https://github.com/google/adk-python/commit/0c4054200fd50041f0dce4b1c8e56292b99a8ea8))
* Fix broken agent graphs ([3b1f2ae](https://github.com/google/adk-python/commit/3b1f2ae9bfdb632b52e6460fc5b7c9e04748bd50))
* Forward `__annotations__` to the fake func for FunctionTool inspection ([9abb841](https://github.com/google/adk-python/commit/9abb8414da1055ab2f130194b986803779cd5cc5))
* Handle the case when agent loading error doesn't have msg attribute in agent loader ([c224626](https://github.com/google/adk-python/commit/c224626ae189d02e5c410959b3631f6bd4d4d5c1))
* Prevent agent_graph.py throwing when workflow agent is root agent ([4b1c218](https://github.com/google/adk-python/commit/4b1c218cbe69f7fb309b5a223aa2487b7c196038))
* Remove display_name for non-Vertex file uploads ([cf5d701](https://github.com/google/adk-python/commit/cf5d7016a0a6ccf2b522df6f2d608774803b6be4))


### Documentation

* Add DeepWiki badge to README ([f38c08b](https://github.com/google/adk-python/commit/f38c08b3057b081859178d44fa2832bed46561a9))
* Update code example in tool declaration to reflect BigQuery artifact description ([3ae6ce1](https://github.com/google/adk-python/commit/3ae6ce10bc5a120c48d84045328c5d78f6eb85d4))


## [1.2.1](https://github.com/google/adk-python/compare/v1.2.0...v1.2.1) (2025-06-04)


### Bug Fixes

* Import deprecated from typing_extensions ([068df04](https://github.com/google/adk-python/commit/068df04bcef694725dd36e09f4476b5e67f1b456))


## [1.2.0](https://github.com/google/adk-python/compare/v1.1.1...v1.2.0) (2025-06-04)


### Features

* Add agent engine as a deployment option to the ADK CLI ([2409c3e](https://github.com/google/adk-python/commit/2409c3ef192262c80f5328121f6dc4f34265f5cf))
* Add an option to use gcs artifact service in adk web. ([8d36dbd](https://github.com/google/adk-python/commit/8d36dbda520b1c0dec148e1e1d84e36ddcb9cb95))
* Add index tracking to handle parallel tool call using litellm ([05f4834](https://github.com/google/adk-python/commit/05f4834759c9b1f0c0af9d89adb7b81ea67d82c8))
* Add sortByColumn functionality to List Operation ([af95dd2](https://github.com/google/adk-python/commit/af95dd29325865ec30a1945b98e65e457760e003))
* Add implementation for  `get_eval_case`, `update_eval_case` and `delete_eval_case` for the local eval sets manager. ([a7575e0](https://github.com/google/adk-python/commit/a7575e078a564af6db3f42f650e94ebc4f338918))
* Expose more config of VertexAiSearchTool from latest Google GenAI SDK ([2b5c89b](https://github.com/google/adk-python/commit/2b5c89b3a94e82ea4a40363ea8de33d9473d7cf0))
* New Agent Visualization ([da4bc0e](https://github.com/google/adk-python/commit/da4bc0efc0dd96096724559008205854e97c3fd1))
* Set the max width and height of view image dialog to be 90% ([98a635a](https://github.com/google/adk-python/commit/98a635afee399f64e0a813d681cd8521fbb49500))
* Support Langchain StructuredTool for Langchain tool ([7e637d3](https://github.com/google/adk-python/commit/7e637d3fa05ca3e43a937e7158008d2b146b1b81))
* Support Langchain tools that has run_manager in _run args and don't have args_schema populated ([3616bb5](https://github.com/google/adk-python/commit/3616bb5fc4da90e79eb89039fb5e302d6a0a14ec))
* Update for anthropic models ([16f7d98](https://github.com/google/adk-python/commit/16f7d98acf039f21ec8a99f19eabf0ef4cb5268c))
* Use bigquery scope by default in bigquery credentials. ([ba5b80d](https://github.com/google/adk-python/commit/ba5b80d5d774ff5fdb61bd43b7849057da2b4edf))
* Add jira_agent adk samples code which connect Jira cloud ([8759a25](https://github.com/google/adk-python/commit/8759a2525170edb2f4be44236fa646a93ba863e6))
* Render HTML artifact in chat window ([5c2ad32](https://github.com/google/adk-python/commit/5c2ad327bf4262257c3bc91010c3f8c303d3a5f5))
* Add export to json button in the chat window ([fc3e374](https://github.com/google/adk-python/commit/fc3e374c86c4de87b4935ee9c56b6259f00e8ea2))
* Add tooltip to the export session button ([2735942](https://github.com/google/adk-python/commit/273594215efe9dbed44d4ef85e6234bd7ba7b7ae))


### Bug Fixes

* Add adk icon for UI ([2623c71](https://github.com/google/adk-python/commit/2623c710868d832b6d5119f38e22d82adb3de66b))
* Add cache_ok option to remove sa warning. ([841e10a](https://github.com/google/adk-python/commit/841e10ae353e0b1b3d020a26d6cac6f37981550e))
* Add support for running python main function in UnsafeLocalCodeExecutor when the code has an if __name__ == "__main__" statement. ([95e33ba](https://github.com/google/adk-python/commit/95e33baf57e9c267a758e08108cde76adf8af69b))
* Adk web not working on some env for windows, fixes https://github.com/google/adk-web/issues/34 ([daac8ce](https://github.com/google/adk-python/commit/daac8cedfe6d894f77ea52784f0a6d19003b2c00))
* Assign empty inputSchema to MCP tool when converting an ADK tool that wraps a function which takes no parameters. ([2a65c41](https://github.com/google/adk-python/commit/2a65c4118bb2aa97f2a13064db884bd63c14a5f7))
* Call all tools in parallel calls during partial authentication ([0e72efb](https://github.com/google/adk-python/commit/0e72efb4398ce6a5d782bcdcb770b2473eb5af2e))
* Continue fetching events if there are multiple pages. ([6506302](https://github.com/google/adk-python/commit/65063023a5a7cb6cd5db43db14a411213dc8acf5))
* Do not convert "false" value to dict ([60ceea7](https://github.com/google/adk-python/commit/60ceea72bde2143eb102c60cf33b365e1ab07d8f))
* Enhance agent loader exception handler and expose precise error information ([7b51ae9](https://github.com/google/adk-python/commit/7b51ae97245f6990c089183734aad41fe59b3330))
* Ensure function description is copied when ignoring parameters ([7fdc6b4](https://github.com/google/adk-python/commit/7fdc6b4417e5cf0fbc72d3117531914353d3984a))
* Filter memory by app_name and user_id. ([db4bc98](https://github.com/google/adk-python/commit/db4bc9809c7bb6b0d261973ca7cfd87b392694be))
* Fix filtering by user_id for vertex ai session service listing ([9d4ca4e](https://github.com/google/adk-python/commit/9d4ca4ed44cf10bc87f577873faa49af469acc25))
* fix parameter schema generation for gemini ([5a67a94](https://github.com/google/adk-python/commit/5a67a946d2168b80dd6eba008218468c2db2e74e))
* Handle non-indexed function call chunks with incremental fallback index ([b181cbc](https://github.com/google/adk-python/commit/b181cbc8bc629d1c9bfd50054e47a0a1b04f7410))
* Handles function tool parsing corner case where type hints are stored as strings. ([a8a2074](https://github.com/google/adk-python/commit/a8a20743f92cd63c3d287a3d503c1913dd5ad5ae))
* Introduce PreciseTimestamp to fix mysql datetime precision issue. ([841e10a](https://github.com/google/adk-python/commit/841e10ae353e0b1b3d020a26d6cac6f37981550e))
* match arg case in errors ([b226a06](https://github.com/google/adk-python/commit/b226a06c0bf798f85a53c591ad12ee582703af6d))
* ParallelAgent should only append to its immediate sub-agent, not transitive descendants ([ec8bc73](https://github.com/google/adk-python/commit/ec8bc7387c84c3f261c44cedfe76eb1f702e7b17))
* Relax openapi spec to gemini schema conversion to tolerate more cases ([b1a74d0](https://github.com/google/adk-python/commit/b1a74d099fae44d41750b79e58455282d919dd78))
* Remove labels from config when using API key from Google AI Studio to call model ([5d29716](https://github.com/google/adk-python/commit/5d297169d08a2d0ea1a07641da2ac39fa46b68a4))
* **sample:** Correct text artifact saving in artifact_save_text sample ([5c6001d](https://github.com/google/adk-python/commit/5c6001d90fe6e1d15a2db6b30ecf9e7b6c26eee4))
* Separate thinking from text parts in streaming mode ([795605a](https://github.com/google/adk-python/commit/795605a37e1141e37d86c9b3fa484a3a03e7e9a6))
* Simplify content for ollama provider ([eaee49b](https://github.com/google/adk-python/commit/eaee49bc897c20231ecacde6855cccfa5e80d849))
* Timeout issues for mcpstdio server when mcp tools are incorrect. ([45ef668](https://github.com/google/adk-python/commit/45ef6684352e3c8082958bece8610df60048f4a3))
* **transfer_to_agent:** update docstring for clarity and accuracy ([854a544](https://github.com/google/adk-python/commit/854a5440614590c2a3466cf652688ba57d637205))
* Update unit test code for test_connection ([b0403b2](https://github.com/google/adk-python/commit/b0403b2d98b2776d15475f6b525409670e2841fc))
* Use inspect.cleandoc on function docstrings in generate_function_declaration. ([f7cb666](https://github.com/google/adk-python/commit/f7cb66620be843b8d9f3d197d6e8988e9ee0dfca))
* Restore errors path ([32c5ffa](https://github.com/google/adk-python/commit/32c5ffa8ca5e037f41ff345f9eecf5b26f926ea1))
* Unused import for deprecated ([ccd05e0](https://github.com/google/adk-python/commit/ccd05e0b00d0327186e3b1156f1b0216293efe21))
* Prevent JSON parsing errors and preserve non-ascii characters in telemetry ([d587270](https://github.com/google/adk-python/commit/d587270327a8de9f33b3268de5811ac756959850))
* Raise HTTPException when running evals in fast_api if google-adk[eval] is not installed ([1de5c34](https://github.com/google/adk-python/commit/1de5c340d8da1cedee223f6f5a8c90070a9f0298))
* Fix typos in README for sample bigquery_agent and oauth_calendar_agent ([9bdd813](https://github.com/google/adk-python/commit/9bdd813be15935af5c5d2a6982a2391a640cab23))
* Make tool_call one span for telemetry and renamed to execute_tool ([999a7fe](https://github.com/google/adk-python/commit/999a7fe69d511b1401b295d23ab3c2f40bccdc6f))
* Use media type in chat window. Remove isArtifactImage and isArtifactAudio reference ([1452dac](https://github.com/google/adk-python/commit/1452dacfeb6b9970284e1ddeee6c4f3cb56781f8))
* Set output_schema correctly for LiteLllm ([6157db7](https://github.com/google/adk-python/commit/6157db77f2fba4a44d075b51c83bff844027a147))
* Update pending event dialog style ([1db601c](https://github.com/google/adk-python/commit/1db601c4bd90467b97a2f26fe9d90d665eb3c740))
* Remove the gap between event holder and image ([63822c3](https://github.com/google/adk-python/commit/63822c3fa8b0bdce2527bd0d909c038e2b66dd98))


### Documentation

* Adds a sample agent to illustrate state usage via `callbacks`. ([18fbe3c](https://github.com/google/adk-python/commit/18fbe3cbfc9f2af97e4b744ec0a7552331b1d8e3))
* Fix typos in documentation ([7aaf811](https://github.com/google/adk-python/commit/7aaf8116169c210ceda35c649b5b49fb65bbb740))
* Change eval_dataset to eval_dataset_file_path_or_dir ([62d7bf5](https://github.com/google/adk-python/commit/62d7bf58bb1c874caaf3c56a614500ae3b52f215))
* Fix broken link to A2A example ([0d66a78](https://github.com/google/adk-python/commit/0d66a7888b68380241b92f7de394a06df5a0cc06))
* Fix typo in envs.py ([bd588bc](https://github.com/google/adk-python/commit/bd588bce50ccd0e70b96c7291db035a327ad4d24))
* Updates CONTRIBUTING.md to refine setup process using uv. ([04e07b4](https://github.com/google/adk-python/commit/04e07b4a1451123272641a256c6af1528ea6523e))
* Create and update project documentation including README.md and CONTRIBUTING.md ([f180331](https://github.com/google/adk-python/commit/f1803312c6a046f94c23cfeaed3e8656afccf7c3))
* Rename the root agent in the example to match the example name ([94c0aca](https://github.com/google/adk-python/commit/94c0aca685f1dfa4edb44caaedc2de25cc0caa41))
* ADK: add section comment ([349a414](https://github.com/google/adk-python/commit/349a414120fbff0937966af95864bd683f063d08))


### Chore

* Miscellaneous changes ([0724a83](https://github.com/google/adk-python/commit/0724a83aa9cda00c1b228ed47a5baa7527bb4a0a), [a9dcc58](https://github.com/google/adk-python/commit/a9dcc588ad63013d063dbe37095c0d2e870142c3), [ac52eab](https://github.com/google/adk-python/commit/ac52eab88eccafa451be7584e24aea93ff15f3f3), [a0714b8](https://github.com/google/adk-python/commit/a0714b8afc55461f315ede8451b17aad18d698dd))
* Enable release-please workflow ([57d99aa](https://github.com/google/adk-python/commit/57d99aa7897fb229f41c2a08034606df1e1e6064))
* Added unit test coverage for local_eval_sets_manager.py ([174afb3](https://github.com/google/adk-python/commit/174afb3975bdc7e5f10c26f3eebb17d2efa0dd59))
* Extract common options for `adk web` and `adk api_server` ([01965bd](https://github.com/google/adk-python/commit/01965bdd74a9dbdb0ce91a924db8dee5961478b8))

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
