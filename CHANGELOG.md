# Changelog

## [1.18.0](https://github.com/google/adk-python/compare/v1.17.0...v1.18.0) (2025-11-05)

### Features

* **[ADK Visual Agent Builder]**
  * Core Features
    * Visual workflow designer for agent creation
    * Support for multiple agent types (LLM, Sequential, Parallel, Loop, Workflow)
    * Agent tool support with nested agent tools
    * Built-in and custom tool integration
    * Callback management for all ADK callback types (before/after agent, model, tool)
    * Assistant to help you build your agents with natural language
    * Assistant proposes and writes agent configuration yaml files for you
    * Save to test with chat interfaces as normal
    * Build and debug at the same time in adk web!

* **[Core]**
  * Add support for extracting cache-related token counts from LiteLLM usage ([4f85e86](https://github.com/google/adk-python/commit/4f85e86fc3915f0e67312a39fe22451968d4f1b1))
  * Expose the Python code run by the code interpreter in the logs ([a2c6a8a](https://github.com/google/adk-python/commit/a2c6a8a85cf4f556e9dacfe46cf384d13d964208))
  * Add run_debug() helper method for quick agent experimentation ([0487eea](https://github.com/google/adk-python/commit/0487eea2abcd05d7efd123962d17b8c6c9a9d975))
  * Allow injecting a custom Runner into `agent_to_a2a` ([156d235](https://github.com/google/adk-python/commit/156d23547915e8f7f5c6ba55e0362f4b133c3968))
  * Support MCP prompts via the McpInstructionProvider class ([88032cf](https://github.com/google/adk-python/commit/88032cf5c56bb2d81842353605f9f5ab4b2206ff))

* **[Models]**
  * Add model tracking to LiteLlm and introduce a LiteLLM with fallbacks demo ([d4c63fc](https://github.com/google/adk-python/commit/d4c63fc5629e7d70ad8b8185be09243a01e3428f))
  * Add ApigeeLlm as a model that lets ADK Agent developers to connect with an Apigee proxy ([87dcb3f](https://github.com/google/adk-python/commit/87dcb3f7ba344a2ba7d9edfc4817c9e792d90bfc))

* **[Integrations]**
  * Add example and fix for loading and upgrading old ADK session databases ([338c3c8](https://github.com/google/adk-python/commit/338c3c89c6bce7f3406f729013cedcd78b809a56))
  * Add support for specifying logging level for adk eval cli command ([b1ff85f](https://github.com/google/adk-python/commit/b1ff85fb2347e3402eedd42e3673be7093a99548))
  * Propagate LiteLLM finish_reason to LlmResponse for use in callbacks ([71aa564](https://github.com/google/adk-python/commit/71aa5645f6c3d91fd0e0ddb1ed564188c6727080))
  * Allow LLM request to override the model used in the generate content async method in LiteLLM ([ce8f674](https://github.com/google/adk-python/commit/ce8f674a287368439ba11be3285902671e9bc75a))
  * Add api key argument to Vertex Session and Memory services for Express Mode support ([9014a84](https://github.com/google/adk-python/commit/9014a849eab9f77b82db4a7f2053fb2a96282f03))
  * Added support for enums as arguments for function tools ([240ef5b](https://github.com/google/adk-python/commit/240ef5beea9389911e8c03a6039b353befc716ac))
  * Implement artifact_version related methods in GcsArtifactService ([e194ebb](https://github.com/google/adk-python/commit/e194ebb33c62bc40403ea852a88f77a9511b61a4))

* **[Services]**
  * Add support for Vertex AI Express Mode when deploying to Agent Engine ([d4b2a8b](https://github.com/google/adk-python/commit/d4b2a8b49f98a9991cb44ac7ec6e538b81a08664))
  * Remove custom polling logic for Vertex AI Session Service since LRO polling is supported in express mode ([546c2a6](https://github.com/google/adk-python/commit/546c2a68165f54e694664d5b6b6740566301782b))
  * Make VertexAiSessionService fully asynchronous ([f7e2a7a](https://github.com/google/adk-python/commit/f7e2a7a40ef248dd6fbba9669503b0828a12f0cc))

* **[Tools]**
  * Add Bigquery detect_anomalies tool ([9851340](https://github.com/google/adk-python/commit/9851340ad1df86d6f5c21e8984199573f239bb2b))
  * Extend Bigquery detect_anomalies tool to support future data anomaly detection ([38ea749](https://github.com/google/adk-python/commit/38ea749c9cec8e65f5e768f49fd2de79b5545571))
  * Add get_job_info tool to BigQuery toolset ([6429457](https://github.com/google/adk-python/commit/64294572c1c93590aa3c221015a5cb9b440ee948))

* **[Evals]**
  * Add "final_session_state" to the EvalCase data model ([2274c4f](https://github.com/google/adk-python/commit/2274c4f3040b20da3690aa03272155776ca330c1))
  * Marked expected_invocation as optional field on evaluator interface ([b17c8f1](https://github.com/google/adk-python/commit/b17c8f19e5fc67180d1bdc621f84cd43e357571c))
  * Adds LLM-backed user simulator ([54c4ecc](https://github.com/google/adk-python/commit/54c4ecc73381cffa51cff01c7fb8a2ac59308c53))

* **[Observability]**
  * Add BigQueryLoggingPlugin for event logging to BigQuery ([b7dbfed](https://github.com/google/adk-python/commit/b7dbfed4a3d4a0165e2c6e51594d1f547bec89d3))

* **[Live]**
  * Add token usage to live events for bidi streaming ([6e5c0eb](https://github.com/google/adk-python/commit/6e5c0eb6e0474f5b908eb9df20328e7da85ebed9))

### Bug Fixes

* Reduce logging spam for MCP tools without authentication ([11571c3](https://github.com/google/adk-python/commit/11571c37ab948d43cbaa3a1d82522256dfe4d467))
* Fix typo in several files ([d2888a3](https://github.com/google/adk-python/commit/d2888a3766b87df2baaaa1a67a2235b1b80f138f))
* Disable SetModelResponseTool workaround for Vertex AI Gemini 2+ models ([6a94af2](https://github.com/google/adk-python/commit/6a94af24bf3367c05a5d405b7e7b79810a1fac4e))
* Bug when callback_context_invocation_context is missing in GlobalInstructionPlugin ([f81ebdb](https://github.com/google/adk-python/commit/f81ebdb622211031945eb06c3f00ff5208d94f9b))
* Support models slash prefix in model name extraction ([8dff850](https://github.com/google/adk-python/commit/8dff85099d67623dd6f4a707fb932ea55b8aaf9b))
* Do not consider events with state delta and no content as final response ([1ee93c8](https://github.com/google/adk-python/commit/1ee93c8bcb7ccd6f33658dc76b2095dd7e58aac9))
* Parameter filtering for CrewAI functions with **kwargs ([74a3500](https://github.com/google/adk-python/commit/74a3500fc5d4b07e80f914d83a0d91face28086c))
* Do not treat FinishReason.STOP as error case for LLM responses containing candidates with empty contents ([2f72ceb](https://github.com/google/adk-python/commit/2f72ceb49b452c5a1f257bce6adb004fa5d54472))
* Fixes null check for reflect_retry plugin sample ([86f0155](https://github.com/google/adk-python/commit/86f01550bd1b52d6d160e8bc54cecc6c4fe8611c))
* Creates evalset directory on evalset create ([6c3882f](https://github.com/google/adk-python/commit/6c3882f2d66f169d393171be280b6e6218b52a7c))
* Add ADK_DISABLE_LOAD_DOTENV environment variable that disables automatic loading of .env when running ADK cli, if set to true or 1 ([15afbcd](https://github.com/google/adk-python/commit/15afbcd1587d4102a4dc5c07c0c493917df9d6ea))
* Allow tenacity 9.0.0 ([ee8acc5](https://github.com/google/adk-python/commit/ee8acc58be7421a3e8eab07b051c45f9319f80dc))
* Output file uploading to artifact service should handle both base64 encoded and raw bytes ([496f8cd](https://github.com/google/adk-python/commit/496f8cd6bb36d3ba333d7ab1e94e7796d2960300))
* Correct message part ordering in A2A history ([5eca72f](https://github.com/google/adk-python/commit/5eca72f9bfd05c7c28a3d738391138a59a31167d))
* Change instruction insertion to respect tool call/response pairs ([1e6a9da](https://github.com/google/adk-python/commit/1e6a9daa63050936ab421f1f684935927aebc63e))
* DynamicPickleType to support MySQL dialect ([fc15c9a](https://github.com/google/adk-python/commit/fc15c9a0c3c043c0a61dce625b8cd1ee121b4baf))
* Enable usage metadata in LiteLLM streaming ([f9569bb](https://github.com/google/adk-python/commit/f9569bbb1afbc7f0e8b6e68599590471fd112b9f))
* Fix issue with MCP tools throwing an error ([1a4261a](https://github.com/google/adk-python/commit/1a4261ad4b66cdeb39d39110a086bd6112b17516))
* Remove redundant `format` field from LiteLLM content objects ([489c39d](https://github.com/google/adk-python/commit/489c39db01465e38ecbc2c7f32781c349b8cddc9))
* Update the contribution analysis tool to use original write mode ([54db3d4](https://github.com/google/adk-python/commit/54db3d4434e0706b83a589fa2499d11d439a6e4e))
* Fix agent evaluations detailed output rows wrapping issue([4284c61](https://github.com/google/adk-python/commit/4284c619010b8246c1ecaa011f14b6cc9de512dd))
* Update dependency version constraints to be based on PyPI versions([0b1784e](https://github.com/google/adk-python/commit/0b1784e0e493a0e2df1edfe37e5ed5f4247e7d9d))

### Improvements

* Add Community Repo section to README ([432d30a](https://github.com/google/adk-python/commit/432d30af486329aa83f89c5d5752749a85c0b843))
* Undo adding MCP tools output schema to FunctionDeclaration ([92a7d19](https://github.com/google/adk-python/commit/92a7d1957367d498de773761edd142d8c108d751))
* Refactor ADK README for clarity and consistency ([b0017ae](https://github.com/google/adk-python/commit/b0017aed4472c73c3b07e71f1d65ae97a5293547))
* Add support for reversed proxy in adk web ([a0df75b](https://github.com/google/adk-python/commit/a0df75b6fa35d837086decb8802dbf1c0a6637ad))
* Avoid rendering empty columns as part of detailed results rendering of eval results ([5cb35db](https://github.com/google/adk-python/commit/5cb35db921bf86b5ad0012046bd19fa7cc1e6abb))
* Clear the behavior of disallow_transfer_to_parent ([48ddd07](https://github.com/google/adk-python/commit/48ddd078941f9240b10f052b6de171c310bc2bc6))
* Disable the scheduled execution for issue triage workflow ([a02f321](https://github.com/google/adk-python/commit/a02f321f1bdb8be9ad1873db804e0e8393268dc3))
* Include delimiter when matching events from parent nodes in content processor ([b8a2b6c](https://github.com/google/adk-python/commit/b8a2b6c57080ae29d7a02df7d9fcc2f961d422d2))
* Improve Tau-bench ADK colab stability ([04dbc42](https://github.com/google/adk-python/commit/04dbc42e50ce40ef3924d1c259e425215e12c2e7))
* Implement ADK-based agent factory for Tau-bench ([c0c67c8](https://github.com/google/adk-python/commit/c0c67c8698d70ddb9ed958416661f232ef9a5ed8))
* Add util to run ADK LLM Agent with simulation environment ([87f415a](https://github.com/google/adk-python/commit/87f415a7c36a1f3b6ab84d1fe939726c6ef7f34e))
* Demonstrate CodeExecutor customization for environment setup ([8eeff35](https://github.com/google/adk-python/commit/8eeff35b35d7e1538a5c9662cc8369f6ff7962f8))
* Add sample agent for VertexAiCodeExecutor ([edfe553](https://github.com/google/adk-python/commit/edfe5539421d196ca4da14d3a37fac7b598f8c8d))
* Adds a new sample agent that demonstrates how to integrate PostgreSQL databases using the Model Context Protocol (MCP) ([45a2168](https://github.com/google/adk-python/commit/45a2168e0e6773e595ecfb825d7e4ab0a38c3a38))
* Add example for using ADK with Fast MCP sampling ([d3796f9](https://github.com/google/adk-python/commit/d3796f9b33251d28d05e6701f11e80f02a2a49e1))
* Refactor gepa sample code and clean-up user demo colab([63353b2](https://github.com/google/adk-python/commit/63353b2b74e23e97385892415c5a3f2a59c3504f))

## [1.17.0](https://github.com/google/adk-python/compare/v1.16.0...v1.17.0) (2025-10-22)

### Features

* **[Core]**
  * Add a service registry to provide a generic way to register custom service implementations to be used in FastAPI server. See [short instruction](https://github.com/google/adk-python/discussions/3175#discussioncomment-14745120). ([391628f](https://github.com/google/adk-python/commit/391628fcdc7b950c6835f64ae3ccab197163c990))
  * Add the ability to rewind a session to before a previous invocation ([9dce06f](https://github.com/google/adk-python/commit/9dce06f9b00259ec42241df4f6638955e783a9d1))
  * Support resuming a parallel agent with multiple branches paused on tool confirmation requests ([9939e0b](https://github.com/google/adk-python/commit/9939e0b087094038b90d86c2fd35c26dd63f1157))
  * Support content union as static instruction ([cc24d61](https://github.com/google/adk-python/commit/cc24d616f80c0eba2b09239b621cf3d176f144ea))

* **[Evals]**
  * ADK cli allows developers to create an eval set and add an eval case ([ae139bb](https://github.com/google/adk-python/commit/ae139bb461c2e7c6be154b04f3f2c80919808d31))

* **[Integrations]**
  * Allow custom request and event converters in A2aAgentExecutor ([a17f3b2](https://github.com/google/adk-python/commit/a17f3b2e6d2d48c433b42e27763f3d6df80243ca))

* **[Observability]**
  * Env variable for disabling llm_request and llm_response in spans ([e50f05a](https://github.com/google/adk-python/commit/e50f05a9fc94834796876f7f112f344f788f202e))

* **[Services]**
  * Allow passing extra kwargs to create_session of VertexAiSessionService ([6a5eac0](https://github.com/google/adk-python/commit/6a5eac0bdc9adc6907a28f65a3d4d7234e863049))
  * Implement new methods in in-memory artifact service to support custom metadata, artifact versions, etc. ([5a543c0](https://github.com/google/adk-python/commit/5a543c00df2f7a66018df8a67efcf4ce44d4e0e4))
  * Add create_time and mime_type to ArtifactVersion ([2c7a342](https://github.com/google/adk-python/commit/2c7a34259395b1294319118d0f3d1b3b867b44d6))
  * Support returning all sessions when user id is none ([141318f](https://github.com/google/adk-python/commit/141318f77554ae4eb5a360bea524e98eff4a086c))

* **[Tools]**
  * Support additional headers for Google API toolset ([ed37e34](https://github.com/google/adk-python/commit/ed37e343f0c997d3ee5dc98888c5e0dbd7f2a2b6))
  * Introduces a new AgentEngineSandboxCodeExecutor class that supports executing agent-generated code using the Vertex AI Code Execution Sandbox API ([ee39a89](https://github.com/google/adk-python/commit/ee39a891106316b790621795b5cc529e89815a98))
  * Support dynamic per-request headers in MCPToolset ([6dcbb5a](https://github.com/google/adk-python/commit/6dcbb5aca642290112a7c81162b455526c15cd14))
  * Add `bypass_multi_tools_limit` option to GoogleSearchTool and VertexAiSearchTool ([9a6b850](https://github.com/google/adk-python/commit/9a6b8507f06d8367488aac653efecf665619516c), [6da7274](https://github.com/google/adk-python/commit/6da727485898137948d72906d86d78b6db6331ac))
  * Extend `ReflectAndRetryToolPlugin` to support hallucinating function calls ([f51380f](https://github.com/google/adk-python/commit/f51380f9ea4534591eda76bef27407c0aa7c3fae))
  * Add require_confirmation param for MCP tool/toolset ([78e74b5](https://github.com/google/adk-python/commit/78e74b5bf2d895d72025a44dbcf589f543514a50))

* **[UI]**
  * Granular per agent speech configuration ([409df13](https://github.com/google/adk-python/commit/409df1378f36b436139aa909fc90a9e9a0776b3a))

### Bug Fixes

* Returns dict as result from McpTool to comply with BaseTool expectations ([4df9263](https://github.com/google/adk-python/commit/4df926388b6e9ebcf517fbacf2f5532fd73b0f71))
* Fixes the identity prompt to be one line ([7d5c6b9](https://github.com/google/adk-python/commit/7d5c6b9acf0721dd230f08df919c7409eed2b7d0))
* Fix the broken langchain importing caused their 1.0.0 release ([c850da3](https://github.com/google/adk-python/commit/c850da3a07ec1441037ced1b654d8aacacd277ab))
* Fix BuiltInCodeExecutor to support visualizations ([ce3418a](https://github.com/google/adk-python/commit/ce3418a69de56570847d45f56ffe7139ab0a47aa))
* Relax runner app-name enforcement and improve agent origin inference ([dc4975d](https://github.com/google/adk-python/commit/dc4975dea9fb79ad887460659f8f397a537ee38f))
* Improve error message when adk web is run in wrong directory ([4a842c5](https://github.com/google/adk-python/commit/4a842c5a1334c3ee01406f796651299589fe12ab))
* Handle App objects in eval and graph endpoints ([0b73a69](https://github.com/google/adk-python/commit/0b73a6937bd84a41f79a9ada3fc782dca1d6fb11))
* Exclude `additionalProperties` from Gemini schemas ([307896a](https://github.com/google/adk-python/commit/307896aeceeb97efed352bc0217bae10423e5da6))
* Overall eval status should be NOT_EVALUATED if no invocations were evaluated ([9fbed0b](https://github.com/google/adk-python/commit/9fbed0b15afb94ec8c0c7ab60221bbc97e481b06))
* Create context cache only when prefix matches with previous request ([9e0b1fb](https://github.com/google/adk-python/commit/9e0b1fb62b06de7ecb79bf77d54a999167d001e1))
* Handle `App` instances returned by `agent_loader.load_agent` ([847df16](https://github.com/google/adk-python/commit/847df1638cbf1686aa43e8e094121d4e23e40245))
* Add support for file URIs in LiteLLM content conversion ([85ed500](https://github.com/google/adk-python/commit/85ed500871ff55c74d16e809ddae0d4db66cbc3a))
* Only exclude scores that are None ([998264a](https://github.com/google/adk-python/commit/998264a5b1b98ac660fcc1359fb2d25c84fa0d87))
* Better handling the A2A streaming tasks ([bddc70b](https://github.com/google/adk-python/commit/bddc70b5d004ba5304fe05bcbf6e08210f0e6131))
* Correctly populate context_id in remote_a2a_agent library ([2158b3c](https://github.com/google/adk-python/commit/2158b3c91531e9125761f211f125d9ab41a55e10))
* Remove unnecessary Aclosing ([2f4f561](https://github.com/google/adk-python/commit/2f4f5611bdb30bd5eb2fdb3a70f43d748371392f))
* Fix pickle data was truncated error in database session using MySql ([36c96ec](https://github.com/google/adk-python/commit/36c96ec5b356109b7c874c85d8bb24f0bf6c050d))

### Improvements

* Improve hint message in agent loader ([fe1fc75](https://github.com/google/adk-python/commit/fe1fc75c15a7983829bbe0b023f4b612b1e5c018))
* Fixes MCPToolset --> McpToolset in various places ([d4dc645](https://github.com/google/adk-python/commit/d4dc6454783f747120d407d0dc2cb78f53598d83))
* Add span for context caching handling and new cache creation ([a2d9f13](https://github.com/google/adk-python/commit/a2d9f13fa1d31e00ba9493fba321ca151cdd9366))
* Checks gemini version for `2 and above` for gemini-builtin tools ([0df6759](https://github.com/google/adk-python/commit/0df67599c0eb54a9a5df51af06483b40058953bf))
* Refactor and fix state management in the session service ([8b3ed05](https://github.com/google/adk-python/commit/8b3ed059c24903e8aca0a09d9d503b48af7df850))
* Update agent builder instructions and remove run command details ([89344da](https://github.com/google/adk-python/commit/89344da81364d921f778c8bbea93e1df6ad1097e))
* Clarify how to use adk built-in tool in instruction ([d22b8bf](https://github.com/google/adk-python/commit/d22b8bf8907e723f618dfd18e90dd0a5dbc9518c))
* Delegate the agent state reset logic to LoopAgent ([bb1ea74](https://github.com/google/adk-python/commit/bb1ea74924127d65d763a45b869da3d4ff4d5c5a))
* Adjust the instruction about default model ([214986e](https://github.com/google/adk-python/commit/214986ebeb53b2ef34c8aa37cd6403106de82c1b))
* Migrate invocation_context to callback_context ([e2072af](https://github.com/google/adk-python/commit/e2072af69f40474431b6749b7b9dc22fbcbc7730))
* Correct the callback signatures ([fa84bcb](https://github.com/google/adk-python/commit/fa84bcb5756773eadff486b99c9bd416b4faa9c6))
* Set default for `bypass_multi_tools_limit` to False for GoogleSearchTool and VertexAiSearchTool ([6da7274](https://github.com/google/adk-python/commit/6da727485898137948d72906d86d78b6db6331ac))
* Add more clear instruction to the doc updater agent about one PR for each recommended change ([b21d0a5](https://github.com/google/adk-python/commit/b21d0a50d610407be2f10b73a91274840ffdfe18))
* Add a guideline to avoid content deletion ([16b030b](https://github.com/google/adk-python/commit/16b030b2b25a9b0b489e47b4b148fc4d39aeffcb))
* Add a sample agent for the `ReflectAndRetryToolPlugin` ([9b8a4aa](https://github.com/google/adk-python/commit/9b8a4aad6fe65ef37885e5c3368d2799a2666534))
* Improve error message when adk web is run in wrong directory ([4a842c5](https://github.com/google/adk-python/commit/4a842c5a1334c3ee01406f796651299589fe12ab))
* Add span for context caching handling and new cache creation ([a2d9f13](https://github.com/google/adk-python/commit/a2d9f13fa1d31e00ba9493fba321ca151cdd9366))
* Disable the scheduled execution for issue triage workflow ([bae2102](https://github.com/google/adk-python/commit/bae21027d9bd7f811bed638ecce692262cb33fe5))
* Correct the callback signatures ([fa84bcb](https://github.com/google/adk-python/commit/fa84bcb5756773eadff486b99c9bd416b4faa9c6))

### Documentation

* Format README.md for samples ([0bdba30](https://github.com/google/adk-python/commit/0bdba3026345872fb907aedd1ed75e4135e58a30))
* Bump models in llms and llms-full to Gemini 2.5 ([ce46386](https://github.com/google/adk-python/commit/ce4638651f376fb6579993d8468ae57198134729))
* Update gemini_llm_connection.py - typo spelling correction ([e6e2767](https://github.com/google/adk-python/commit/e6e2767c3901a14187f5527540f318317dd6c8e3))
* Announce the first ADK Community Call in the README ([731bb90](https://github.com/google/adk-python/commit/731bb9078d01359ae770719a8f5c003680ed9f3e))

## [1.16.0](https://github.com/google/adk-python/compare/v1.15.1...v1.16.0) (2025-10-08)

### Features

* **[Core]**
  * Implementation of LLM context compaction ([e0dd06f](https://github.com/google/adk-python/commit/e0dd06ff04f9d3c2f022873ce145aaae2de02f45))
  * Support pause and resume an invocation in ADK ([ce9c39f](https://github.com/google/adk-python/commit/ce9c39f5a85ed12c22009693b5e6bc65f4641633),
    [2f1040f](https://github.com/google/adk-python/commit/2f1040f296db365080b62d6372474d90196ce0d6),
    [1ee01cc](https://github.com/google/adk-python/commit/1ee01cc05add44ce460d2cfd3726dceb0c76dceb),
    [f005414](https://github.com/google/adk-python/commit/f005414895a57befe880fd58c0d778e499a20d8e),
    [fbf7576](https://github.com/google/adk-python/commit/fbf75761bb8d89a70b32c43bbd3fa2f48b81d67c))
* **[Models]**
  * Add `citation_metadata` to `LlmResponse` ([3f28e30](https://github.com/google/adk-python/commit/3f28e30c6da192e90a8100f270274cb9a55a5348))
  * Add support for gemma model via gemini api ([2b5acb9](https://github.com/google/adk-python/commit/2b5acb98f577f5349e788bcf9910c8d7107e63b3))
* **[Tools]**
  * Add `dry_run` functionality to BigQuery `execute_sql` tool ([960eda3](https://github.com/google/adk-python/commit/960eda3d1f2f46dc93a365eb3de03dc3483fe9bb))
  * Add BigQuery analyze_contribution tool ([4bb089d](https://github.com/google/adk-python/commit/4bb089d386d4e8133e9aadbba5c42d31ff281cf6))
  * Spanner ADK toolset supports customizable template SQL and parameterized SQL ([da62700](https://github.com/google/adk-python/commit/da62700d739cb505149554962a8bcfb30f9428cc))
  * Support OAuth2 client credentials grant type ([5c6cdcd](https://github.com/google/adk-python/commit/5c6cdcd197a6780fc86d9183fa208f78c8a975d9))
  * Add `ReflectRetryToolPlugin` to reflect from errors and retry with different arguments when tool errors ([e55b894](https://github.com/google/adk-python/commit/e55b8946d6a2e01aaf018d6a79d11d13c5286152))
  * Support using `VertexAiSearchTool` built-in tool with other tools in the same agent ([4485379](https://github.com/google/adk-python/commit/4485379a049a5c84583a43c85d444ea1f1ba6f12))
  * Support using google search built-in tool with other tools in the same agent ([d3148da](https://github.com/google/adk-python/commit/d3148dacc97f0a9a39b6d7a9640f7b7b0d6f9a6c))
* **[Evals]**
  * Add HallucinationsV1 evaluation metric ([8c73d29](https://github.com/google/adk-python/commit/8c73d29c7557a75d64917ac503da519361d1d762))
  * Add Rubric based tool use metric ([c984b9e](https://github.com/google/adk-python/commit/c984b9e5529b48fff64865a8b805e7e93942ea53))
* **[UI]**
  * Adds `adk web` options for custom logo ([822efe0](https://github.com/google/adk-python/commit/822efe00659607bad2d19ec9a2d14c649fca2d8d))
* **[Observability]**
  * **otel:** Switch CloudTraceSpanExporter to telemetry.googleapis.com ([bd76b46](https://github.com/google/adk-python/commit/bd76b46ce296409d929ae69c5c43347c73e7b365))

### Bug Fixes

* Adapt to new computer use tool name in genai sdk 1.41.0 ([c6dd444](https://github.com/google/adk-python/commit/c6dd444fc947571d089b784fde3a81e17b10cf28))
* Add AuthConfig json serialization in vertex ai session service ([636def3](https://github.com/google/adk-python/commit/636def3687a85e274e3ab44d906f6d92d49e84c0))
* Added more agent instructions for doc content changes ([7459962](https://github.com/google/adk-python/commit/745996212db156878554386be34f58658482e687))
* Convert argument to pydantic model when tool declares it accepts pydantic model as argument ([571c802](https://github.com/google/adk-python/commit/571c802fbaa80b3e65f9ce2db772b9db5a13dbc4))
* Do not re-create `App` object when loader returns an `App` ([d5c46e4](https://github.com/google/adk-python/commit/d5c46e496009eb55d78637f47162df7fcaf3a7ac))
* Fix compaction logic ([3f2b457](https://github.com/google/adk-python/commit/3f2b457efd27ed47160811705e30efa6dd09d7c0))
* Fix the instruction in workflow_triage example agent ([8f3ca03](https://github.com/google/adk-python/commit/8f3ca0359e5b1306c1395770759a74aa48a52347))
* Fixes a bug that causes intermittent `pydantic` validation errors when uploading files ([e680063](https://github.com/google/adk-python/commit/e68006386fdd0da98feb9c3dce9322e44a9c914d))
* Handle A2A Task Status Update Event when streaming in remote_a2a_agent ([a5cf80b](https://github.com/google/adk-python/commit/a5cf80b952887c07bb1d56b7bdec28808edcc4a9))
* Make compactor optional in Events Compaction Config and add a default ([3f4bd67](https://github.com/google/adk-python/commit/3f4bd67b49cd60e6a2e43ccd5192efe450a6e009))
* Rename SlidingWindowCompactor to LlmEventSummarizer and refine its docstring ([f1abdb1](https://github.com/google/adk-python/commit/f1abdb1938e474564a3a76279a1a0a511f74a750))
* Rollback compaction handling from _get_contents ([84f2f41](https://github.com/google/adk-python/commit/84f2f417f77ead3748c5bbeac7f144164b9a9416))
* Set `max_output_tokens` for the agent builder ([2e2d61b](https://github.com/google/adk-python/commit/2e2d61b6fecb90cd474d6f51255678ff74b67a9b))
* Set default response modality to AUDIO in run_session ([68402bd](https://github.com/google/adk-python/commit/68402bda49083f2d56f8e8488fe13aa58b3bc18c))
* Update remote_a2a_agent to better handle streaming events and avoid duplicate responses ([8e5f361](https://github.com/google/adk-python/commit/8e5f36126498f751171bb2639c7f5a9e7dca2558))
* Update the load_artifacts tool so that the model can reliably call it for follow up questions about the same artifact ([238472d](https://github.com/google/adk-python/commit/238472d083b5aa67551bde733fc47826ff062679))
* Fix VertexAiSessionService base_url override to preserve initialized http_options ([8110e41](https://github.com/google/adk-python/commit/8110e41b36cceddb8b92ba17cffaacf701706b36), [c51ea0b](https://github.com/google/adk-python/commit/c51ea0b52e63de8e43d3dccb24f9d20987784aa5))
* Handle `App` instances returned by `agent_loader.load_agent` ([847df16](https://github.com/google/adk-python/commit/847df1638cbf1686aa43e8e094121d4e23e40245))

### Improvements

* Migrate VertexAiSessionService to use Agent Engine SDK ([90d4c19](https://github.com/google/adk-python/commit/90d4c19c5115c7af361effa8e12c248225ccf6ab))
* Migrate VertexAiMemoryBankService to use Agent Engine SDK ([d1efc84](https://github.com/google/adk-python/commit/d1efc8461e82fc31df940b701f1d1b5422214296), [97b950b](https://github.com/google/adk-python/commit/97b950b36b9c16467f0f42216b2dc8395346d7fe), [83fd045](https://github.com/google/adk-python/commit/83fd0457188decdabeae58b4e8be25daa89f2943))
* Add support for resolving $ref and $defs in OpenAPI schemas ([a239716](https://github.com/google/adk-python/commit/a239716930c72a0dbd2ccabeea69be46110ca48d))

### Documentation

* Update BigQuery samples README ([3021266](https://github.com/google/adk-python/commit/30212669ff61f3cbd6603c3dceadfbcc4cec42f8))

## [1.15.1](https://github.com/google/adk-python/compare/v1.15.0...v1.15.1) (2025-09-26)

### Bug Fixes

* Fix the deployment failure for Agent Engine ([e172811](https://github.com/google/adk-python/commit/e172811bc7173b9004572f2a2afc7024145d7713))

## [1.15.0](https://github.com/google/adk-python/compare/v1.14.1...v1.15.0) (2025-09-24)

### Features

* **[Core]**
  * Adding the ContextFilterPlugin ([a06bf27](https://github.com/google/adk-python/commit/a06bf278cbc89f521c187ed51b032d82ffdafe2d))
  * Adds plugin to save artifacts for issue [#2176](https://github.com/google/adk-python/issues/2176) ([657369c](https://github.com/google/adk-python/commit/657369cffe142ef3745cd5950d0d24a49f42f7fd))
  * Expose log probs of candidates in LlmResponse ([f7bd3c1](https://github.com/google/adk-python/commit/f7bd3c111c211e880d7c1954dd4508b952704c68))
* **[Context Caching]**
  * Support context caching ([c66245a](https://github.com/google/adk-python/commit/c66245a3b80192c16cb67ee3194f82c9a7c901e5))
    - Support explicit context caching auto creation and lifecycle management.

      Usage: `App(root_agent=..., plugins=..., context_cache_config=...)`
  * Support non-text content in static instruction ([61213ce](https://github.com/google/adk-python/commit/61213ce4d4c10f7ecaf6ddb521672059cee27942))
  * Support static instructions ([9be9cc2](https://github.com/google/adk-python/commit/9be9cc2feee92241fd2fbf9dea3a42de5a78e9ce))
    - Support static instruction that won't change, put at the beginning of
      the instruction.
      Static instruction support inline_data and file_data as contents.
      Dynamic instruction moved to the end of LlmRequest, increasing prefix
      caching matching size.

      Usage:
      `LlmAgent(model=...,static_instruction =types.Content(parts=...), ... )`
* **[Observability]**
  * Add --otel_to_cloud experimental support ([1ae0b82](https://github.com/google/adk-python/commit/1ae0b82f5602a57ad1ca975ca0b7c85003d1a28a), [b131268](https://github.com/google/adk-python/commit/b1312680f4ea9f21c3246a1d24392619643d71f5), [7870480](https://github.com/google/adk-python/commit/7870480c63bb4fc08cfb3cabc0e1f0458f0e85bd))
  * Add GenAI Instrumentation if --otel_to_cloud is enabled ([cee365a](https://github.com/google/adk-python/commit/cee365a13d0d1b1f2be046c1cc29e24a8d1fdbcc))
  * Support standard OTel env variables for exporter endpoints ([f157b2e](https://github.com/google/adk-python/commit/f157b2ee4caf4055e78f4657254e45913895f5de))
  * Temporarily disable Cloud Monitoring integration in --otel_to_cloud ([3b80337](https://github.com/google/adk-python/commit/3b80337faf427460e4743e25dbb92578f823513f))
* **[Services]**
  * Add endpoint to generate memory from session ([2595824](https://github.com/google/adk-python/commit/25958242db890b4d2aac8612f7f7cfbb561727fa))
* **[Tools]**
  * Add Google Maps Grounding Tool to ADK ([6b49391](https://github.com/google/adk-python/commit/6b493915469ecb42068e24818ab547b0856e4709))
  * **MCP:** Initialize tool_name_prefix in MCPToolset ([86dea5b](https://github.com/google/adk-python/commit/86dea5b53ac305367283b7e353b60d0f4515be3b))
* **[Evals]**
  * Data model for storing App Details and data model for steps ([01923a9](https://github.com/google/adk-python/commit/01923a9227895906ca8ae32712d65b178e2cd7d5))
  * Adds Rubric based final response evaluator ([5a485b0](https://github.com/google/adk-python/commit/5a485b01cd64cb49735e13ebd5e7fa3da02cd85f))
  * Populate AppDetails to each Invocation ([d486795](https://github.com/google/adk-python/commit/d48679582de91050ca9c5106402319be9a8ae7e8))
* **[Samples]**
  * Make the bigquery sample agent run with ADC out-of-the-box ([10cf377](https://github.com/google/adk-python/commit/10cf37749417856e394e62896231e41b13420f18))

### Bug Fixes

* Close runners after running eval ([86ee6e3](https://github.com/google/adk-python/commit/86ee6e3fa3690148d60358fc3dacb0e0ab40942b))
* Filter out thought parts when saving agent output to state ([632bf8b](https://github.com/google/adk-python/commit/632bf8b0bcf18ff4e4505e4e5f4c626510f366a2))
* Ignore empty function chunk in LiteLlm streaming response ([8a92fd1](https://github.com/google/adk-python/commit/8a92fd18b600da596c22fd80c6148511a136dfd0))
* Introduces a `raw_mcp_tool` method in `McpTool` to provide direct access to the underlying MCP tool ([6158075](https://github.com/google/adk-python/commit/6158075a657f8fe0835679e509face6191905403))
* Make a copy of the `columns` instead of modifying it in place ([aef1ee9](https://github.com/google/adk-python/commit/aef1ee97a55a310f3959d475b8d7d6bc3915ae48))
* Prevent escaping of Latin characters in LLM response ([c9ea80a](https://github.com/google/adk-python/commit/c9ea80af28e586c9cc1f643b365cdba82f80c700))
* Retain the consumers and transport registry when recreating the ClientFactory in remote_a2a_agent.py ([6bd33e1](https://github.com/google/adk-python/commit/6bd33e1be36f741a6ed0514197550f9f336262ed))
* Remove unsupported 'type': 'unknown' in test_common.py for fastapi 0.117.1 ([3745221](https://github.com/google/adk-python/commit/374522197fa6843f786bfd12d17ce0fc20461dfd))

### Documentation

* Correct the documentation of `after_agent_callback` ([b9735b2](https://github.com/google/adk-python/commit/b9735b2193267645781b268231d63c23c6fec654))

## [1.14.1](https://github.com/google/adk-python/compare/v1.14.0...v1.14.1) (2025-09-12)

### Bug Fixes

* Fix logging issues with RemoteA2aAgent [0c1f1fa](https://github.com/google/adk-python/commit/0c1f1fadeb5a6357af9cad0eff5d5e7103fc88b0)

## [1.14.0](https://github.com/google/adk-python/compare/v1.13.0...v1.14.0) (2025-09-10)

### Features

* **[A2A]**
  * Allow users to pass their own agent card to to_a2a method [a1679da](https://github.com/google/adk-python/commit/a1679dae3fef70f1231afba3e97d45b59c314ae3)
  * Allow custom part converters in A2A classes [b05fef9](https://github.com/google/adk-python/commit/b05fef9ba71f95ab2658eb4eb5608c141d49f82f)
* **[Tools]**
  * Allow setting agent/application name and compute project for BigQuery tools [11a2ffe](https://github.com/google/adk-python/commit/11a2ffe35adbae977b49ceccf0e76e20c6dc90b6)
  * Add BigQuery forecast tool [0935a40](https://github.com/google/adk-python/commit/0935a40011a3276ee7f7fa3b91678b4d63f22ba5)
  * Add GkeCodeExecutor for sandboxed code execution on GKE [72ff9c6](https://github.com/google/adk-python/commit/72ff9c64a291aebb50b07446378f375e58882c4e)
  * Add a tool confirmation flow that can guard tool execution with explicit confirmation and custom input [a17bcbb](https://github.com/google/adk-python/commit/a17bcbb2aa0f5c6aca460db96ed1cb7dd86fef84)
  * Add audience and prompt as configurable for OAuth flows [edda922](https://github.com/google/adk-python/commit/edda922791f15ac37830ed95ebf76b9f836d9db4)
  * Allow user specify embedding model for file retrieval [67f23df](https://github.com/google/adk-python/commit/67f23df25ad47aff3cb36d0fc9ce2c9b97bde09b)
* **[Core]**
  * Allow all possible values for `agent_class` field in all Agent Configs [3bc2d77](https://github.com/google/adk-python/commit/3bc2d77b4d180e9c42b30d4d1ce580aa75abe501)
  * Allow agent loader to load built-in agents from special directories in adk folder [578fad7](https://github.com/google/adk-python/commit/578fad7034a7b369a490ad0afa4dd2820463c22d)
  * Upgrade ADK runner to use App in addition to root_agent [4df79dd](https://github.com/google/adk-python/commit/4df79dd5c92d96096d031b26470458d0bca79a79)
  * Allow inject artifact into instructions [bb4cfde](https://github.com/google/adk-python/commit/bb4cfdec12370955d4038d6d8c86e04691f2308e)
* **[Misc]** Create an initial ADK release analyzer agent to find the doc updates needed between releases [e3422c6](https://github.com/google/adk-python/commit/e3422c616d18ec3850454ee83f2ef286198543ec)

### Bug Fixes

* Add a NOTE to agent transfer instructions listing available agents [43eec82](https://github.com/google/adk-python/commit/43eec82f8444c19455089655ee288200ec966577)
* Fix pagination of list_sessions in VertexAiSessionService [e63fe0c](https://github.com/google/adk-python/commit/e63fe0c0eb73ac6e22d975387dd2df3d2ba3f521)
* Fix AttributeError and indentation in parameter processing of LiteLlm [1e23652](https://github.com/google/adk-python/commit/1e23652968164c5fdfa5564e966e78799237d94b)
* Allow AgentTool to inherit/use plugins from its invocation context when running [1979dcf](https://github.com/google/adk-python/commit/1979dcf496be3fb75fa2063fc96f480bedeb5de2)
* Enforce foreign key constraint for SQLite DB [0c87907](https://github.com/google/adk-python/commit/0c87907bcb2e5687a4ad08bab450fc888a5b5233)
* Add back installing requirements.txt to Dockerfile template for cloud run [8e43f0d](https://github.com/google/adk-python/commit/8e43f0dd8321ea31d6ad970ad4402feb48cdbd3d)
* Only process the auth responses in the last event with content (if applicable i.e. it's authored by user) [3b922a2](https://github.com/google/adk-python/commit/3b922a2f6da373b0de78b022db5d5bcb5453379f)
* Extract a utility for aggregating partial streaming responses and emitting LlmResponses for them as needed [7975e8e](https://github.com/google/adk-python/commit/7975e8e1961c8e375e2af3506ea546580ff7e45d)
* Support saving text artifacts in GCS artifact service [cecf7e8](https://github.com/google/adk-python/commit/cecf7e805d19d20e940319a6e16bfc9015ead202)
* Fixes `thought` handling in contents.py and refactors its unit tests [a30851e](https://github.com/google/adk-python/commit/a30851ee16114103dca7b9736e79cb31e82ee4d8)
* Fixes the `thought` field handling in _planning.py [fe8b37b](https://github.com/google/adk-python/commit/fe8b37b0d3046a9c0dd90e8ddca2940c28d1a93f)
* Pass state_delta to runner in /run endpoint [a3410fa](https://github.com/google/adk-python/commit/a3410fab7b25cc0e9c5908e23a087b501466df76)
* Fix discussion answering github action workflow to escape the quote in the discussion content JSON [43c9681](https://github.com/google/adk-python/commit/43c96811da891a5b0c9cf1be525665e65f346a13)
* Send full MIME types for image/video/pdf in get_content [e45c3be](https://github.com/google/adk-python/commit/e45c3be23895b5ec68908ad9ee19bd622dcbd003)
* Fix flaky unit tests: tests/unittests/flows/llm_flows/test_functions_simple.py [b92b288](https://github.com/google/adk-python/commit/b92b288c978a9b3d1a76c8bcb96cc8f439ce610b)
* Make UT of a2a consistent about how tests should be skipped when python version < 3.10 [98b0426](https://github.com/google/adk-python/commit/98b0426cd2dc5e28014ead22b22dbf50d42d0a9a)

### Improvements

* Update contribution guide [8174a29](https://github.com/google/adk-python/commit/8174a29c6db9fd22a5a563f3088bd538b90e9a50)
* Skip PR triage for already triaged or Google-contributor PRs [78eea1a](https://github.com/google/adk-python/commit/78eea1aa550790097a1005237acaec56309cd61e)
* Avoid mutable default arguments in `local_eval_service` and `runners` [64f11a6](https://github.com/google/adk-python/commit/64f11a6a67e7042768270c5587e87528c358bd06)
* Avoid mutable default arguments in `local_eval_service` and `runners` [5b465fd](https://github.com/google/adk-python/commit/5b465fd71b601a2a1ab95a74f7c9ddafe09085e5)
* Reorder dependencies in `pyproject.toml` [ca5f7f1](https://github.com/google/adk-python/commit/ca5f7f1ff0afb2b3c2457fb9efdf029dcf7494b7)
* Follow pydantic convention to make field_validator a public method [1448406](https://github.com/google/adk-python/commit/14484065c64396cebc4a1dde84d6b8b51439b990)
* Update comment to clarify `after_run` callbacks [7720616](https://github.com/google/adk-python/commit/7720616c5f1dc302f019c348a6dfa70d1cf0b135)
* Tune instructions to not ask root directory if it's already provided in the context [25df6c2](https://github.com/google/adk-python/commit/25df6c22d5942ead3a329f90ed2c10b374051ae6)
* Load discussion data from event content to avoid additional GraphQL API call [a503a0c](https://github.com/google/adk-python/commit/a503a0c807e50ec9dde7d5095f8e020861d1375d)
* Refactor discussion answering agent to merge answer_discussions.py into main.py [408d3df](https://github.com/google/adk-python/commit/408d3dfeb1475da343a15ae13e9b128985460a5d)
* Add community repo dependency group to pyproject toml [7b077ac](https://github.com/google/adk-python/commit/7b077ac3517f2b88d1bc4b732815ca766c791168)
* Add warning for using Gemini models via LiteLLM [9291daa](https://github.com/google/adk-python/commit/9291daaa8e399ca052f5a52dbb600d719dcc9fa8)

### Documentation

* Update root_agent description for clarity [467df1a](https://github.com/google/adk-python/commit/467df1a36f3ded1a0e324defcd94c557871c9190)
* Update the ask_data_insights docstring [aad1533](https://github.com/google/adk-python/commit/aad153322e54cc39c97e3e0bc71cbed72bcab477)
* Add contributing Spanner tools RAG agent sample [fcd748e](https://github.com/google/adk-python/commit/fcd748e17f4e0e7a3146716816c579f2ee973e6b)

### Tests

* Add functional telemetry tests [bc6b546](https://github.com/google/adk-python/commit/bc6b5462a76ee1cd718c75360daac94373d7c071)
* Add unit tests for the `App` class and improve `Runner` initialization tests [fc90ce9](https://github.com/google/adk-python/commit/fc90ce968f114f84b14829f8117797a4c256d710)

### Chores

* Use lazy % formatting in logging functions to fix pylint warnings [b431072](https://github.com/google/adk-python/commit/b4310727d90421a81a8afc47e3c344646ee7aee8)
* Update release cadence in README [decc19b](https://github.com/google/adk-python/commit/decc19b188fbf097995824f9ad7b7be1263b6338)
* Add `custom_metadata` to DatabaseSessionService [fb009d8](https://github.com/google/adk-python/commit/fb009d8ea672bbbef4753e4cd25229dbebd0ff8d)
* Update create_session endpoint to use Request message as post body [219815d](https://github.com/google/adk-python/commit/219815d2d7f45ac0cff28265f23fbf4f4e77163f)

## 1.13.0 (2025-08-27)

### Features

* [Tools] Add the ask_data_insights tool for natural language queries on BigQuery data [47b88d2](https://github.com/google/adk-python/commit/47b88d2b06d247a698915ebf74564dbb5d81153e)

### Bug Fixes

* Add the missing `from_config` class method in BaseToolset [2dd432c](https://github.com/google/adk-python/commit/2dd432cc1fe265a79986a28e2afb59ee2c83abb3)
* Change LlmResponse to use Content for transcriptions [3b997a0](https://github.com/google/adk-python/commit/3b997a0a07d1a2915bc64d64355f4dbabb7e0ba0)
* AgentTool returns last content, instead of the content in the last event [bcf0dda](https://github.com/google/adk-python/commit/bcf0dda8bcc221974098f3077007c9e84c63021a)
* Fix adk deploy docker file permission [ad81aa5](https://github.com/google/adk-python/commit/ad81aa54de1f38df580915b7f47834ea8e5f1004)
* Updating BaseAgent.clone() and LlmAgent.clone() to properly clone fields that are lists [29bb75f](https://github.com/google/adk-python/commit/29bb75f975fe0c9c9d9a7e534a9c20158e1cbe1e)
* Make tool description for bigquery `execute_sql` for various write modes self contained [167182b](https://github.com/google/adk-python/commit/167182be0163117f814c70f453d5b2e19bf474df)
* Set invocation_id and branch for event generated when both output_schema and tools are used [3f3aa7b](https://github.com/google/adk-python/commit/3f3aa7b32d63cae5750d71bc586c088427c979ea)
* Rework parallel_agent.py to always aclose async generators [826f554](https://github.com/google/adk-python/commit/826f5547890dc02e707be33a3d6a58b527dac223)
* Add table metadata info into Spanner tool `get_table_schema` and fix the key usage info [81a53b5](https://github.com/google/adk-python/commit/81a53b53d6336011187a50ae8f1544de9b2764a8)
* Fix Spanner DatabaseSessionService support [54ed079](https://github.com/google/adk-python/commit/54ed0791005350542708eb2c38f32ce8b92356bc)
* Add support for required params [c144b53](https://github.com/google/adk-python/commit/c144b5347cc459496d4fd41e0c63715ffffb4952)
* Replaced hard coded value for user_id to the value from the tool context from parent agent. [0b89f18](https://github.com/google/adk-python/commit/0b89f1882dccc1acd0ee109832053edecec04850)

### Improvements

* Allow user to specify protocol for A2A RPC URL in to_a2a utility [157f731](https://github.com/google/adk-python/commit/157f73181d123b0fddc34205dc74434fcbc43b2a)
* Passthrough extra args for `adk deploy cloud_run` as Cloud Run args [6806dea](https://github.com/google/adk-python/commit/6806deaf8811eb7f02ed958648886323aba16adb)
* Renames MCPTool and MCPToolset to McpTool and McpToolset [4c70606](https://github.com/google/adk-python/commit/4c7060612967253dae824a14c5c3f853a547469b)
* Ignore hidden files in autoformat.sh [0eb65c0](https://github.com/google/adk-python/commit/0eb65c07d52f71cf555f0c32dc34b2e4ac8cf2a2)

### Documentation

* Clean up docs in sample [a360bc2](https://github.com/google/adk-python/commit/a360bc25429bf4bef6a80da59afe30d6933a844b)
* Fixes root_agent.yaml in tool_mcp_stdio_notion_config for Agent Config sample and add README.md [2c088ac](https://github.com/google/adk-python/commit/2c088acc9b34f030537b02b45a4afd458445d15b)
* Add What's new section to README.md [ccab076](https://github.com/google/adk-python/commit/ccab076aceff917591eb3a3cc89a9f85226b832a)

## 1.12.0 (2025-08-21)

### Features

**[Agent Config]** 🌟 **NEW FEATURE**: Support using config file (YAML) to author agents in addition to python code. See the [documentation](https://google.github.io/adk-docs/agents/config/) for details.
* [Agent Config] Support deploying config agent to Agent Engine in CLI ([b3b7003](https://github.com/google/adk-python/commit/b3b70035c432670a5f0b5cdd1e9467f43b80495c))
* [Tools] Add a dedicated Bigtable toolset to provide an easier, integrated way to interact
with Bigtable for building AI Agent applications(experimental feature) ([a953807](https://github.com/google/adk-python/commit/a953807cce341425ba23e3f0a85eae58d6b0630f))
* [Tools] Support custom tool_name_prefix in auto-generated GoogleApiToolset ([a2832d5](https://github.com/google/adk-python/commit/a2832d5ac7ba5264ee91f6d5a6a0058cfe4c9e8a)) See [oauth_calendar_agent](https://github.com/google/adk-python/tree/main/contributing/samples/oauth_calendar_agent) as an example.
* [CLI] Add `build_image` option for `adk deploy cloud_run` CLI ([c843503](https://github.com/google/adk-python/commit/c84350345af0ea6a232e0818b20c4262b228b103))
* [Services] Add setdefault method to the ADK State object ([77ed1f5](https://github.com/google/adk-python/commit/77ed1f5f15ed3f009547ed0e20f86d949de12ec2))


### Bug Fixes

* Lazy load VertexAiCodeExecutor and ContainerCodeExecutor ([018db79](https://github.com/google/adk-python/commit/018db79d1354f93b8328abb8416f63070b25f9f1))
* Fix the path for agent card in A2A demo ([fa64545](https://github.com/google/adk-python/commit/fa64545a9de216312a69f93126cfd37f1016c14b))
* Fix the path for agent card in A2A demo ([a117cf0](https://github.com/google/adk-python/commit/a117cf0af335c5e316ae9d61336a433052316462))
* litellm-test due to breaking change in dep library of extension extra ([004a0a0](https://github.com/google/adk-python/commit/004a0a0f2d9a4f7ae6bff42a7cad96c11a99acaf))
* Using base event's invocation id when merge multiple function response event ([279e4fe](https://github.com/google/adk-python/commit/279e4fedd0b1c0d1499c0f9a4454357af7da490e))
* Avoid crash when there is no candidates_token_count, which is Optional ([22f34e9](https://github.com/google/adk-python/commit/22f34e9d2c552fbcfa15a672ef6ff0c36fa32619))
* Fix the packaging version comparison logic in adk cli ([a2b7909](https://github.com/google/adk-python/commit/a2b7909fc36e7786a721f28e2bf75a1e86ad230d))
* Add Spanner admin scope to Spanner tool default OAuth scopes ([b66054d](https://github.com/google/adk-python/commit/b66054dd0d8c5b3d6f6ad58ac1fbd8128d1da614))
* Fixes SequentialAgent.config_type type hint ([8a9a271](https://github.com/google/adk-python/commit/8a9a271141678996c9b84b8c55d4b539d011391c))
* Fixes the host in the ansi bracket of adk web ([cd357bf](https://github.com/google/adk-python/commit/cd357bf5aeb01f1a6ae2a72349a73700ca9f1ed2))
* Add spanner tool name prefix ([a27927d](https://github.com/google/adk-python/commit/a27927dc8197c391c80acb8b2c23d610fba2f887))

### Improvements

* Support `ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS` as environment variable to suppress experimental warnings ([4afc9b2](https://github.com/google/adk-python/commit/4afc9b2f33d63381583cea328f97c02213611529))
* Uses pydantic `Field` for Agent configs, so that the generated AgentConfig.json json schema can carry field description ([5b999ed](https://github.com/google/adk-python/commit/5b999ed6fd23a0fc1da56ccff4c09621f433846b))
* Update `openai` dependency version, based on correct OPENAI release ([bb8ebd1](https://github.com/google/adk-python/commit/bb8ebd15f90768b518cd0e21a59b269e30d6d944))
* Add the missing license header for core_callback_config init file ([f8fd6a4](https://github.com/google/adk-python/commit/f8fd6a4f09ab520b8ecdbd8f9fe48228dbff7ebe))
* Creates yaml_utils.py in utils to allow adk dump yaml in the same style ([1fd58cb](https://github.com/google/adk-python/commit/1fd58cb3633992cd88fa7e09ca6eda0f9b34236f))
* Return explicit None type for DELETE endpoints ([f03f167](https://github.com/google/adk-python/commit/f03f1677790c0a9e59b6ba6f46010d0b7b64be50))
* Add _config suffix to all yaml-based agent examples ([43f302c](https://github.com/google/adk-python/commit/43f302ce1ab53077ee8f1486d5294540678921e6))
* Rename run related method and request to align with the conventions ([ecaa7b4](https://github.com/google/adk-python/commit/ecaa7b4c9847b478c7cdc37185b1525f733bb403))
* Update models in samples/ folder to be gemini 2.0+ ([6c217ba](https://github.com/google/adk-python/commit/6c217bad828edf62b41ec06b168f8a6cb7ece2ed))
* Remove the "one commit" requirement from the contributing guide ([c32cb6e](https://github.com/google/adk-python/commit/c32cb6eef9ce320ea5a1f3845fc57b83762c237e))
* Bump version to 1.11.0 ([8005270](https://github.com/google/adk-python/commit/80052700f6cee947322080ae6c415d3a428b6c91))

### Documentation

* Add contributing bigtable sample ([fef5318](https://github.com/google/adk-python/commit/fef5318a22f3dcaadb7ecb858725eb61a0350140))
* Fix core_callback example ([ba6e85e](https://github.com/google/adk-python/commit/ba6e85eb3fb06f58ce9077574eac193298e18bea))
* Adds a minimal sample to demo how to use Agent Config to create a multi-agent setup ([1328e6e](https://github.com/google/adk-python/commit/1328e6ef62e9e6260048c0078579edb85a0440bc))


## [1.11.0](https://github.com/google/adk-python/compare/v1.10.0...v1.11.0) (2025-08-14)

### Features

* [Tools] Support adding prefix to tool names returned by toolset ([ebd726f](https://github.com/google/adk-python/commit/ebd726f1f5e0a76f383192cace4a80a83204325b))
* [Eval] Expose `print_detailed_results` param to `AgentEvaluator.evaluate` ([7e08808](https://github.com/google/adk-python/commit/7e0880869b340e9a5e0d68d6936219e64ab41212))
* [Tools] Add Spanner toolset (breaking change to BigQueryTool, consolidating into generic GoogleTool) ([1fc8d20](https://github.com/google/adk-python/commit/1fc8d20ae88451b7ed764aa86c17c3cdfaffa1cf))
* [Core] Support both output_schema and tools at the same time in LlmAgent([sample](https://github.com/google/adk-python/tree/main/contributing/samples/output_schema_with_tools)) ([af63567](https://github.com/google/adk-python/commit/af635674b5d3c128cf21737056e091646283aeb7))

### Bug Fixes

* A2A RPC URL got overridden by host and port param of adk api server ([52284b1](https://github.com/google/adk-python/commit/52284b1bae561e0d6c93c9d3240a09f210551b97))
* Aclose all async generators to fix OTel tracing context ([a30c63c](https://github.com/google/adk-python/commit/a30c63c5933a770b960b08a6e2f8bf13eece8a22))
* Use PreciseTimestamp for create and update time in database session service to improve precision ([585141e](https://github.com/google/adk-python/commit/585141e0b7dda20abb024c7164073862c8eea7ae))
* Ignore AsyncGenerator return types in function declarations ([e2518dc](https://github.com/google/adk-python/commit/e2518dc371fe77d7b30328d8d6f5f864176edeac))
* Make all subclass of BaseToolset to call parent constructor ([8c65967](https://github.com/google/adk-python/commit/8c65967cdc2dc79fa925ff49a2a8d67c2a248fa9))
* Path parameter extraction for complex Google API endpoints ([54680ed](https://github.com/google/adk-python/commit/54680edf3cac7477c281680ec988c0a207c0915d))
* Docstring concatenation in 3.13 ([88f759a](https://github.com/google/adk-python/commit/88f759a941c95beef0571f36f8e7a34f27971ba8))
* Lazy load retrieval tools and prompt users to install extensions if import failed ([9478a31](https://github.com/google/adk-python/commit/9478a31bf2257f0b668ae7eb91a10863e87c7bed))
* Incorrect logic in LlmRequest.append_tools and make BaseTool to call it ([b4ce3b1](https://github.com/google/adk-python/commit/b4ce3b12d109dd0386f4985fc4b27d5b93787532))
* Creates an InMemoryMemoryService within the EvaluationGenerator ([e4d54b6](https://github.com/google/adk-python/commit/e4d54b66b38ed334ca92c3bb1a83aca01b19e490))
* Uncomment OTel tracing in base_llm_flow.py ([9cfe433](https://github.com/google/adk-python/commit/9cfe43334ae50f814fed663cca7cbe330e663b8c))

### Improvements

* Added upper version bounds to dependencies in "pyproject.toml" ([a74d334](https://github.com/google/adk-python/commit/a74d3344bc19e587c5e9f55f3c90fa9d22c478d8))
* Update python-version in .github/workflows/python-unit-tests.yml to \["3.9", "3.10", "3.11", "3.12", "3.13"] ([ddf2e21](https://github.com/google/adk-python/commit/ddf2e2194b49667c8e91b4a6afde694474674250))
* Update comment to reference "Streamable HTTP Client" ([c52f956](https://github.com/google/adk-python/commit/c52f9564330f0c00d82338cc58df28cb22400b6f))
* Remove logging that contains full event data from DatabaseSessionService ([bb3735c](https://github.com/google/adk-python/commit/bb3735c9cab1baa1af2cc22981af3b3984ddfe15))
* Add the missing env variables in discussion_answering.yml ([a09a5e6](https://github.com/google/adk-python/commit/a09a5e67aa95cf71b51732ab445232dc4815d83d))
* Add Gemini API docs as a new datastore for the ADK Answering Agent ([5fba196](https://github.com/google/adk-python/commit/5fba1963c31eec512558325c480812ccb919a7bb))
* Add the missing license header for some sample agents' files ([7d2cb65](https://github.com/google/adk-python/commit/7d2cb654f0d64728741b5de733e572c44c8a5b04))
* Add docstring to clarify the behavior of preload memory tool ([88114d7](https://github.com/google/adk-python/commit/88114d7c739ca6a1b9bd19d40ed7160e53054a89))
* Add experimental messages for a2a related API ([d0b3b5d](https://github.com/google/adk-python/commit/d0b3b5d857d8105c689bd64204e367102a67eded))
* Fixes generate_image sample ([d674178](https://github.com/google/adk-python/commit/d674178a0535be3769edbf6af5a3d8cd3d47fcd2))
* Make all FastAPI endpoints async ([7f12387](https://github.com/google/adk-python/commit/7f12387eb19b9335a64b80df00609c3c765480e7))
* Group FastAPI endpoints with tags ([c323de5](https://github.com/google/adk-python/commit/c323de5c692223e55372c3797e62d4752835774d))
* Allow implementations to skip defining a close method on Toolset ([944e39e](https://github.com/google/adk-python/commit/944e39ec2a7c9ad7f20c08fd66bf544de94a23d7))
* Add sample agent to test support of output_schema and tools at the same time for gemini model ([f2005a2](https://github.com/google/adk-python/commit/f2005a20267e1ee8581cb79c37aa55dc8e18c0ea))
* Add GitHub workflow config for uploading ADK docs to knowledge store ([5900273](https://github.com/google/adk-python/commit/59002734559d49a46940db9822b9c5f490220a8c))
* Update ADK Answering agent to reference doc site instead of adk-docs repo ([b5a8bad](https://github.com/google/adk-python/commit/b5a8bad170e271b475385dac440c7983ed207df8))

### Documentation

* Fixes tool_functions, which is a config-based sample for using tools ([c5af44c](https://github.com/google/adk-python/commit/c5af44cfc0224e2f07ddc7a649a8561e7141fcdc))
* Add workflow_triage sample for multi-agent request orchestration ([e295feb](https://github.com/google/adk-python/commit/e295feb4c67cbe8ac4425d9ae230210840378b2e))
* Add examples for config agents ([d87feb8](https://github.com/google/adk-python/commit/d87feb8ddb6a5e402c63bd3c35625160eb94e132))
* Adds pypi badge to README.md ([dc26aad](https://github.com/google/adk-python/commit/dc26aad663b6ae72223cfec9b91eaf73a636402d))
* Update StreamableHTTPConnectionParams docstring to remove SSE references ([8f937b5](https://github.com/google/adk-python/commit/8f937b517548a1ce0569f9698ea55c0a130ef221))

## [1.10.0](https://github.com/google/adk-python/compare/v1.9.0...v1.10.0) (2025-08-07)

### Features

* [Live] Implement Live Session Resumption ([71fbc92](https://github.com/google/adk-python/commit/71fbc9275b3d74700ec410cb4155ba0cb18580b7))
* [Tool] Support parallel execution of parallel function calls ([57cd41f](https://github.com/google/adk-python/commit/57cd41f424b469fb834bb8f2777b5f7be9aa6cdf))
* [Models] Allow max tokens to be customizable in Claude ([7556ebc](https://github.com/google/adk-python/commit/7556ebc76abd3c776922c2803aed831661cf7f82))
* [Tool] Create enterprise_web_search_tool as a tool instance ([0e28d64](https://github.com/google/adk-python/commit/0e28d64712e481cfd3b964be0166f529657024f6))

### Bug Fixes

* Fix shared default plugin manager and cost manager instances among multiple invocations ([423542a](https://github.com/google/adk-python/commit/423542a43fb8316195e9f79d97f87593751bebd3))
* Correct the type annotation in anthropic_llm implementation ([97318bc](https://github.com/google/adk-python/commit/97318bcd199acdacadfe8664da3fbfc3c806cdd2))
* Fix adk deploy cloud_run cli, which was broken in v1.9.0 ([e41dbcc](https://github.com/google/adk-python/commit/e41dbccf7f610e249108f9321f60f71fe2cc10f4))
* Remove thoughts from contents in llm requests from history contents ([d620bcb](https://github.com/google/adk-python/commit/d620bcb384d3068228ea2059fb70274e68e69682))
* Annotate response type as None for transfer_to_agent tool ([86a4487](https://github.com/google/adk-python/commit/86a44873e9b2dfc7e62fa31a9ac3be57c0bbff7b))
* Fix incompatible a2a sdk changes ([faadef1](https://github.com/google/adk-python/commit/faadef167ee8e4dd1faf4da5685a577c3155556e))
* Fix adk cli options and method parameters mismatching ([8ef2177](https://github.com/google/adk-python/commit/8ef2177658fbfc74b1a74b0c3ea8150bae866796))

### Improvements

* Add GitHub workflow config for the ADK Answering agent ([8dc0c94](https://github.com/google/adk-python/commit/8dc0c949afb9024738ff7ac1b2c19282175c3200))
* Import AGENT_CARD_WELL_KNOWN_PATH from adk instead of from a2a directly ([37dae9b](https://github.com/google/adk-python/commit/37dae9b631db5060770b66fce0e25cf0ffb56948))
* Make `LlmRequest.LiveConnectConfig` field default to a factory ([74589a1](https://github.com/google/adk-python/commit/74589a1db7df65e319d1ad2f0676ee0cf5d6ec1d))
* Update the prompt to make the ADK Answering Agent more objective ([2833030](https://github.com/google/adk-python/commit/283303032a174d51b8d72f14df83c794d66cb605))
* Add sample agent for testing parallel functions execution ([90b9193](https://github.com/google/adk-python/commit/90b9193a20499b8dd7f57d119cda4c534fcfda10))
* Hide the ask_data_insights tool until the API is publicly available ([bead607](https://github.com/google/adk-python/commit/bead607364be7ac8109357c9d3076d9b345e9e8a))
* Change `LlmRequest.config`'s default value to be `types.GenerateContentConfig()` ([041f04e](https://github.com/google/adk-python/commit/041f04e89cee30532facccce4900d10f1b8c69ce))
* Prevent triggering of _load_from_yaml_config in AgentLoader ([db975df](https://github.com/google/adk-python/commit/db975dfe2a09a6d056d02bc03c1247ac10f6da7d))

### Documentation

* Fix typos ([16a15c8](https://github.com/google/adk-python/commit/16a15c8709b47c9bebe7cffe888e8e7e48ec605a))


## [1.9.0](https://github.com/google/adk-python/compare/v1.8.0...v1.9.0) (2025-07-31)


### Features

* [CLI] Add `-v`, `--verbose` flag to enable DEBUG logging as a shortcut for `--log_level DEBUG` ([3be0882](https://github.com/google/adk-python/commit/3be0882c63bf9b185c34bcd17e03769b39f0e1c5))
* [CLI] Add a CLI option to update an agent engine instance ([206a132](https://github.com/google/adk-python/commit/206a13271e5f1bb0bb8114b3bb82f6ec3f030cd7))
* [CLI] Modularize fast_api.py to allow simpler construction of API Server ([bfc203a](https://github.com/google/adk-python/commit/bfc203a92fdfbc4abaf776e76dca50e7ca59127b), [dfc25c1](https://github.com/google/adk-python/commit/dfc25c17a98aaad81e1e2f140db83d17cd78f393), [e176f03](https://github.com/google/adk-python/commit/e176f03e8fe13049187abd0f14e63afca9ccff01))
* [CLI] Refactor AgentLoader into base class and add InMemory impl alongside existing filesystem impl ([bda3df2](https://github.com/google/adk-python/commit/bda3df24802d0456711a5cd05544aea54a13398d))
* [CLI] Respect the .ae_ignore file when deploying to agent engine ([f29ab5d](https://github.com/google/adk-python/commit/f29ab5db0563a343d6b8b437a12557c89b7fc98b))
* [Core] Add new callbacks to handle tool and model errors ([00afaaf](https://github.com/google/adk-python/commit/00afaaf2fc18fba85709754fb1037bb47f647243))
* [Core] Add sample plugin for logging ([20537e8](https://github.com/google/adk-python/commit/20537e8bfa31220d07662dad731b4432799e1802))
* [Core] Expose Gemini RetryOptions to client ([1639298](https://github.com/google/adk-python/commit/16392984c51b02999200bd4f1d6781d5ec9054de))
* [Evals] Added an Fast API new endpoint to serve eval metric info ([c69dcf8](https://github.com/google/adk-python/commit/c69dcf87795c4fa2ad280b804c9b0bd3fa9bf06f))
* [Evals] Refactored AgentEvaluator and updated it to use LocalEvalService ([1355bd6](https://github.com/google/adk-python/commit/1355bd643ba8f7fd63bcd6a7284cc48e325d138e))


### Bug Fixes

* Add absolutize_imports option when deploying to agent engine ([fbe6a7b](https://github.com/google/adk-python/commit/fbe6a7b8d3a431a1d1400702fa534c3180741eb3))
* Add space to allow adk deploy cloud_run --a2a ([70c4616](https://github.com/google/adk-python/commit/70c461686ec2c60fcbaa384a3f1ea2528646abba))
* Copy the original function call args before passing it to callback or tools to avoid being modified ([3432b22](https://github.com/google/adk-python/commit/3432b221727b52af2682d5bf3534d533a50325ef))
* Eval module not found exception string ([7206e0a](https://github.com/google/adk-python/commit/7206e0a0eb546a66d47fb411f3fa813301c56f42))
* Fix incorrect token count mapping in telemetry ([c8f8b4a](https://github.com/google/adk-python/commit/c8f8b4a20a886a17ce29abd1cfac2858858f907d))
* Import cli's artifact dependencies directly ([282d67f](https://github.com/google/adk-python/commit/282d67f253935af56fae32428124a385f812c67d))
* Keep existing header values while merging tracking headers for `llm_request.config.http_options` in `Gemini.generate_content_async` ([6191412](https://github.com/google/adk-python/commit/6191412b07c3b5b5a58cf7714e475f63e89be847))
* Merge tracking headers even when `llm_request.config.http_options` is not set in `Gemini.generate_content_async` ([ec8dd57](https://github.com/google/adk-python/commit/ec8dd5721aa151cfc033cc3aad4733df002ae9cb))
* Restore bigquery sample agent to runnable form ([16e8419](https://github.com/google/adk-python/commit/16e8419e32b54298f782ba56827e5139effd8780))
* Return session state in list_session API endpoint ([314d6a4](https://github.com/google/adk-python/commit/314d6a4f95c6d37c7da3afbc7253570564623322))
* Runner was expecting Event object instead of Content object when using early exist feature ([bf72426](https://github.com/google/adk-python/commit/bf72426af2bfd5c2e21c410005842e48b773deb3))
* Unable to acquire impersonated credentials ([9db5d9a](https://github.com/google/adk-python/commit/9db5d9a3e87d363c1bac0f3d8e45e42bd5380d3e))
* Update `agent_card_builder` to follow grammar rules ([9c0721b](https://github.com/google/adk-python/commit/9c0721beaa526a4437671e6cc70915073be835e3)), closes [#2223](https://github.com/google/adk-python/issues/2223)
* Use correct type for actions parameter in ApplicationIntegrationToolset ([ce7253f](https://github.com/google/adk-python/commit/ce7253f63ff8e78bccc7805bd84831f08990b881))


### Documentation

* Update documents about the information of vibe coding ([0c85587](https://github.com/google/adk-python/commit/0c855877c57775ad5dad930594f9f071164676da))


## [1.8.0](https://github.com/google/adk-python/compare/v1.7.0...v1.8.0) (2025-07-23)

### Features

* [Core]Add agent card builder ([18f5bea](https://github.com/google/adk-python/commit/18f5bea411b3b76474ff31bfb2f62742825b45e5))
* [Core]Add a to_a2a util to convert adk agent to A2A ASGI application ([a77d689](https://github.com/google/adk-python/commit/a77d68964a1c6b7659d6117d57fa59e43399e0c2))
* [Core]Add camel case converter for agents ([0e173d7](https://github.com/google/adk-python/commit/0e173d736334f8c6c171b3144ac6ee5b7125c846))
* [Evals]Use LocalEvalService to run all evals in cli and web ([d1f182e](https://github.com/google/adk-python/commit/d1f182e8e68c4a5a4141592f3f6d2ceeada78887))
* [Evals]Enable FinalResponseMatchV2 metric as an experiment ([36e45cd](https://github.com/google/adk-python/commit/36e45cdab3bbfb653eee3f9ed875b59bcd525ea1))
* [Models]Add support for `model-optimizer-*` family of models in vertex ([ffe2bdb](https://github.com/google/adk-python/commit/ffe2bdbe4c2ea86cc7924eb36e8e3bb5528c0016))
* [Services]Added a sample for History Management ([67284fc](https://github.com/google/adk-python/commit/67284fc46667b8c2946762bc9234a8453d48a43c))
* [Services]Support passing fully qualified agent engine resource name when constructing session service and memory service ([2e77804](https://github.com/google/adk-python/commit/2e778049d0a675e458f4e35fe4104ca1298dbfcf))
* [Tools]Add ComputerUseToolset ([083dcb4](https://github.com/google/adk-python/commit/083dcb44650eb0e6b70219ede731f2fa78ea7d28))
* [Tools]Allow toolset to process llm_request before tools returned by it ([3643b4a](https://github.com/google/adk-python/commit/3643b4ae196fd9e38e52d5dc9d1cd43ea0733d36))
* [Tools]Support input/output schema by fully-qualified code reference ([dfee06a](https://github.com/google/adk-python/commit/dfee06ac067ea909251d6fb016f8331065d430e9))
* [Tools]Enhance LangchainTool to accept more forms of functions ([0ec69d0](https://github.com/google/adk-python/commit/0ec69d05a4016adb72abf9c94f2e9ff4bdd1848c))

### Bug Fixes

* **Attention**: Logging level for some API requests and responses was moved from `INFO` to `DEBUG` ([ff31f57](https://github.com/google/adk-python/commit/ff31f57dc95149f8f309f83f2ec983ef40f1122c))
  * Please set `--log_level=DEBUG`, if you are interested in having those API request and responses in logs.
* Add buffer to the write file option ([f2caf2e](https://github.com/google/adk-python/commit/f2caf2eecaf0336495fb42a2166b1b79e57d82d8))
* Allow current sub-agent to finish execution before exiting the loop agent due to a sub-agent's escalation. ([2aab1cf](https://github.com/google/adk-python/commit/2aab1cf98e1d0e8454764b549fac21475a633409))
* Check that `mean_score` is a valid float value ([65cb6d6](https://github.com/google/adk-python/commit/65cb6d6bf3278e6c3529938a7b932e3ef6d6c2ae))
* Handle non-json-serializable values in the `execute_sql` tool ([13ff009](https://github.com/google/adk-python/commit/13ff009d34836a80f107cb43a632df15f7c215e4))
* Raise `NotFoundError` in `list_eval_sets` function when app_name doesn't exist ([b17d8b6](https://github.com/google/adk-python/commit/b17d8b6e362a5b2a1b6a2dd0cff5e27a71c27925))
* Fixed serialization of tools with nested schema ([53df35e](https://github.com/google/adk-python/commit/53df35ee58599e9816bd4b9c42ff48457505e599))
* Set response schema for function tools that returns `None` ([33ac838](https://github.com/google/adk-python/commit/33ac8380adfff46ed8a7d518ae6f27345027c074))
* Support path level parameters for open_api_spec_parser ([6f01660](https://github.com/google/adk-python/commit/6f016609e889bb0947877f478de0c5729cfcd0c3))
* Use correct type for actions parameter in ApplicationIntegrationToolset ([ce7253f](https://github.com/google/adk-python/commit/ce7253f63ff8e78bccc7805bd84831f08990b881))
* Use the same word extractor for query and event contents in InMemoryMemoryService ([1c4c887](https://github.com/google/adk-python/commit/1c4c887bec9326aad2593f016540160d95d03f33))

### Documentation

* Fix missing toolbox-core dependency and improve installation guide ([2486349](https://github.com/google/adk-python/commit/24863492689f36e3c7370be40486555801858bac))


## 1.7.0 (2025-07-16)

### Features

* Add ability to send state change with message [3f9f773](https://github.com/google/adk-python/commit/3f9f773d9b5fcca343e32f76f6d5677b7cf4c327)
* [Eval] Support for persisting eval run results [bab3be2](https://github.com/google/adk-python/commit/bab3be2cf31dc9afd00bcce70103bdaa5460f1a3)
* Introduce [Plugin]: Plugin is simply a class that packages these individual callback functions together for a broader purpose[162228d](https://github.com/google/adk-python/commit/162228d208dca39550a75221030edf9876bf8e3a)

### Bug Fixes

* Create correct object for image and video content in litellm [bf7745f](https://github.com/google/adk-python/commit/bf7745f42811de3c9c80ec0998001ae50960dafc)
*  Support project-based gemini model path for BuiltInCodeExecutor and all built-in tools [a5d6f1e](https://github.com/google/adk-python/commit/a5d6f1e52ee36d84f94693086f74e4ca2d0bed65)
*  Add instruction in long running tool description to avoid being invoked again by model [62a6119](https://github.com/google/adk-python/commit/62a611956f8907e0580955adb23dfb6d7799bf4f)
*  [A2A] Import A2A well known path from A2A sdk [a6716a5](https://github.com/google/adk-python/commit/a6716a55140f63834ae4e3507b38786da9fdbee2)
*  Fix the long running function response event merge logic [134ec0d](https://github.com/google/adk-python/commit/134ec0d71e8de4cf9bcbe370c7e739e7ada123f3)
*  [A2A] Return final task result in task artifact instead of status message [a8fcc1b](https://github.com/google/adk-python/commit/a8fcc1b8ab0d47eccf6612a6eb8be021bff5ed3a)
* Make InMemoryMemoryService thread-safe [10197db](https://github.com/google/adk-python/commit/10197db0d752defc5976d1f276c7b5405a94c75b)

### Improvements

* Improve partial event handling and streaming aggregation [584c8c6](https://github.com/google/adk-python/commit/584c8c6d91308e62285c94629f020f2746e88f6f)

### Documentation

* Update agent transfer related doc string and comments [b1fa383](https://github.com/google/adk-python/commit/b1fa383e739d923399b3a23ca10435c0fba3460b)
* Update doc string for GcsArtifactService [498ce90](https://github.com/google/adk-python/commit/498ce906dd9b323b6277bc8118e1bcc68c38c1b5)

## [1.6.1](https://github.com/google/adk-python/compare/v1.5.0...v1.6.1) (2025-07-09)

### Features

* Add A2A support as experimental features [f0183a9](https://github.com/google/adk-python/commit/f0183a9b98b0bcf8aab4f948f467cef204ddc9d6)
  * Install google-adk with a2a extra: pip install google-adk[a2a]
  * Users can serve agents as A2A agent with `--a2a` option for `adk web` and
    `adk api_server`
  * Users can run a remote A2A agent with `RemoteA2AAgent` class
  * Three A2A agent samples are added:
    * contributing/samples/a2a_basic
    * contributing/samples/a2a_auth
    * contributing/samples/a2a_human_in_loop

* Support agent hot reload.[e545e5a](https://github.com/google/adk-python/commit/e545e5a570c1331d2ed8fda31c7244b5e0f71584)
  Users can add `--reload_agents` flag to `adk web` and `adk api_server` command
  to reload agents automatically when new changes are detected.

* Eval features
  * Implement auto rater-based evaluator for responses [75699fb](https://github.com/google/adk-python/commit/75699fbeca06f99c6f2415938da73bb423ec9b9b)
  * Add Safety evaluator metric [0bd05df](https://github.com/google/adk-python/commit/0bd05df471a440159a44b5864be4740b0f1565f9)
  * Add BaseEvalService declaration and surrounding data models [b0d88bf](https://github.com/google/adk-python/commit/b0d88bf17242e738bcd409b3d106deed8ce4d407)

* Minor features
  * Add `custom_metadata` to VertexAiSessionService when adding events [a021222](https://github.com/google/adk-python/commit/a02122207734cabb26f7c23e84d2336c4b8b0375)
  * Support protected write in BigQuery `execute_sql` tool [dc43d51](https://github.com/google/adk-python/commit/dc43d518c90b44932b3fdedd33fca9e6c87704e2)
  * Added clone() method to BaseAgent to allow users to create copies of an agent [d263afd] (https://github.com/google/adk-python/commit/d263afd91ba4a3444e5321c0e1801c499dec4c68)

### Bug Fixes

* Support project-based gemini model path to use enterprise_web_search_tool [e33161b](https://github.com/google/adk-python/commit/e33161b4f8650e8bcb36c650c4e2d1fe79ae2526)
* Use inspect.signature() instead of typing.get_type_hints for examining function signatures[4ca77bc](https://github.com/google/adk-python/commit/4ca77bc056daa575621a80d3c8d5014b78209233)
* Replace Event ID generation with UUID4 to prevent SQLite integrity constraint failures [e437c7a](https://github.com/google/adk-python/commit/e437c7aac650ac6a53fcfa71bd740e3e5ec0f230)
* Remove duplicate options from `adk deploy` [3fa2ea7](https://github.com/google/adk-python/commit/3fa2ea7cb923c9f8606d98b45a23bd58a7027436)
* Fix scenario where a user can access another users events given the same session id [362fb3f](https://github.com/google/adk-python/commit/362fb3f2b7ac4ad15852d00ce4f3935249d097f6)
* Handle unexpected 'parameters' argument in FunctionTool.run_async [0959b06](https://github.com/google/adk-python/commit/0959b06dbdf3037fe4121f12b6d25edca8fb9afc)
* Make sure each partial event has different timestamp [17d6042](https://github.com/google/adk-python/commit/17d604299505c448fcb55268f0cbaeb6c4fa314a)
* Avoid pydantic.ValidationError when the model stream returns empty final chunk [9b75e24](https://github.com/google/adk-python/commit/9b75e24d8c01878c153fec26ccfea4490417d23b)
* Fix google_search_tool.py to support updated Gemini LIVE model naming [77b869f](https://github.com/google/adk-python/commit/77b869f5e35a66682cba35563824fd23a9028d7c)
* Adding detailed information on each metric evaluation [04de3e1](https://github.com/google/adk-python/commit/04de3e197d7a57935488eb7bfa647c7ab62cd9d9)
* Converts litellm generate config err [3901fad](https://github.com/google/adk-python/commit/3901fade71486a1e9677fe74a120c3f08efe9d9e)
* Save output in state via output_key only when the event is authored by current agent [20279d9](https://github.com/google/adk-python/commit/20279d9a50ac051359d791dea77865c17c0bbf9e)
* Treat SQLite database update time as UTC for session's last update time [3f621ae](https://github.com/google/adk-python/commit/3f621ae6f2a5fac7f992d3d833a5311b4d4e7091)
* Raise ValueError when sessionId and userId are incorrect combination(#1653) [4e765ae](https://github.com/google/adk-python/commit/4e765ae2f3821318e581c26a52e11d392aaf72a4)
* Support API-Key for MCP Tool authentication [045aea9](https://github.com/google/adk-python/commit/045aea9b15ad0190a960f064d6e1e1fc7f964c69)
* Lock LangGraph version to <= 0.4.10 [9029b8a](https://github.com/google/adk-python/commit/9029b8a66e9d5e0d29d9a6df0e5590cc7c0e9038)
* Update the retry logic of create session polling [3d2f13c](https://github.com/google/adk-python/commit/3d2f13cecd3fef5adfa1c98bf23d7b68ff355f4d)

### Chores

* Extract mcp client creation logic to a separate method [45d60a1](https://github.com/google/adk-python/commit/45d60a1906bfe7c43df376a829377e2112ea3d17)
* Add tests for live streaming configs [bf39c00](https://github.com/google/adk-python/commit/bf39c006102ef3f01e762e7bb744596a4589f171)
* Update ResponseEvaluator to use newer version of Eval SDK [62c4a85](https://github.com/google/adk-python/commit/62c4a8591780a9a3fdb03a0de11092d84118a1b9)
* Add util to build our llms.txt and llms-full.txt files [a903c54](https://github.com/google/adk-python/commit/a903c54bacfcb150dc315bec9c67bf7ce9551c07)
* Create an example for multi agent live streaming [a58cc3d](https://github.com/google/adk-python/commit/a58cc3d882e59358553e8ea16d166b1ab6d3aa71)
* Refactor the ADK Triaging Agent to make the code easier to read [b6c7b5b](https://github.com/google/adk-python/commit/b6c7b5b64fcd2e83ed43f7b96ea43791733955d8)


### Documentation

* Update the a2a example link in README.md [d0fdfb8](https://github.com/google/adk-python/commit/d0fdfb8c8e2e32801999c81de8d8ed0be3f88e76)
* Adds AGENTS.md to provide relevant project context for the Gemini CLI [37108be](https://github.com/google/adk-python/commit/37108be8557e011f321de76683835448213f8515)
* Update CONTRIBUTING.md [ffa9b36](https://github.com/google/adk-python/commit/ffa9b361db615ae365ba62c09a8f4226fb761551)
* Add adk project overview and architecture [28d0ea8](https://github.com/google/adk-python/commit/28d0ea876f2f8de952f1eccbc788e98e39f50cf5)
* Add docstring to clarify that inmemory service are not suitable for production [dc414cb](https://github.com/google/adk-python/commit/dc414cb5078326b8c582b3b9072cbda748766286)
* Update agents.md to include versioning strategy [6a39c85](https://github.com/google/adk-python/commit/6a39c854e032bda3bc15f0e4fe159b41cf2f474b)
* Add tenacity into project.toml [df141db](https://github.com/google/adk-python/commit/df141db60c1137a6bcddd6d46aad3dc506868543)
* Updating CONTRIBUTING.md with missing extra [e153d07](https://github.com/google/adk-python/commit/e153d075939fb628a7dc42b12e1b3461842db541)

## [1.5.0](https://github.com/google/adk-python/compare/v1.4.2...v1.5.0) (2025-06-25)


### Features

* Add a new option `eval_storage_uri` in adk web & adk eval to specify GCS bucket to store eval data ([fa025d7](https://github.com/google/adk-python/commit/fa025d755978e1506fa0da1fecc49775bebc1045))
* Add ADK examples for litellm with add_function_to_prompt ([f33e090](https://github.com/google/adk-python/commit/f33e0903b21b752168db3006dd034d7d43f7e84d))
* Add implementation of VertexAiMemoryBankService and support in FastAPI endpoint ([abc89d2](https://github.com/google/adk-python/commit/abc89d2c811ba00805f81b27a3a07d56bdf55a0b))
* Add rouge_score library to ADK eval dependencies, and implement RougeEvaluator that is computes ROUGE-1 for "response_match_score" metric ([9597a44](https://github.com/google/adk-python/commit/9597a446fdec63ad9e4c2692d6966b14f80ff8e2))
* Add usage span attributes to telemetry ([#356](https://github.com/google/adk-python/issues/356)) ([ea69c90](https://github.com/google/adk-python/commit/ea69c9093a16489afdf72657136c96f61c69cafd))
* Add Vertex Express mode compatibility for VertexAiSessionService ([00cc8cd](https://github.com/google/adk-python/commit/00cc8cd6433fc45ecfc2dbaa04dbbc1a81213b4d))


### Bug Fixes

* Include current turn context when include_contents='none' ([9e473e0](https://github.com/google/adk-python/commit/9e473e0abdded24e710fd857782356c15d04b515))
* Make LiteLLM streaming truly asynchronous ([bd67e84](https://github.com/google/adk-python/commit/bd67e8480f6e8b4b0f8c22b94f15a8cda1336339))
* Make raw_auth_credential and exchanged_auth_credential optional given their default value is None ([acbdca0](https://github.com/google/adk-python/commit/acbdca0d8400e292ba5525931175e0d6feab15f1))
* Minor typo fix in the agent instruction ([ef3c745](https://github.com/google/adk-python/commit/ef3c745d655538ebd1ed735671be615f842341a8))
* Typo fix in sample agent instruction ([ef3c745](https://github.com/google/adk-python/commit/ef3c745d655538ebd1ed735671be615f842341a8))
* Update contributing links ([a1e1441](https://github.com/google/adk-python/commit/a1e14411159fd9f3e114e15b39b4949d0fd6ecb1))
* Use starred tuple unpacking on GCS artifact blob names ([3b1d9a8](https://github.com/google/adk-python/commit/3b1d9a8a3e631ca2d86d30f09640497f1728986c))


### Chore

* Do not send api request when session does not have events ([88a4402](https://github.com/google/adk-python/commit/88a4402d142672171d0a8ceae74671f47fa14289))
* Leverage official uv action for install([09f1269](https://github.com/google/adk-python/commit/09f1269bf7fa46ab4b9324e7f92b4f70ffc923e5))
* Update google-genai package and related deps to latest([ed7a21e](https://github.com/google/adk-python/commit/ed7a21e1890466fcdf04f7025775305dc71f603d))
* Add credential service backed by session state([29cd183](https://github.com/google/adk-python/commit/29cd183aa1b47dc4f5d8afe22f410f8546634abc))
* Clarify the behavior of Event.invocation_id([f033e40](https://github.com/google/adk-python/commit/f033e405c10ff8d86550d1419a9d63c0099182f9))
* Send user message to the agent that returned a corresponding function call if user message is a function response([7c670f6](https://github.com/google/adk-python/commit/7c670f638bc17374ceb08740bdd057e55c9c2e12))
* Add request converter to convert a2a request to ADK request([fb13963](https://github.com/google/adk-python/commit/fb13963deda0ff0650ac27771711ea0411474bf5))
* Support allow_origins in cloud_run deployment ([2fd8feb](https://github.com/google/adk-python/commit/2fd8feb65d6ae59732fb3ec0652d5650f47132cc))

## [1.4.2](https://github.com/google/adk-python/compare/v1.4.1...v1.4.2) (2025-06-20)


### Bug Fixes

* Add type checking to handle different response type of genai API client ([4d72d31](https://github.com/google/adk-python/commit/4d72d31b13f352245baa72b78502206dcbe25406))
  * This fixes the broken VertexAiSessionService
* Allow more credentials types for BigQuery tools ([2f716ad](https://github.com/google/adk-python/commit/2f716ada7fbcf8e03ff5ae16ce26a80ca6fd7bf6))

## [1.4.1](https://github.com/google/adk-python/compare/v1.3.0...v1.4.1) (2025-06-18)


### Features

* Add Authenticated Tool (Experimental) ([dcea776](https://github.com/google/adk-python/commit/dcea7767c67c7edfb694304df32dca10b74c9a71))
* Add enable_affective_dialog and proactivity to run_config and llm_request ([fe1d5aa](https://github.com/google/adk-python/commit/fe1d5aa439cc56b89d248a52556c0a9b4cbd15e4))
* Add import session API in the fast API ([233fd20](https://github.com/google/adk-python/commit/233fd2024346abd7f89a16c444de0cf26da5c1a1))
* Add integration tests for litellm with and without turning on add_function_to_prompt ([8e28587](https://github.com/google/adk-python/commit/8e285874da7f5188ea228eb4d7262dbb33b1ae6f))
* Allow data_store_specs pass into ADK VAIS built-in tool ([675faef](https://github.com/google/adk-python/commit/675faefc670b5cd41991939fe0fc604df331111a))
* Enable MCP Tool Auth (Experimental) ([157d9be](https://github.com/google/adk-python/commit/157d9be88d92f22320604832e5a334a6eb81e4af))
* Implement GcsEvalSetResultsManager to handle storage of eval sets on GCS, and refactor eval set results manager ([0a5cf45](https://github.com/google/adk-python/commit/0a5cf45a75aca7b0322136b65ca5504a0c3c7362))
* Re-factor some eval sets manager logic, and implement GcsEvalSetsManager to handle storage of eval sets on GCS ([1551bd4](https://github.com/google/adk-python/commit/1551bd4f4d7042fffb497d9308b05f92d45d818f))
* Support real time input config ([d22920b](https://github.com/google/adk-python/commit/d22920bd7f827461afd649601326b0c58aea6716))
* Support refresh access token automatically for rest_api_tool ([1779801](https://github.com/google/adk-python/commit/177980106b2f7be9a8c0a02f395ff0f85faa0c5a))

### Bug Fixes

* Fix Agent generate config err ([#1305](https://github.com/google/adk-python/issues/1305)) ([badbcbd](https://github.com/google/adk-python/commit/badbcbd7a464e6b323cf3164d2bcd4e27cbc057f))
* Fix Agent generate config error ([#1450](https://github.com/google/adk-python/issues/1450)) ([694b712](https://github.com/google/adk-python/commit/694b71256c631d44bb4c4488279ea91d82f43e26))
* Fix liteLLM test failures ([fef8778](https://github.com/google/adk-python/commit/fef87784297b806914de307f48c51d83f977298f))
* Fix tracing for live ([58e07ca](https://github.com/google/adk-python/commit/58e07cae83048d5213d822be5197a96be9ce2950))
* Merge custom http options with adk specific http options in model api request ([4ccda99](https://github.com/google/adk-python/commit/4ccda99e8ec7aa715399b4b83c3f101c299a95e8))
* Remove unnecessary double quote on Claude docstring ([bbceb4f](https://github.com/google/adk-python/commit/bbceb4f2e89f720533b99cf356c532024a120dc4))
* Set explicit project in the BigQuery client ([6d174eb](https://github.com/google/adk-python/commit/6d174eba305a51fcf2122c0fd481378752d690ef))
* Support streaming in litellm + adk and add corresponding integration tests ([aafa80b](https://github.com/google/adk-python/commit/aafa80bd85a49fb1c1a255ac797587cffd3fa567))
* Support project-based gemini model path to use google_search_tool ([b2fc774](https://github.com/google/adk-python/commit/b2fc7740b363a4e33ec99c7377f396f5cee40b5a))
* Update conversion between Celsius and Fahrenheit ([1ae176a](https://github.com/google/adk-python/commit/1ae176ad2fa2b691714ac979aec21f1cf7d35e45))

### Chores

* Set `agent_engine_id` in the VertexAiSessionService constructor, also use the `agent_engine_id` field instead of overriding `app_name` in FastAPI endpoint ([fc65873](https://github.com/google/adk-python/commit/fc65873d7c31be607f6cd6690f142a031631582a))



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
* Set output_schema correctly for LiteLlm ([6157db7](https://github.com/google/adk-python/commit/6157db77f2fba4a44d075b51c83bff844027a147))
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
* Add [BigQuery first-party tools](https://github.com/google/adk-python/commit/d6c6bb4b2489a8b7a4713e4747c30d6df0c07961).


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
* Docstring improvements, typo fixings, github action to enforce code styles on formatting and imports, etc.

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

* Fix typo in method name in `Event`: has_trailing_code_execution_result --> has_trailing_code_execution_result.

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

* Adds unit tests in GitHub action.
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
