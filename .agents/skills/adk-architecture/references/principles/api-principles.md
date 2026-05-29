# API Principles

Guidelines for designing and maintaining the ADK public API surface.

## Public API Surface

The public API surface of ADK includes:
- All public classes, methods, and functions in the `google.adk` namespace.
- The names, required parameters, and expected behavior of all built-in Tools.
- The structure and schema of persisted data (Sessions, Memory, Evaluation datasets).
- The JSON request/response format of the ADK API server.
- The command-line interface (CLI) commands, arguments, and flags.
- The expected file structure for agent definitions (e.g., `agent.py` convention loaded by CLI).

## Design Principles

### 1. Stability and Backward Compatibility
- ADK adheres to Semantic Versioning 2.0.0.
- Any change that forces a developer to alter their existing code to upgrade is a **breaking change** and necessitates a MAJOR version bump.
- Avoid breaking changes whenever possible by using optional parameters and deprecation cycles.

### 2. Self-Containment
- Each package should be as self-contained as possible to reduce coupling.
- Within the ADK framework, importing from a package's `__init__.py` is **not allowed**. Import from the specific module directly.

### 3. Explicit Exports
- The public API of a package must be explicitly exported in `__init__.py`.
- **Only public names** should be imported into `__init__.py`. This keeps `__init__.py` minimal and prevents accidental exposure of internal implementation details.

### 4. Intuitive Naming
- Public method and class names should be concise and intuitive.
- Private method names can be longer and more self-explanatory to reduce the need for comments.

#### Examples

**Public Naming**
- **Good**: `Runner.run()`, `Session.get_events()`
- **Bad**: `Runner.orchestrate_agent_invocation_loop()`, `Session.retrieve_all_events_from_storage()`

**Private Naming**
- **Good**: `_prepare_context_for_llm()`, `_should_trim_history()`
- **Bad**: `_prep()`, `_trim()`
