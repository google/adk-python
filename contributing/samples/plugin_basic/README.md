# ADK Agent with Plugin

### What is ADK Plugin?

At its core, ADK extensibility is built on
[**callbacks**](https://google.github.io/adk-docs/callbacks/): functions you
write that ADK automatically executes at key stages of an agent's lifecycle.
**A Plugin is simply a class that packages these individual callback functions
together for a broader purpose.**

While a standard Agent Callback is configured on a *single agent, a single tool*
for a *specific task*, a Plugin is registered *once* on the `Runner` and its
callbacks apply *globally* to every agent, tool, and LLM call managed by that
runner. This makes Plugins the ideal solution for implementing horizontal
features that cut across your entire application.

### What can plugins do?

Plugins are incredibly versatile. By implementing different callback methods, you
can achieve a wide range of functionalities.

*   **Logging & Tracing**: Create detailed logs of agent, tool, and LLM activity
    for debugging and performance analysis.
*   **Policy Enforcement**: Implement security guardrails. For example, a
    before\_tool\_callback can check if a user is authorized to use a specific
    tool and prevent its execution by returning a value.
*   **Monitoring & Metrics**: Collect and export metrics on token usage,
    execution times, and invocation counts to monitoring systems like Prometheus
    or Stackdriver.
*   **Caching**: In before\_model\_callback or before\_tool\_callback, you can
    check if a request has been made before. If so, you can return a cached
    response, skipping the expensive LLM or tool call entirely.
*   **Request/Response Modification**: Dynamically add information to LLM prompts
    (e.g., in before\_model\_callback) or standardize tool outputs (e.g., in
    after\_tool\_callback).

### Run the agent

**Note: Plugin is NOT supported in `adk web`yet.**

Use following command to run the main.py

```bash
python3 -m contributing.samples.plugin_basic.main
```

It should output the following content. Note that the outputs from plugin are
printed.

```bash
[Plugin] Agent run count: 1
[Plugin] LLM request count: 1
** Got event from hello_world
Hello world: query is [hello world]
** Got event from hello_world
[Plugin] LLM request count: 2
** Got event from hello_world
```