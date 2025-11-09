# Sample Agent to demo session state persistence.

## Lifecycle of session state

After assigning a state using the context object (e.g.
`tool_context.state['log_query_var'] = 'log_query_var_value'`):

* The state is available for use in a later callback.
* Once the resulting event is processed by the runner and appended in the
  session, the state will be also persisted in the session.

This sample agent is for demonstrating the aforementioned behavior.

## Run the agent

Run below command:

```bash
$ adk run contributing/samples/session_state_agent --replay contributing/samples/session_state_agent/input.json
```

And you should see below output:

```bash
[user]: hello world!
===================== In before_agent_callback ==============================
** Asserting keys are cached in context: ['before_agent_callback_state_key'] pass ✅
** Asserting keys are already persisted in session: [] pass ✅
** Asserting keys are not persisted in session yet: ['before_agent_callback_state_key'] pass ✅
============================================================
===================== In before_model_callback ==============================
** Asserting keys are cached in context: ['before_agent_callback_state_key', 'before_model_callback_state_key'] pass ✅
** Asserting keys are already persisted in session: ['before_agent_callback_state_key'] pass ✅
** Asserting keys are not persisted in session yet: ['before_model_callback_state_key'] pass ✅
============================================================
===================== In after_model_callback ==============================
** Asserting keys are cached in context: ['before_agent_callback_state_key', 'before_model_callback_state_key', 'after_model_callback_state_key'] pass ✅
** Asserting keys are already persisted in session: ['before_agent_callback_state_key'] pass ✅
** Asserting keys are not persisted in session yet: ['before_model_callback_state_key', 'after_model_callback_state_key'] pass ✅
============================================================
[root_agent]: Hello! How can I help you verify something today?

===================== In after_agent_callback ==============================
** Asserting keys are cached in context: ['before_agent_callback_state_key', 'before_model_callback_state_key', 'after_model_callback_state_key', 'after_agent_callback_state_key'] pass ✅
** Asserting keys are already persisted in session: ['before_agent_callback_state_key', 'before_model_callback_state_key', 'after_model_callback_state_key'] pass ✅
** Asserting keys are not persisted in session yet: ['after_agent_callback_state_key'] pass ✅
============================================================
```

## Detailed Explanation

As rule of thumb, to read and write session state, user should assume the
state is available after writing via the context object
(`tool_context`, `callback_context` or `readonly_context`).

### Current Behavior

The current behavior of persisting states are:

* for `before_agent_callback`: state delta will be persisted after all callbacks are processed.
* for `before_model_callback`: state delta will be persisted with the final LlmResponse,
  aka. after `after_model_callback` is processed.
* for `after_model_callback`: state delta will be persisted together with the event of LlmResponse.
* for `after_agent_callback`: state delta will be persisted after all callbacks are processed.

**NOTE**: the current behavior is considered implementation detail and may be changed later. **DO NOT** rely on it.
