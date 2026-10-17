# Context Filter Plugin

`ContextFilterPlugin` trims the conversation that each model request carries,
so a long session does not keep growing the prompt. Only the request is
trimmed; the session keeps every event.

**Options:**

- `num_invocations_to_keep`: how many recent invocations stay in the request.
  An invocation starts with one or more consecutive user messages and covers
  every model turn and tool call until the next user message.
- `custom_filter`: a function that receives the contents and returns the ones
  to keep, for rules of your own.
- `remove_amount`: how many invocations to drop at once when the limit is
  passed.

The plugin keeps function calls paired with their responses, so trimming never
leaves a tool response whose matching call was dropped.

## Sample

The agent looks up delivery status for orders, and the sample asks about three
of them before asking what has been discussed so far. It registers two plugins:

```python
plugins=[
    ContextFilterPlugin(num_invocations_to_keep=2),
    ContentCounterPlugin(),
]
```

`ContentCounterPlugin` is a few lines defined in the sample. It prints how many
contents each request carries. Plugin callbacks run in registration order, so
it sees the request after the filter has trimmed it.

Run it with:

```bash
python contributing/samples/plugins/plugin_context_filter/main.py
```

Output:

```
user: Look up the delivery status for order A1.
[request 1] contents sent to the model: 1
[request 2] contents sent to the model: 3
agent: Order A1 is in transit.

user: Now look up order B2.
[request 3] contents sent to the model: 5
[request 4] contents sent to the model: 7
agent: Order B2 is in transit.

user: Now look up order C3.
[request 5] contents sent to the model: 5
[request 6] contents sent to the model: 7
agent: Order C3 is in transit.

user: Which orders have I asked about so far?
[request 7] contents sent to the model: 5
agent: You have only asked about order C3.

events kept in the session: 14
```

Two things to read from that output:

- The request stops growing. Without the plugin the count keeps climbing with
  every turn; here it settles at 5 to 7 contents.
- The last answer is wrong about history, and that is the trade-off. The model
  can only answer from what it still sees, so keep enough invocations for the
  questions your agent has to answer, or use `custom_filter` to keep the parts
  that matter.
