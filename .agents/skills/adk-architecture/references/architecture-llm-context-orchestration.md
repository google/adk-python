# LLM context orchestration from events

## Source versus view

The event stream and the LLM context are not the same thing:

- **Events are the ground truth.** Immutable records of what happened — user
  messages, model responses, tool calls, results. They are the audit log and
  the persistence state.
- **LLM context is an orchestrated view.** What gets sent to a model is not a
  dump of the event log. It is filtered and transformed for the role, task and
  branch of the agent currently running.

Treating the two as interchangeable is the root of most "why did the model see
that?" bugs.

## Delegation

A coordinator hands work to a sub-agent with the `transfer_to_agent` tool,
which sets `actions.transfer_to_agent` on the event rather than calling the
sub-agent inline. The alternative is `AgentTool`, which wraps an agent so it is
invoked as an ordinary tool and returns its result to the caller.

Either way the sub-agent does not inherit the coordinator's full transcript
verbatim; it is given the task framing plus whatever history its branch
exposes.

## Branch isolation

Events from every node and branch share one session, in chronological order.
Isolation comes from the `branch` field, a dot-separated path built by
`_BranchPath`: `create_sub_branch('parent', name='child', run_id='1')` yields
`'parent.child@1'`. A node running on a sub-branch sees only events on its own
path, which is what keeps parallel siblings from polluting each other.

`Event.isolation_scope` is a separate, coarser tag — NodeRunner stamps it from
`ctx.isolation_scope` when the event does not carry one already.

## History trimming and compaction

Long histories overflow the context window and drag stale retry loops back into
the prompt. When the app sets `events_compaction_config`, the Runner compacts
events after the invocation. Because compaction rewrites aged history, do not
store transient status on an event and expect to read it back later.
