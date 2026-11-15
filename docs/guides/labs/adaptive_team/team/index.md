# adaptive_team

`adaptive_team` builds a workflow that answers an open task by planning it,
choosing one specialist at a time, and replacing the plan when the work
stalls.

## Introduction

Some tasks do not have a fixed path. A research brief, a comparison, or a
draft that needs both reading and calculation can start with a plan and then
change that plan when a step repeats itself. `Workflow` can already run a
Python function that calls `ctx.run_node`. What each application would
otherwise rebuild is the plan, the record of each turn, the stall limit, the
optional pause for a person to accept the plan, and the final answer.

`adaptive_team` returns that workflow. The caller supplies a coordinator
`LlmAgent` and the specialist agents. The helper supplies the instructions
for the plan, the progress check, and the final answer. The coordinator's
own instruction is appended as extra guidance. Specialist tools stay on the
specialists. The coordinator's tools are not called during those three
phases, because each phase has to return one structured object or one final
reply.

The package is experimental. `ADAPTIVE_TEAM` is enabled by default and logs a
warning the first time the helper is used. Set `ADK_DISABLE_ADAPTIVE_TEAM=1`
to turn it off.

## Get started

The workflow below compares two heating fuels. The researcher collects facts.
The analyst compares them. The coordinator's instruction only adds guidance
about sources and estimates.

```python
from google.adk import Agent
from google.adk.labs.adaptive_team import adaptive_team

researcher = Agent(
    name="researcher",
    description="Collects qualitative facts and cites where they come from.",
    instruction="Gather the facts the assignment asks for.",
)
analyst = Agent(
    name="analyst",
    description="Compares figures and states the main uncertainty.",
    instruction="Compare the facts you are given.",
)
coordinator = Agent(
    name="coordinator",
    description="Turns an open question into a plan and a final answer.",
    instruction="Prefer sources a reader can check.",
)

root_agent = adaptive_team(
    name="energy_brief",
    manager=coordinator,
    participants=[researcher, analyst],
    max_rounds=6,
    stall_limit=2,
    replan_limit=1,
)
```

Run it the same way as any other workflow. The workflow's output is the final
answer text.

## How it works

One driver node inside the workflow owns the loop.

1. The coordinator writes a `TaskPlan`: a goal, facts already known, open
   questions, and ordered steps. Each step is supposed to name a specialist
   from the roster.
1. When `review_plan` is true, the workflow pauses with `RequestInput` and a
   `PlanDecision` schema. Approving the plan continues it. Rejecting it, and
   sending notes, spends one replacement and asks for a new plan. With no
   replacements left, the current plan continues.
1. Each round, the coordinator returns a `RoundAssessment`. `satisfied` ends
   the loop. Otherwise `participant` must be a roster name and `assignment`
   is the user message for that specialist.
1. The specialist runs through `ctx.run_node` and the turn is appended to the
   ledger. A turn is stalled when the assessment says so, when that
   specialist was just given the same assignment, or when the name is not on
   the roster or the assignment is empty.
1. `stall_limit` stalled turns in a row replace the plan, up to
   `replan_limit` replacements. The new plan is shown the previous plan and
   the reason the team stalled. After the last allowed stall, the coordinator
   writes a best-effort answer instead.
1. `max_rounds` counts specialist turns across every attempt. Reaching it
   also writes the answer.

The ledger is a `TeamSnapshot` in session state, under
`team_state_key(name)`. It stores the task, the plan, every round, the stall
window, and the final answer. The snapshot is tied to the current invocation.
A pause for review stays in that invocation, so the run continues from the
saved ledger and does not repeat model calls that already finished. A new
user message is a new invocation and starts a new ledger.

Coordinator events set `custom_metadata["adaptive_team"]` to a dict whose
`kind` is a `TeamNotice`: `plan`, `replan`, `progress`, or `answer`. The
`answer` event's output is the final text. Specialist events keep the
specialist's author, so a caller can show those turns without treating them
as the answer.

`stop_reason` on the snapshot is one of `satisfied`, `round_limit`,
`replan_limit`, or `empty_task`.

## Configuration options

| Option         | Type                    | Default          | Description                                                                                                                          |
| :------------- | :---------------------- | :--------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `name`         | `str`                   |                  | Workflow name. Also the suffix of the session-state key.                                                                             |
| `manager`      | `LlmAgent`              |                  | Coordinator. Its instruction is extra guidance. Its tools are not used for the plan, the progress check, or the answer.              |
| `participants` | sequence of `BaseAgent` |                  | Specialists the coordinator may call. Names must be unique Python identifiers and must not match the coordinator or its phase names. |
| `max_rounds`   | `int`                   | `8`              | Maximum specialist turns across all attempts.                                                                                        |
| `stall_limit`  | `int`                   | `2`              | Consecutive stalled turns before the plan is replaced.                                                                               |
| `replan_limit` | `int`                   | `1`              | How many replacement plans are allowed. Zero keeps the first plan only.                                                              |
| `review_plan`  | `bool`                  | `False`          | Pause for a `PlanDecision` before each plan is executed.                                                                             |
| `description`  | `str`                   | a short sentence | Description stored on the workflow.                                                                                                  |

Give every specialist a `description`. The roster the coordinator sees is the
name plus that description. The specialist's own `instruction` still applies
when that agent runs.

`stall_limit` counts consecutive stalled turns, not every turn. A turn that
the coordinator marks as progress resets the streak. Replacing the plan also
resets the streak. Earlier turns stay in the transcript so the new plan and
the final answer can use them.

A rejection during review spends the same budget as a stall replacement. The
notes on the rejection are the reason the next plan has to address.

## Advanced applications

Pause for a person to edit the plan before any specialist runs:

```python
root_agent = adaptive_team(
    name="energy_brief",
    manager=coordinator,
    participants=[researcher, analyst],
    review_plan=True,
    replan_limit=2,
)
```

The pause is a normal `adk_request_input` function call. Resume it with a
`PlanDecision`, for example `{"approved": false, "notes": "focus on cost"}`.
The interrupt id is `review-v` plus the plan version, starting at
`review-v0`.

Read the ledger after the run:

```python
from google.adk.labs.adaptive_team import TeamSnapshot
from google.adk.labs.adaptive_team import team_state_key

snapshot = TeamSnapshot.model_validate(
    session.state[team_state_key("energy_brief")]
)
```

Filter the event stream by kind when the caller wants the plan and the answer
without assembling them from specialist text:

```python
from google.adk.labs.adaptive_team import TEAM_EVENT_METADATA_KEY
from google.adk.labs.adaptive_team import TeamNotice

kind = (event.custom_metadata or {}).get(TEAM_EVENT_METADATA_KEY, {})
if kind.get("kind") == TeamNotice.ANSWER:
    final_answer = event.output
```

## Limitations

The coordinator phases do not see session history and do not call the
manager's tools. Everything they should know has to fit in the task, the
plan, and the transcript. A specialist keeps the tools and the mode
configured on that agent.

Participant names are matched exactly, after stripping surrounding space.
Descriptions are the only roster text the coordinator is given, so a missing
description makes the roster less useful.

The ledger lives in session state for the invocation. It is not a memory
service, and a new user message does not continue the previous plan.

The feature is experimental. Field names on `TaskPlan`, `RoundAssessment`,
`PlanDecision`, and `TeamSnapshot` can still change.

## Related samples

- [Adaptive team](../../../../../contributing/samples/workflows/adaptive_team/agent.py) - Two specialists and a coordinator with a stall limit of two rounds.
