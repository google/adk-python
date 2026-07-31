# Live Workflow Sample

## Overview

This sample composes three short, single-purpose **live (voice) agents** into a graph-based workflow:

1. `greeter_agent` — greets and confirms the caller's name.
1. `dob_verifier_agent` — captures and validates the caller's date of birth
   (using the `validate_date_of_birth` tool).
1. `goals_agent` — once identity is verified, delivers the call goals and wraps
   up the conversation.

Each stage runs in `mode='task'` and hands a typed result to the next
(`GreeterOutput`, `DobOutput`). The stages are wired directly into the
workflow's `edges`, so the framework runs them in order.

## Running the agent

```bash
uv run adk web contributing/samples/live/live_workflow
```

Open the ADK web interface and start a Live Session with the agent.

## Evaluating this agent

`test_config.json` and `live_workflow.evalset.json` evaluate the workflow in
**live mode** with an `llm_audio` user simulator (each user turn is synthesized
to audio and streamed to the live agent). The eval scores response quality,
tool-use quality, and overall trajectory quality against custom rubrics across
the staged conversation.

This sample uses the in-process, rubric-based LLM-as-judge metrics
(`rubric_based_final_response_quality_v1`, `rubric_based_tool_use_quality_v1`,
and `rubric_based_multi_turn_trajectory_quality_v1`), which support multi-agent
conversations. The first two are scored per turn; the trajectory metric judges
the whole conversation end-to-end.

The eval case uses a `conversation_scenario`, so an LLM-simulated user adapts to
each stage of the workflow instead of following a fixed script.

1. Install the eval extra: `uv pip install -e ".[eval]"`.
1. Add a `.env` in this directory with Vertex AI credentials (see
   `live_bidi_streaming_single_agent/.env`). The project needs access to both
   the Live API and Gemini TTS models.
1. Run the eval:
   ```bash
   uv run adk eval \
     contributing/samples/live/live_workflow \
     contributing/samples/live/live_workflow/live_workflow.evalset.json \
     --config_file_path contributing/samples/live/live_workflow/test_config.json
   ```
