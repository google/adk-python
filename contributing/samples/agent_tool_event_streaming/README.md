# AgentTool Event Streaming Demo

This sample demonstrates the AgentTool event streaming feature (Issue #3984).


**Before the fix:**
- When a coordinator agent delegates to a sub-agent via AgentTool, the sub-agent's execution acts as a "black box"
- No events are yielded during sub-agent execution
- Frontend appears unresponsive for the duration of sub-agent execution
- Only the final result is returned after sub-agent completes

**After the fix:**
- Sub-agent events are streamed in real-time to the parent Runner
- Frontend receives immediate feedback about sub-agent progress
- Users can see intermediate steps, tool calls, and responses as they happen
- Much better UX for hierarchical multi-agent systems

## Running the Demo

```bash
cd contributing/samples
adk web .

```

Then in the web UI, select agent_tool_event_streaming from the dropdown
1. Ask: "Research the history of artificial intelligence"
2. Watch the events stream in real-time - you'll see:
   - Coordinator agent's function call
   - Research agent's step-by-step progress
   - Research agent's intermediate responses
   - Final summary


## Expected Behavior

With event streaming enabled, you should see:

1. **Coordinator events:**
   - Function call to `research_agent`

2. **Research agent events (streamed in real-time):**
   - "Step 1: Acknowledging task..."
   - "Step 2: Researching topic..."
   - "Step 3: Analyzing findings..."
   - "Final summary: ..."

3. **Coordinator final response:**
   - Summary of the research

All events should appear progressively, not all at once at the end.

## Before/After Comparison

To see the difference:

1. **Before fix:** Run on a branch without the event streaming feature
   - You'll see: Coordinator call → (long pause) → Final result
   
2. **After fix:** Run on this branch
   - You'll see: Coordinator call → Research steps streaming → Final result

