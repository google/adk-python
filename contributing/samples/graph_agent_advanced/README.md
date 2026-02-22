# GraphAgent Advanced Example - All Features

This example demonstrates **all features** of the GraphAgent interrupt and observability framework through a realistic research paper writing workflow.

---

## Features Demonstrated

### 1. **Checkpointing** ✅
- Automatic checkpoint creation at each node
- Resume from any checkpoint
- Checkpoint listing and management
- State preservation across restarts

**Code**: See `scenario_3_checkpointing_and_resume()`

```python
# Enable checkpointing
graph = GraphAgent(
    name="research_workflow",
    checkpointing=True,
    checkpoint_service=checkpoint_service,
)

# List checkpoints
checkpoints = await checkpoint_service.list_checkpoints(session)

# Restore from checkpoint
restored = await checkpoint_service.restore_checkpoint(session, checkpoint_id)
```

---

### 2. **LLM-based Interrupt Reasoning** ✅
- InterruptReasoner analyzes interrupt messages
- Context-aware decisions (uses current node, state, execution path)
- Available actions: continue, rerun, pause, defer, skip
- Extensible via custom_actions

**Code**: See `scenario_2_interrupt_with_reasoning()`

```python
# Create LLM-based reasoner
reasoner = InterruptReasoner(
    config=InterruptReasonerConfig(
        model="gemini-2.0-flash-exp",
        available_actions=["continue", "rerun", "pause", "defer", "skip"],
        instruction="You are an interrupt reasoning agent...",
    )
)

# Use in GraphAgent
graph = GraphAgent(
    name="research_workflow",
    interrupt_config=InterruptConfig(
        mode=InterruptMode.AFTER,
        reasoner=reasoner,  # LLM decides actions
    ),
)

# Send interrupt - LLM will analyze and decide
await interrupt_service.send_interrupt(
    session_id=session.id,
    text="The literature review missed key papers on neural architecture search",
    action="defer",  # Suggestion, but LLM may override
)
```

---

### 3. **Callback-based Observability** ✅
- Custom before/after node callbacks
- Full access to state, iteration, invocation context
- Rich multi-content events (text, JSON, metadata)
- Developers control format (no hardcoded strings)

**Code**: See `research_observability_callback()`

```python
async def research_observability_callback(ctx: NodeCallbackContext) -> Optional[Event]:
    """Custom observability with rich content."""
    parts = [
        types.Part(text=f"📝 **Executing**: {ctx.node.name}"),
        types.Part(text=f"Progress: {progress:.1f}%"),
        types.Part(text=f"**State**:\n```json\n{json.dumps(ctx.state.data, indent=2)}\n```"),
    ]

    return Event(
        author="observability",
        content=types.Content(parts=parts),
        actions=EventActions(
            state_delta={
                "observability_node": ctx.node.name,
                "observability_progress": progress,
            },
        ),
    )

# Use in GraphAgent
graph = GraphAgent(
    name="research_workflow",
    before_node_callback=research_observability_callback,
    after_node_callback=create_nested_observability_callback(),
)
```

---

### 4. **Flexible Interrupt Timings** ✅
- **BEFORE**: Validate before node execution (pre-conditions)
- **AFTER**: Correct after node execution (retrospective feedback)
- **BOTH**: Both before and after
- Per-node configuration

**Code**: See `scenario_5_all_interrupt_timings()`

```python
# Interrupt AFTER node execution (default, retrospective)
graph = GraphAgent(
    interrupt_config=InterruptConfig(
        mode=InterruptMode.AFTER,  # Check after each node
        nodes=None,  # All nodes (or specify specific nodes)
    ),
)

# Interrupt BEFORE specific nodes (validation)
graph = GraphAgent(
    interrupt_config=InterruptConfig(
        mode=InterruptMode.BEFORE,  # Check before execution
        nodes=["peer_review"],  # Only before peer_review
    ),
)

# Interrupt BOTH before and after
graph = GraphAgent(
    interrupt_config=InterruptConfig(
        mode=InterruptMode.BOTH,  # Check before AND after
        nodes=["write_paper", "peer_review"],
    ),
)
```

---

### 5. **Immediate Cancellation (ESC-like)** ✅
- Cancels **during** node execution (not just between nodes)
- State preservation on cancel (partial results, execution path)
- Resume capability after cancellation
- Clean session cleanup

**Code**: See `scenario_4_immediate_cancellation()`

```python
# Cancel immediately (ESC-like)
await interrupt_service.cancel_session(session.id)

# Check preserved state
print(session.state.get("graph_cancelled"))  # True
print(session.state.get("graph_cancelled_at_node"))  # "write_paper"
print(session.state.get("graph_can_resume"))  # True
print(session.state.get("graph_data"))  # Partial domain data saved
```

**Cancellation Paths**:
1. **Between nodes**: Cancels at iteration start
2. **During node execution**: Cancels mid-execution (TRUE immediate)
3. **Task cancellation**: Handles `asyncio.CancelledError`

All paths save:
- `graph_state`: Partial execution state
- `graph_cancelled_at_node`: Where cancellation occurred
- `graph_path`: Execution path so far
- `graph_partial_output`: Partial node output (if mid-execution)
- `graph_can_resume`: Resume capability flag

---

### 6. **All Interrupt Actions** ✅

| Action | Description | Example Use Case |
|--------|-------------|------------------|
| `continue` | Proceed normally | "Looks good, continue" |
| `rerun` | Re-execute current node with guidance | "Rerun with more details" |
| `pause` | Pause execution (escalate=True) | "Wait for human approval" |
| `defer` | Save for later (add to todos) | "Good idea, but not urgent" |
| `skip` | Skip current node | "No need for peer review" |

**Code**: See `scenario_2_interrupt_with_reasoning()`

```python
# Defer action - saves to session.state["_interrupt_todos"]
await interrupt_service.send_interrupt(
    session_id=session.id,
    text="Add section on ethical implications",
    action="defer",
)

# Check deferred todos
todos = session.state.get("_interrupt_todos", [])
print(f"Deferred: {len(todos)} items")

# Rerun action - adds guidance to state metadata
await interrupt_service.send_interrupt(
    session_id=session.id,
    text="Rerun with more focus on practical applications",
    action="rerun",
)

# Pause action - escalates to pause execution
await interrupt_service.send_interrupt(
    session_id=session.id,
    text="Pause for team review",
    action="pause",
)
```

---

## Workflow Overview

**Research Paper Writing Workflow**:

```
┌─────────────────────┐
│ Literature Review   │ ──> Review existing papers
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Generate Hypotheses │ ──> Propose testable hypotheses
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Design Methodology  │ ──> Plan experimental methods
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Analyze Results     │ ──> Run simulated experiments
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Write Paper         │ ◄─┐ Write academic paper
└──────────┬──────────┘   │
           │              │
┌──────────▼──────────┐   │
│ Peer Review         │   │ Review quality
└──────────┬──────────┘   │
           │              │
     [needs revision?] ───┘ (loop back if score < 7/10)
           │
     [accept or max revisions]
           │
        [END]
```

**Checkpoints created**:
- After each major node
- Stored in session.state
- Can resume from any checkpoint

**Interrupt points**:
- AFTER each node (default)
- Can send interrupts at any time
- LLM reasons about best action

---

## Running the Examples

### Prerequisites

```bash
# Install dependencies
pip install google-adk

# Set Gemini API key
export GOOGLE_API_KEY="your-api-key"
```

### Run All Scenarios

```bash
python -m contributing.samples.graph_agent_advanced.agent
```

### Run Individual Scenarios

```python
# In Python REPL
from contributing.samples.graph_agent_advanced.agent import *

# Scenario 1: Basic execution with observability
await scenario_1_basic_execution()

# Scenario 2: Interrupt with LLM reasoning
await scenario_2_interrupt_with_reasoning()

# Scenario 3: Checkpointing and resume
await scenario_3_checkpointing_and_resume()

# Scenario 4: Immediate cancellation
await scenario_4_immediate_cancellation()

# Scenario 5: All interrupt timings
await scenario_5_all_interrupt_timings()
```

---

## Expected Output

### Scenario 1: Basic Execution
```
==================================================================================
SCENARIO 1: Basic Execution with Observability
==================================================================================

Running research workflow...

[observability] 📝 **Executing**: literature_review
Progress: 5.0% (iteration 1)
**State**:
```json
{
  "topic": "Impact of AI on software development"
}
```

[literature_review] Based on recent literature, key papers include...

[observability] 📝 **Executing**: generate_hypotheses
Progress: 10.0% (iteration 2)
...

✅ Workflow completed!
Final state keys: ['topic', 'literature_review', 'hypotheses', 'methodology', 'analysis_results', 'paper', 'peer_review']
```

### Scenario 2: Interrupt with Reasoning
```
🔔 Sending interrupt: 'The literature review missed key papers on neural architecture search'

🔶 [interrupt_reasoner] Analyzed interrupt: The researcher is providing important feedback about missing references. Action: defer (save for later revision)

📊 LLM Decision: defer - The feedback is valuable but doesn't require immediate action. Save for the revision phase.

📝 Deferred todos: 1 items
   First todo: The literature review missed key papers on neural architecture search. Please include them.
```

### Scenario 3: Checkpointing
```
⏸️  Pausing workflow after methodology design...

📦 Checkpoints created: 3
   - checkpoint_001: literature_review
   - checkpoint_002: generate_hypotheses
   - checkpoint_003: design_methodology

▶️  Resuming from checkpoint: checkpoint_003
✅ Restored state keys: ['topic', 'literature_review', 'hypotheses', 'methodology']
```

### Scenario 4: Immediate Cancellation
```
🛑 Cancelling workflow immediately (ESC)...

⚠️  Cancellation event received: ⚠️ Execution cancelled during node 'write_paper'

📊 Session state after cancel:
   - Cancelled: True
   - Cancelled at node: write_paper
   - Can resume: True
   - Partial state saved: True
   - Partial state keys: ['topic', 'literature_review', 'hypotheses', 'methodology', 'analysis_results']
```

---

## Key Takeaways

1. **Observability**: Developers control event format via callbacks (no hardcoded strings)
2. **Interrupt Reasoning**: LLM analyzes context and decides best action
3. **Flexible Timings**: BEFORE (validate), AFTER (correct), BOTH (comprehensive)
4. **Immediate Cancel**: TRUE immediate interrupt (cancels during execution, not just between nodes)
5. **State Preservation**: All cancellation paths save partial state for resume
6. **Extensible Actions**: continue, rerun, pause, defer, skip (+ custom actions)

---

## Next Steps

**Try modifying the example**:
1. Add your own custom callback with different observability
2. Create custom interrupt actions via `InterruptReasonerConfig.custom_actions`
3. Experiment with different interrupt timings (BEFORE/AFTER/BOTH)
4. Test resume from checkpoint after cancellation
5. Build your own workflow with conditional routing

**Explore the codebase**:
- `src/google/adk/agents/graph/graph_agent.py` - Core orchestration
- `src/google/adk/agents/graph/interrupt_reasoner.py` - LLM reasoning
- `src/google/adk/agents/graph/callbacks.py` - Callback infrastructure
- `src/google/adk/agents/graph/interrupt_service.py` - Interrupt management
- `src/google/adk/checkpoints/checkpoint_service.py` - Checkpoint management

**Questions?** See the design docs or run the tests to understand the implementation.
