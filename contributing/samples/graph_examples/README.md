# GraphAgent Examples - All Features

Comprehensive collection of small, focused examples demonstrating every GraphAgent feature.

---

## API Overview

GraphAgent supports both **explicit** and **convenience** APIs for building workflows:

### Convenience API (Recommended)

```python
# Fluent chaining pattern
graph = (
    GraphAgent(name="workflow")
    .add_node("validate", agent=validator)
    .add_node("process", agent=processor)
    .add_edge("validate", "process")  # Positional args
    .add_edge("process", "output", priority=10)  # With priority
    .set_start("validate")
    .set_end("output")
)

# Or step-by-step
graph = GraphAgent(name="workflow")
graph.add_node("step1", agent=agent1)
graph.add_node("step2", function=custom_function)
graph.add_edge("step1", "step2")
```

### Explicit API (Also Supported)

```python
# Using GraphNode and EdgeCondition explicitly
graph.add_node(GraphNode(name="step1", agent=agent1))
graph.add_edge("source", EdgeCondition(
    target_node="target",
    priority=10,
    condition=lambda s: s.data.get("valid")
))
```

### Key API Features

- **add_node()**: Convenience syntax `add_node("name", agent=...)` or explicit `add_node(GraphNode(...))`
- **add_edge()**: Positional `add_edge("source", "target")` or keyword `add_edge(source_node="a", target_node="b")`
- **EdgeCondition**: Support for `add_edge("src", EdgeCondition(target_node="tgt", condition=...))`
- **checkpoint_service**: Optional parameter, not required
- **Fluent chaining**: All builder methods return self for method chaining

---

## Quick Start

Run any example with:
```bash
cd /path/to/adk-python
source venv/bin/activate
python -m contributing.samples.graph_examples.<example_name>.agent
```

### Deterministic vs LLM Modes

Examples support two execution modes:

**Deterministic Mode (default)** - Uses BaseAgent subclasses with deterministic outputs. No API keys required.
```bash
python -m contributing.samples.graph_examples.01_basic.agent
```

**LLM Mode (optional)** - Uses real Gemini LLM endpoints. Requires API credentials.
```bash
# Via command-line flag
python -m contributing.samples.graph_examples.01_basic.agent --use-llm

# Via environment variable
USE_LLM=1 python -m contributing.samples.graph_examples.01_basic.agent
```

**Note:** LLM mode is only available for simple examples (01_basic, 02_conditional_routing, etc.). Examples that require precise state management (05_interrupts, parallel execution) use deterministic agents to demonstrate graph mechanics reliably.

---

## Examples Overview

### 🟢 Core Features

#### **01_basic** - Basic GraphAgent Workflow
Simple directed graph with nodes and edges.
```bash
python -m contributing.samples.graph_examples.01_basic.agent
```
**Demonstrates:**
- Creating a graph with fluent API
- Adding nodes with convenience syntax: `add_node("name", agent=...)`
- Adding edges with positional syntax: `add_edge("source", "target")`
- Executing workflow

---

#### **02_conditional_routing** - Conditional Routing
State-based routing decisions.
```bash
python -m contributing.samples.graph_examples.02_conditional_routing.agent
```
**Demonstrates:**
- Conditional edges with `condition` parameter
- EdgeCondition support: `add_edge("src", EdgeCondition(target_node="tgt", condition=...))`
- State-based decisions
- Multiple routing paths
- Priority-based routing with `priority` parameter

---

#### **03_cyclic_execution** - Cyclic Graph Execution
Loops and iteration control.
```bash
python -m contributing.samples.graph_examples.03_cyclic_execution.agent
```
**Demonstrates:**
- Cyclic graphs with back-edges
- Writing routing signals via state_delta (ADK pattern)
- GraphState.data auto-sync from session.state
- max_iterations guard

**Key Pattern:**
Agents write routing signals via `Event.actions.state_delta`. GraphAgent automatically syncs `session.state` into `GraphState.data` before edge evaluation — no `output_mapper` needed.

---

#### **04_checkpointing** - Checkpointing & Resume
Automatic state persistence.
```bash
python -m contributing.samples.graph_examples.04_checkpointing.agent
```
**Demonstrates:**
- Automatic checkpointing with optional `checkpoint_service` parameter
- State persistence
- Checkpoint metadata
- Execution path tracking
- Resume from checkpoint capability

---

#### **05_interrupts_basic** - Basic Interrupts
Human-in-the-loop interrupts.
```bash
python -m contributing.samples.graph_examples.05_interrupts_basic.agent
```
**Demonstrates:**
- All 8 interrupt actions: continue, rerun, pause, go_back, skip, defer, update_state, change_condition
- Concurrent injection via asyncio.create_task
- SlowNode (2 sub-steps × 1s) timing pattern
- AFTER interrupt check behavior (queued during execution, consumed after node completes)

---

#### **06_interrupts_reasoning** - Interrupt with Reasoning
Condition-based action selection.
```bash
python -m contributing.samples.graph_examples.06_interrupts_reasoning.agent
```
**Demonstrates:**
- Interrupt with condition evaluation
- Automated action selection based on state
- Draft-review workflow with interrupt points

---

#### **07_callbacks** - Node Callbacks
Lifecycle hooks for nodes.
```bash
python -m contributing.samples.graph_examples.07_callbacks.agent
```
**Demonstrates:**
- `before_node_callback` - executed before node runs
- `after_node_callback` - executed after node completes
- Timing and performance tracking
- Telemetry integration patterns

---

#### **08_rewind** - Rewind Integration
Time-travel debugging.
```bash
python -m contributing.samples.graph_examples.08_rewind.agent
```
**Demonstrates:**
- Invocation tracking
- Rewinding to specific node
- State restoration
- Re-execution after rewind

---

### ⚡ Parallel Execution

#### **09_parallel_wait_all** - Parallel Execution (WAIT_ALL)
Concurrent node execution, wait for all.
```bash
python -m contributing.samples.graph_examples.09_parallel_wait_all.agent
```
**Demonstrates:**
- Parallel node execution
- WAIT_ALL join strategy
- Speedup vs sequential (2.25x)
- Event streaming from parallel nodes

**Output (example):**
```
✅ Fetched data from products_db
✅ Fetched data from users_db
✅ Fetched data from orders_db

All three fetched in parallel.
```

---

#### **10_parallel_wait_any** - Parallel Execution (WAIT_ANY)
Race condition, first-to-complete wins.
```bash
python -m contributing.samples.graph_examples.10_parallel_wait_any.agent
```
**Demonstrates:**
- Racing multiple data sources
- WAIT_ANY join strategy
- Automatic cancellation of slower nodes
- Cache-DB-API fallback pattern

**Output (example):**
```
✅ Data from CACHE

Winner: Cache
Cancelled: Database, API
```

---

#### **11_parallel_wait_n** - Parallel Execution (WAIT_N)
Continue after N of M complete.
```bash
python -m contributing.samples.graph_examples.11_parallel_wait_n.agent
```
**Demonstrates:**
- WAIT_N join strategy (e.g., 2 out of 3)
- ML model ensemble pattern
- Partial completion workflows
- Automatic cancellation of remaining nodes

---

#### **12_parallel_checkpointing** - Parallel + Checkpointing
State persistence across parallel execution.
```bash
python -m contributing.samples.graph_examples.12_parallel_checkpointing.agent
```
**Demonstrates:**
- Parallel execution with automatic checkpointing
- State recovery after interruption
- Checkpoint metadata tracking
- Resume from mid-parallel execution

---

#### **13_parallel_interrupts** - Parallel + Interrupts
Interrupt handling inside parallel branches.
```bash
python -m contributing.samples.graph_examples.13_parallel_interrupts.agent
```
**Demonstrates:**
- Interrupts within parallel node execution
- Branch-specific interrupt handling
- Pause/resume in parallel context
- Interrupt isolation across branches

---

### 🔗 Combined Features

#### **14_parallel_rewind** - Parallel Execution + Rewind
Rewind works with parallel workflows!
```bash
python -m contributing.samples.graph_examples.14_parallel_rewind.agent
```
**Demonstrates:**
- Parallel + Rewind integration
- Invocation tracking in parallel groups
- Re-execution of entire parallel group
- State consistency across rewind

**Key Insight:**
- Rewind to parallel node → entire parallel group re-executes
- All branches get new invocations
- Deterministic re-execution

---

#### **15_enhanced_routing** - Enhanced Routing Patterns
Priority, weighted, and fallback routing.
```bash
python -m contributing.samples.graph_examples.15_enhanced_routing.agent
```
**Demonstrates:**
- Priority-based routing (higher priority evaluated first)
- Weighted random selection (probabilistic routing)
- Fallback edges (priority=0 always matches)
- Three routing patterns in one example

---

## Feature Matrix

| Example | Parallel | Rewind | Checkpoints | Interrupts | Callbacks | Cyclic | Routing |
|---------|----------|--------|-------------|------------|-----------|--------|---------|
| 01_basic | - | - | - | - | - | - | Simple |
| 02_conditional_routing | - | - | - | - | - | - | Conditional |
| 03_cyclic_execution | - | - | - | - | - | ✅ | Conditional |
| 04_checkpointing | - | - | ✅ | - | - | - | Simple |
| 05_interrupts_basic | - | - | - | ✅ | - | - | Simple |
| 06_interrupts_reasoning | - | - | - | ✅ | - | - | Conditional |
| 07_callbacks | - | - | - | - | ✅ | - | Simple |
| 08_rewind | - | ✅ | - | - | - | - | Simple |
| 09_parallel_wait_all | ✅ | - | - | - | - | - | Parallel |
| 10_parallel_wait_any | ✅ | - | - | - | - | - | Parallel |
| 11_parallel_wait_n | ✅ | - | - | - | - | - | Parallel |
| 12_parallel_checkpointing | ✅ | - | ✅ | - | - | - | Parallel |
| 13_parallel_interrupts | ✅ | - | - | ✅ | - | - | Parallel |
| 14_parallel_rewind | ✅ | ✅ | - | - | - | - | Parallel |
| 15_enhanced_routing | - | - | - | - | - | - | Advanced |

---

## Architectural Insights

### Parallel Execution Architecture

```
┌─────────────┐
│   validate  │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       │              │              │
       ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  fetch_A     │ │  fetch_B     │ │  fetch_C     │
│  (isolated)  │ │  (isolated)  │ │  (isolated)  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │   aggregate  │
              │(merged state)│
              └──────────────┘
```

**Key Points:**
- Each branch has **isolated state** during execution
- No race conditions possible
- State **merged** after all branches complete
- Events **streamed** as branches complete (FIRST_COMPLETED)

---

### Rewind with Parallel Execution

```
1. Initial Execution:
   validate → (fetch_A || fetch_B || fetch_C) → aggregate

   Invocations created:
   - validate: ["inv_1"]
   - fetch_A: ["inv_2"]
   - fetch_B: ["inv_3"]
   - fetch_C: ["inv_4"]
   - aggregate: ["inv_5"]

2. Rewind to fetch_A (inv_2):
   Session state restored to BEFORE inv_2

3. Re-execution:
   (fetch_A || fetch_B || fetch_C) → aggregate

   New invocations:
   - fetch_A: ["inv_2", "inv_6"]
   - fetch_B: ["inv_3", "inv_7"]
   - fetch_C: ["inv_4", "inv_8"]
   - aggregate: ["inv_5", "inv_9"]
```

**Key Points:**
- Rewind works seamlessly with parallel groups
- Entire parallel group re-executes
- New invocations created on re-execution
- Deterministic behavior guaranteed

---

### State Isolation

**Problem:** Multiple nodes modifying same state → race conditions

**Solution:** Isolated state copies per branch

```python
# During parallel execution
for node in parallel_group.nodes:
    # Each branch gets ISOLATED copy
    branch_state = state.copy()

    # Modify branch state
    execute_node(node, branch_state)

# After all complete
merged_state = merge(all_branch_states)
```

**Benefits:**
- No race conditions
- Deterministic results
- Safe concurrent execution

---

## Performance Comparison

### Sequential vs Parallel (WAIT_ALL)

**Scenario:** Fetch from 3 sources (100ms, 150ms, 200ms each)

**Sequential:**
```
Total time = 100 + 150 + 200 = 450ms
```

**Parallel (WAIT_ALL):**
```
Total time = max(100, 150, 200) = 200ms
Speedup: 450ms / 200ms = 2.25x
```

**Parallel (WAIT_ANY):**
```
Total time = min(100, 150, 200) = 100ms
Speedup: 450ms / 100ms = 4.5x
```

---

## Common Patterns

### 1. Data Pipeline (WAIT_ALL)
Fetch data from multiple sources concurrently.
```python
ParallelNodeGroup(
    nodes=["fetch_users", "fetch_products", "fetch_orders"],
    join_strategy=JoinStrategy.WAIT_ALL
)
```

### 2. Cache-DB-API Fallback (WAIT_ANY)
Race multiple data sources, use fastest.
```python
ParallelNodeGroup(
    nodes=["from_cache", "from_db", "from_api"],
    join_strategy=JoinStrategy.WAIT_ANY
)
```

### 3. ML Model Ensemble (WAIT_N)
Run multiple models, proceed when N complete.
```python
ParallelNodeGroup(
    nodes=["model1", "model2", "model3"],
    join_strategy=JoinStrategy.WAIT_N,
    wait_n=2  # 2 out of 3
)
```

### 4. Interrupt-Driven Review
Human review after key nodes.
```python
InterruptConfig(
    mode=InterruptMode.AFTER,
    nodes=["draft", "review"]
)
```

### 5. Checkpoint-Resume Workflow
Long-running workflows with state persistence.
```python
GraphAgent(
    name="workflow",
    checkpoint_service=checkpoint_service  # Optional parameter
)
```

---

## Error Handling

### Parallel Error Policies

#### FAIL_FAST (default)
```python
ParallelNodeGroup(
    nodes=["task1", "task2", "task3"],
    error_policy=ErrorPolicy.FAIL_FAST
)
# One error → cancel all → raise exception
```

#### CONTINUE
```python
ParallelNodeGroup(
    nodes=["task1", "task2", "task3"],
    error_policy=ErrorPolicy.CONTINUE
)
# One error → continue others → log error
```

#### COLLECT
```python
ParallelNodeGroup(
    nodes=["task1", "task2", "task3"],
    error_policy=ErrorPolicy.COLLECT
)
# All errors → collect all → raise at end
```

---

## Testing

Run tests:
```bash
pytest tests/unittests/agents/test_graph_*.py -v
```

---

## Next Steps

1. **Try the examples** - Run each one to see features in action
2. **Modify examples** - Change parameters, add nodes, experiment
3. **Combine features** - Mix parallel + rewind + checkpoints
4. **Build your workflow** - Use patterns for your use case

---

## Related Samples: graph_agent_* (Complex Real-World Examples)

In addition to the numbered graph_examples, there are advanced samples at `contributing/samples/graph_agent_*`:

| Sample | Description | Pattern |
|--------|-------------|---------|
| **graph_agent_basic** | Basic research workflow | LLM-powered, `agents.py` + `agent.py` |
| **graph_agent_advanced** | Complex research paper workflow | Multi-phase with review loop |
| **graph_agent_react_pattern** | ReAct pattern (Reason + Act) | Thought-action-observation cycle |
| **graph_agent_multi_agent** | Multiple specialized agents | Delegation and collaboration |
| **graph_agent_dynamic_queue** | Dynamic node queueing | Runtime graph modification |
| **graph_agent_parallel_features** | Parallel feature demonstrations | Showcases parallel capabilities |
| **graph_agent_pattern_dynamic_node** | Dynamic node creation | Runtime node injection |
| **graph_agent_pattern_nested_graph** | Nested GraphAgents | Graph as node pattern |
| **graph_agent_pattern_parallel_group** | Parallel group pattern | Advanced parallel workflows |

**Key Differences from graph_examples**:
- **LLM-Required**: Use LlmAgent, require API credentials (no deterministic mode)
- **Structure**: Single `agent.py` with inline agent definitions
- **Complexity**: Multi-phase workflows, real-world use cases
- **Purpose**: Demonstrate production patterns, not individual features
- **Note**: Some samples have `__init__.py` for package structure

**Run Example**:
```bash
cd /path/to/adk-python
source venv/bin/activate
python -m contributing.samples.graph_agent_advanced.agent
```

---

## Support

Questions? Check:
- Examples: `contributing/samples/graph_examples/`
- Advanced: `contributing/samples/graph_agent_*/`
- Tests: `tests/unittests/agents/test_graph_*.py`
- Source: `src/google/adk/agents/graph/`
