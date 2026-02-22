## GraphAgent Parallel Execution & Rewind Features

Comprehensive demonstration of advanced GraphAgent features including parallel node execution, rewind integration, and edge cases.

---

## Features Demonstrated

### 1. **Parallel Node Execution** ✅

Execute independent nodes concurrently for better throughput.

**Join Strategies:**
- `WAIT_ALL`: Wait for all nodes to complete (default)
- `WAIT_ANY`: Proceed when first node completes (race condition)
- `WAIT_N`: Wait for N nodes to complete (e.g., 2 out of 3)

**Error Policies:**
- `FAIL_FAST`: Cancel all on first error
- `CONTINUE`: Continue others on error
- `COLLECT`: Collect all errors

**Code Example:**
```python
# Create parallel group
graph.add_parallel_group(
    "fetch_group",
    ParallelNodeGroup(
        nodes=["fetch_users", "fetch_products", "fetch_orders"],
        join_strategy=JoinStrategy.WAIT_ALL,
        error_policy=ErrorPolicy.FAIL_FAST,
    ),
)

# All three nodes execute concurrently
graph.add_edge("validate", "fetch_users")
graph.add_edge("validate", "fetch_products")
graph.add_edge("validate", "fetch_orders")
```

---

### 2. **Rewind Integration with Parallel Workflows** ✅

Rewind to any node, even those that trigger parallel execution.

**How It Works:**
- GraphAgent tracks invocation IDs per node
- `rewind_to_node()` restores session state to before node execution
- Re-execution from rewind point re-runs parallel groups

**Code Example:**
```python
# Execute workflow
async for event in runner.run_async(...):
    pass

# Rewind to a node that triggers parallel execution
await graph.rewind_to_node(
    session_service,
    app_name="my_app",
    user_id="user1",
    session_id="session1",
    node_name="fetch_users",  # Part of parallel group
    invocation_index=-1,      # Last invocation
)

# Re-execute - parallel group runs again
async for event in runner.run_async(...):
    pass
```

---

### 3. **Checkpointing with Parallel Execution** ✅

Checkpoints capture state after parallel branches complete.

**Architecture:**
- Parallel branches have isolated state during execution
- After all branches complete, state is merged
- Checkpoint created after merge includes all results

**Code Example:**
```python
# Enable checkpointing
graph = GraphAgent(
    name="workflow",
    checkpointing=True,
)

# Checkpoints created automatically after each node
# Including after parallel groups complete
```

---

### 4. **Interrupts During Parallel Execution** ✅

Interrupts can cancel all parallel branches immediately.

**Behavior:**
- InterruptService can mark session as cancelled
- GraphAgent checks for cancellation between events
- All parallel branches stop immediately
- Partial state preserved for potential resume

**Code Example:**
```python
# During parallel execution, send interrupt
await interrupt_service.send_interrupt(
    session_id=session.id,
    text="User requested abort",
    action="continue",  # Or "pause", "rerun", etc.
)

# GraphAgent detects cancellation, stops all branches
# State saved: {graph_cancelled: true, ...}
```

---

## Architectural Considerations

### State Isolation in Parallel Branches

**Problem:** How to prevent race conditions when multiple nodes modify state?

**Solution:** Each parallel branch gets an **isolated copy** of the state.

```python
# Parallel execution pseudocode
for node_name in parallel_group.nodes:
    # Create isolated state copy
    branch_state = GraphState(
        data=state.data.copy()
    )

    # Execute node with isolated state
    execute_node(node, branch_state, ctx)

# After all complete, merge results
merged_state = merge(branch_states)
```

**Benefits:**
- No race conditions
- Deterministic behavior
- Branches can't interfere with each other

---

### Rewind with Merged States

**Question:** Can rewind work if parallel branches have been merged?

**Answer:** YES! Here's how:

1. **During Execution:**
   - Parallel branches have isolated state
   - Each branch emits events independently
   - Results merged after all complete

2. **Invocation Tracking:**
   - GraphAgent tracks invocation IDs per node
   - Parallel nodes each get their own invocation ID
   - These IDs persist in session state

3. **Rewind Process:**
   - `rewind_to_node()` identifies the invocation ID
   - Uses `Runner.rewind_async()` to restore session state
   - State reverted to BEFORE node execution
   - Re-execution re-runs parallel group from scratch

**Example:**
```python
# Execution creates invocations:
# {
#   "validate": ["inv_1"],
#   "fetch_users": ["inv_2"],
#   "fetch_products": ["inv_3"],
#   "aggregate": ["inv_4"]
# }

# Rewind to fetch_users (inv_2)
await graph.rewind_to_node(..., node_name="fetch_users", invocation_index=-1)

# State restored to BEFORE inv_2
# Re-execution will:
# 1. Run fetch_users (new invocation)
# 2. Run fetch_products in parallel (new invocation)
# 3. Run aggregate (new invocation)
```

---

### Session State Communication

**Question:** What if session state is not communicated between parallel branches?

**Answer:** This is BY DESIGN for safety!

**Rationale:**
- Parallel branches should be **independent**
- Shared mutable state leads to race conditions
- Isolation ensures deterministic results

**Communication Patterns:**

1. **Before Parallel Execution:**
   ```python
   # All branches start with same initial state
   state = GraphState(data={"shared_input": "value"})
   ```

2. **After Parallel Execution:**
   ```python
   # Merge results using StateReducer
   merged_state = reducer(branch_results)
   ```

3. **If Communication Needed:**
   - Use separate coordination node
   - Don't put nodes in parallel group
   - Use sequential edges with conditions

---

### Interrupts During Parallel Execution

**Question:** What happens if we interrupt during parallel execution?

**Answer:** Clean cancellation with state preservation.

**Flow:**
1. User sends interrupt (or timeout triggers)
2. `InterruptService` marks session as inactive
3. GraphAgent checks `is_active()` between events
4. All parallel branches detect cancellation
5. Tasks cancelled via `task.cancel()`
6. Partial state saved to session:
   ```python
   {
       "graph_cancelled": True,
       "graph_cancelled_at_node": "fetch_users",
       "graph_iteration": 2,
       "graph_data": {...},  # Partial domain data
       "graph_can_resume": True
   }
   ```

**Resume After Interrupt:**
```python
# State preserved, can resume from checkpoint
# Or restart from beginning
# Or rewind to specific point
```

---

## Scenarios

### Scenario 1: Parallel Execution (WAIT_ALL)
Fetch data from 3 sources concurrently, wait for all.

**Workflow:**
```
validate → (fetch_users || fetch_products || fetch_orders) → aggregate
```

### Scenario 2: Parallel Execution (WAIT_ANY)
Race 3 data sources, proceed with first to complete.

**Workflow:**
```
validate → (fetch_cache || fetch_db || fetch_api) → transform
```

### Scenario 3: Parallel Execution (WAIT_N)
Run 3 ML models, proceed when 2 out of 3 complete.

**Workflow:**
```
validate → (model1 || model2 || model3) → aggregate
```

### Scenario 4: Rewind with Parallel Execution
Execute workflow, rewind to parallel group, re-execute.

**Demonstrates:**
- Invocation tracking across parallel nodes
- Rewind restores state before parallel execution
- Re-execution runs parallel group again

### Scenario 5: Checkpointing with Parallel Execution
Enable checkpointing, execute workflow with parallel nodes.

**Demonstrates:**
- Checkpoints created after each node
- Parallel group checkpoint captures merged state
- Resume from checkpoint works correctly

### Scenario 6: Interrupts During Parallel Execution
Show interrupt behavior and considerations.

**Demonstrates:**
- How interrupts cancel all parallel branches
- State preservation on cancellation
- Architecture for resume capability

### Scenario 7: State Isolation in Parallel Branches
Parallel branches modify same state key, show isolation.

**Demonstrates:**
- Each branch has isolated state
- No race conditions
- Deterministic results

---

## Running the Examples

```bash
# Run all scenarios
python -m contributing.samples.graph_agent_parallel_features.agent

# Or run from the adk-python directory
cd /path/to/adk-python
source venv/bin/activate
python -m contributing.samples.graph_agent_parallel_features.agent
```

---

## Expected Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  GraphAgent Parallel Execution & Rewind Features - Comprehensive Demo       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
SCENARIO 1: Parallel Execution with WAIT_ALL
================================================================================

📊 Executing workflow with parallel fetch operations...
   Strategy: WAIT_ALL (wait for all 3 fetches to complete)

  ✅ Data validation passed
  ✅ Fetched 3 records from users_db
  ✅ Fetched 3 records from products_db
  ✅ Fetched 3 records from orders_db
  ✅ Aggregated results from all sources

✅ Scenario 1 complete: 5 events emitted
   Note: All 3 fetch operations ran concurrently!

... (more scenarios) ...

================================================================================
✅ ALL SCENARIOS COMPLETE
================================================================================

Key Takeaways:
1. Parallel execution works with WAIT_ALL, WAIT_ANY, WAIT_N strategies
2. Rewind integration works - can rewind to nodes that trigger parallel groups
3. Checkpointing captures state after parallel branches complete
4. Interrupts can cancel parallel execution (state preserved)
5. Parallel branches have isolated state (no race conditions)

Architectural Answers:
- Q: Can rewind work with parallel execution?
  A: YES! Rewind restores to before node execution, re-runs parallel group
- Q: What about session state communication?
  A: Branches are isolated during execution, merged after completion
- Q: What if we interrupt during parallel execution?
  A: All branches cancelled, partial state saved for resume
```

---

## Architecture Diagrams

### Parallel Execution Flow

```
┌─────────────┐
│  validate   │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       │              │              │
       ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ fetch_users  │ │fetch_products│ │ fetch_orders │
│  (isolated)  │ │  (isolated)  │ │  (isolated)  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │  aggregate   │
              │ (merged state)│
              └──────────────┘
```

### Rewind Flow

```
1. Initial Execution:
   validate → parallel_group → aggregate

2. Rewind to parallel_group:
   [Session State] ← Restore ← Before parallel_group

3. Re-execution:
   parallel_group → aggregate
   (New invocations created)
```

### State Isolation

```
Main State:
┌────────────────────────────────┐
│ data: {input: "value"}         │
└────────────────────────────────┘
                │
                │ copy()
                ├─────────────┬─────────────┬─────────────┐
                ▼             ▼             ▼             ▼
         ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
         │ Branch 1 │   │ Branch 2 │   │ Branch 3 │   │  ...     │
         │ (isolated)│   │ (isolated)│   │ (isolated)│   │          │
         └─────┬────┘   └─────┬────┘   └─────┬────┘   └─────┬────┘
               │              │              │              │
               └──────────────┴──────────────┴──────────────┘
                                    │
                                    ▼
                              Merge Results
                         ┌────────────────────┐
                         │   Merged State     │
                         └────────────────────┘
```

---

## Performance Considerations

### Parallel vs Sequential

**Sequential Execution:**
```
Total time = sum(all node execution times)
Example: 100ms + 150ms + 200ms = 450ms
```

**Parallel Execution (WAIT_ALL):**
```
Total time = max(all node execution times)
Example: max(100ms, 150ms, 200ms) = 200ms
Speedup: 2.25x faster
```

**Parallel Execution (WAIT_ANY):**
```
Total time = min(all node execution times)
Example: min(100ms, 150ms, 200ms) = 100ms
Speedup: 4.5x faster (but only uses first result)
```

---

## Best Practices

1. **Use Parallel Execution When:**
   - Nodes are independent (no data dependencies)
   - Operations are I/O bound (API calls, DB queries)
   - Order doesn't matter

2. **Avoid Parallel Execution When:**
   - Nodes have sequential dependencies
   - Order matters for correctness
   - Shared mutable resources (use locks)

3. **Join Strategy Selection:**
   - `WAIT_ALL`: When you need all results
   - `WAIT_ANY`: When any result is acceptable (cache/DB/API fallback)
   - `WAIT_N`: When you need quorum (ML ensemble, consensus)

4. **Error Handling:**
   - `FAIL_FAST`: When any failure invalidates the entire operation
   - `CONTINUE`: When partial results are acceptable
   - `COLLECT`: When you need to analyze all failures

5. **State Management:**
   - Don't rely on shared mutable state in parallel branches
   - Use isolated state copies (automatic)
   - Merge results after completion

---

## Related Examples

- `graph_agent_basic` - Basic GraphAgent workflow
- `graph_agent_advanced` - Interrupts, checkpointing, callbacks
- `graph_agent_builder` - Graph construction patterns

---

## References

- GraphAgent: `src/google/adk/agents/graph/graph_agent.py`
- Parallel Execution: `src/google/adk/agents/graph/parallel.py`
- Rewind Integration: `src/google/adk/agents/graph/graph_agent.py:rewind_to_node()`
- Tests: `tests/unittests/agents/test_graph_parallel.py`
