# BranchContext Fix for Parallel Agent Event Visibility (GitHub Issue #3470)

## Problem Statement

Parallel agents in subsequent stages of a Sequential agent couldn't see outputs from previous parallel stages due to broken string-based branch filtering.

**Example that was broken:**
```python
# Sequential[Parallel1[A,B,C], Parallel2[D,E,F]]
# Agents D, E, F could NOT see outputs from A, B, C
```

### Root Cause

The old string-based branch system used prefix matching:
- Parallel1 agents got branches like `"0.0"`, `"0.1"`, `"0.2"`
- Parallel2 agents got branches like `"1.0"`, `"1.1"`, `"1.2"`
- `"1.0".startswith("0.0")` → `False` ❌

This broke event visibility in complex agent architectures.

## Solution: Token-Set Based Branch Tracking

Replaced string branches with **BranchContext** - an immutable, token-set based provenance tracking system.

### Key Concepts

1. **Fork**: Create N child contexts, each with a unique token
   ```python
   parent = BranchContext()  # tokens = {}
   children = parent.fork(3)  # [{1}, {2}, {3}]
   ```

2. **Join**: Merge child contexts back together
   ```python
   joined = parent.join(children)  # tokens = {1, 2, 3}
   ```

3. **Visibility**: Check using subset relationships
   ```python
   event_ctx.can_see(invocation_ctx)  # event_ctx.tokens ⊆ invocation_ctx.tokens
   ```

### How It Works

**Sequential[Parallel1[A,B,C], Parallel2[D,E,F]]:**

1. Root Sequential starts with `BranchContext()` (empty `{}`)
2. Parallel1 forks: A gets `{1}`, B gets `{2}`, C gets `{3}`
3. Parallel1 joins: context becomes `{1,2,3}`
4. Parallel2 forks from `{1,2,3}`: D gets `{1,2,3,4}`, E gets `{1,2,3,5}`, F gets `{1,2,3,6}`
5. **D can see A** because `{1} ⊆ {1,2,3,4}` ✅

## Files Modified

### Core Implementation

1. **`src/google/adk/agents/branch_context.py`** (NEW - 184 lines)
   - `TokenFactory`: Thread-safe token generation
   - `BranchContext`: Immutable Pydantic model with fork/join/can_see operations

2. **`src/google/adk/events/event.py`**
   - Changed `branch: Optional[str]` → `branch: Optional[BranchContext]`

3. **`src/google/adk/agents/invocation_context.py`**
   - Changed `branch: Optional[str]` → `branch: Optional[BranchContext]`
   - Updated `_get_events()` to use `can_see()` instead of string matching

4. **`src/google/adk/agents/parallel_agent.py`** (CRITICAL FIX)
   - Replaced string concatenation with `fork()` and `join()`
   - **MAJOR BUG FIX**: Track sub-agent contexts to collect final branches
   - Key logic:
     ```python
     parent_branch = ctx.branch or BranchContext()
     child_branches = parent_branch.fork(len(self.sub_agents))
     
     # Create contexts and track them
     sub_agent_contexts = []
     for i, sub_agent in enumerate(self.sub_agents):
       sub_agent_ctx = ctx.model_copy()
       sub_agent_ctx.branch = child_branches[i]
       sub_agent_contexts.append(sub_agent_ctx)
       agent_runs.append(sub_agent.run_async(sub_agent_ctx))
     
     # ... run agents ...
     
     # Join using FINAL branches (sub-agents may have modified them)
     final_child_branches = [sac.branch for sac in sub_agent_contexts]
     joined_branch = parent_branch.join(final_child_branches)
     ctx.branch = joined_branch
     ```
   - **Why this matters**: In nested parallel architectures, inner ParallelAgents modify their branch contexts (fork/join). The outer ParallelAgent must use these modified branches when joining, not the original forked branches, otherwise nested tokens are lost.

5. **`src/google/adk/agents/base_agent.py`**
   - Added branch propagation after `_run_async_impl` completes:
     ```python
     if ctx.branch != parent_context.branch:
       parent_context.branch = ctx.branch
     ```
   - This ensures joined branches propagate up to parent agents

6. **`src/google/adk/flows/llm_flows/contents.py`**
   - Replaced `invocation_branch.startswith(event.branch)` with `invocation_branch.can_see(event.branch)`

7. **`src/google/adk/agents/callback_context.py`**
   - Updated `_branch_ctx` field type

### Supporting Changes

- Updated all Event creation sites to include `branch` parameter
- Updated `base_llm_flow.py`, `transcription_manager.py`, `audio_cache_manager.py` for branch propagation

## Tests

### Unit Tests (21 tests - ALL PASSING)

**`tests/unittests/agents/test_branch_context.py`:**
- Core BranchContext operations (fork, join, can_see)
- Thread safety
- Pydantic serialization
- GitHub issue #3470 scenarios

### Integration Tests (2 tests - BOTH PASSING) ✨

**`tests/unittests/agents/test_github_issue_3470.py`:**

1. **`test_nested_parallel_reduce_architecture`**: Tests the complex nested architecture
   ```
   Sequential1 = Parallel[A, B, C] -> Reducer1
   Sequential2 = Parallel[D, E, F] -> Reducer2
   Final = Parallel[Sequential1, Sequential2] -> Reducer3
   ```
   
   **Token Flow (CORRECT):**
   - Alice={1,3}, Bob={1,4}, Charlie={1,5}
   - Reducer1={1,3,4,5} ✓ sees A, B, C
   - David={2,6}, Eve={2,7}, Frank={2,8}
   - Reducer2={2,6,7,8} ✓ sees D, E, F
   - Final_Reducer={1,2,3,4,5,6,7,8} ✓ sees both reducers AND all nested agents
   
   **This test revealed the critical bug**: Original implementation had Final_Reducer={1,2} only, missing all nested tokens.

2. **`test_sequence_of_parallel_agents`**: Tests sequential parallel groups
   ```
   Sequential[Parallel1[A,B,C], Parallel2[D,E,F], Parallel3[G,H,I]]
   ```
   
   **Token Flow (CORRECT):**
   - Parallel1: A={9}, B={10}, C={11}, joins to {9,10,11}
   - Parallel2 forks from {9,10,11}: D={9,10,11,12}, E={9,10,11,13}, F={9,10,11,14}
   - Parallel3 forks from joined: G={9,10,11,12,13,14,15}, ...
   - Each subsequent parallel group can see all previous groups ✓

### Regression Tests

**All 367 existing agent tests PASS** ✅ (was 365, now includes 2 new integration tests)

## Benefits

1. **Correctness**: Fixes event visibility in complex agent architectures
2. **Mathematical Rigor**: Token-set semantics are well-defined and provably correct
3. **Performance**: Set operations (subset check) are O(n) where n is number of tokens
4. **Immutability**: BranchContext is frozen, preventing accidental mutations
5. **Thread-Safe**: TokenFactory uses threading.Lock for safe parallel execution
6. **Serializable**: Pydantic model supports JSON serialization

## Migration Notes

### For ADK Users

No breaking changes for simple agent usage. Complex architectures automatically benefit from the fix.

### For ADK Developers

- Branch is no longer a string - use `BranchContext` methods
- Don't use string operations on branches
- Use `ctx.branch.can_see(event.branch)` for visibility checks

## Future Improvements

1. Add branch visualization tools for debugging
2. Optimize token storage for very deep agent hierarchies
3. Add branch pruning for completed sub-trees

## Related Issues

- GitHub Issue #3470: "Parallel agents in sequential stages cannot see previous outputs"
- **Two failing architectures identified in the issue - both now fixed:**
  1. **Nested Parallel + Reduce**: `Sequential[Parallel[A,B,C], Reducer1]` in parallel with `Sequential[Parallel[D,E,F], Reducer2]`, followed by Reducer3
  2. **Sequence of Parallels**: `Sequential[Parallel1[A,B,C], Parallel2[D,E,F], Parallel3[G,H,I]]`

## Key Discoveries

### Critical Bug Found: ParallelAgent Join Logic

While implementing integration tests for GitHub issue #3470, we discovered a critical bug in `ParallelAgent`:

**Problem:** When `ParallelAgent` executed nested parallel agents, it was joining using the **original forked branches** instead of the **final modified branches** from sub-agents. This caused token loss in nested architectures.

**Example:**
```python
# Nested architecture: Sequential[Parallel[A,B,C], Reducer] in parallel
Final_Parallel.fork() → {1}, {2}  # Two sequential groups
  Sequential1 (branch={1}):
    Parallel1.fork() → {1,3}, {1,4}, {1,5}  # Agents A, B, C
    Parallel1.join() → {1,3,4,5}  # Reducer1 gets this
  Sequential2 (branch={2}):
    Parallel2.fork() → {2,6}, {2,7}, {2,8}  # Agents D, E, F
    Parallel2.join() → {2,6,7,8}  # Reducer2 gets this

# BUG: Final_Parallel.join() used original {1}, {2}
# Result: Final_Reducer = {1,2} ❌ Cannot see nested tokens!

# FIX: Final_Parallel.join() uses final {1,3,4,5}, {2,6,7,8}
# Result: Final_Reducer = {1,2,3,4,5,6,7,8} ✅ Can see everything!
```

**Solution:** Track `sub_agent_contexts` and collect final branches: `[sac.branch for sac in sub_agent_contexts]`

This ensures proper token flow in nested parallel architectures, which are common in production agent systems.

## Credits

Implementation based on standard provenance tracking patterns from distributed systems and version control.
