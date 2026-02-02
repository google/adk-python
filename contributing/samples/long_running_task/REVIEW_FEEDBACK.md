# Design Document Review: Durable Session Persistence for Long-Horizon ADK Agents

**Reviewer:** Claude Code
**Date:** 2026-02-01
**Document:** `long_running_task_design.md`

---

## Executive Summary

The design document is **well-structured and comprehensive**, covering a real problem with a thorough technical approach. However, there are **critical accuracy issues** regarding ADK's current capabilities that must be addressed before the document can be considered accurate for review.

**Overall Assessment:** Good foundation, requires significant revisions to accurately reflect ADK's existing resumability features.

---

## 1. Reference Validation

### External URLs (7 total) - ALL VALID

| # | URL | Status | Notes |
|---|-----|--------|-------|
| 1 | LangGraph durable-execution | VALID | Content matches claims |
| 2 | LangGraph persistence | VALID | Checkpointing docs |
| 3 | LangGraph overview | VALID | Framework intro |
| 4 | LangGraph checkpoints reference | VALID | API docs |
| 5 | Deep Agents overview | VALID | LangChain library |
| 6 | Deep Agents long-term memory | VALID | Memory patterns |
| 7 | Anthropic harnesses article | VALID | Published 2025-11-26 |

---

## 2. CRITICAL ISSUE: ADK Already Has Resumability

### Problem Statement Inaccuracy

The document states (Section 2):
> "Current ADK sessions are optimized for synchronous 'serving' patterns... state is ephemeral... background execution is not a first-class runtime mode"

**This is inaccurate.** ADK already has an experimental resumability feature:

```python
# src/google/adk/apps/app.py lines 42-58
@experimental
class ResumabilityConfig(BaseModel):
  """The "resumability" in ADK refers to the ability to:
  1. pause an invocation upon a long-running function call.
  2. resume an invocation from the last event, if it's paused or failed midway
  through.
  """
  is_resumable: bool = False
```

### Existing ADK Capabilities Not Mentioned

| Capability | Location | Status |
|------------|----------|--------|
| `ResumabilityConfig` | `src/google/adk/apps/app.py:42-58` | Experimental |
| `should_pause_invocation()` | `src/google/adk/agents/invocation_context.py:355-389` | Implemented |
| `long_running_tool_ids` | `src/google/adk/events/event.py` | Implemented |
| Resume from last event | `src/google/adk/runners.py:1294` | Implemented |

### Required Fix

**The document must:**
1. Acknowledge existing `ResumabilityConfig` and pause/resume capability
2. Clearly articulate how this proposal **extends** existing features vs. replacing them
3. Update Section 2 (Problem Statement) to reflect actual gaps (e.g., durable cross-process persistence, BigQuery-based audit, external event triggers)

---

## 3. Technical Review

### 3.1 SQL Schema (Appendix B) - VALID WITH MINOR ISSUES

**Strengths:**
- Proper partitioning strategy (`PARTITION BY DATE`)
- Sensible clustering choices
- JSON columns for flexibility

**Issues:**

1. **Missing primary key constraint on checkpoints:**
   ```sql
   -- Should add:
   PRIMARY KEY (session_id, checkpoint_seq)
   ```

2. **events table lacks PRIMARY KEY:**
   ```sql
   -- Consider adding:
   PRIMARY KEY (event_id)  -- or composite key
   ```

3. **View `v_latest_checkpoint` uses ARRAY_AGG with OFFSET(0):**
   - This is valid but will error if no checkpoints exist
   - Consider `SAFE_OFFSET(0)` or handle NULL case

### 3.2 Python Code Snippets - MOSTLY VALID

**Section 7.1 `write_checkpoint()`:**
- Logic is sound (two-phase commit pattern)
- Consider adding error handling for partial failures

**Section 7.2 `reconcile_on_resume()`:**
- Good idempotency pattern
- Missing: what happens if `bq.get_job()` fails?

### 3.3 Leasing Approach (Section 7.3) - REASONABLE

The BQ-based optimistic lease is correctly noted as best-effort. The suggestion to use Firestore/Spanner for stronger guarantees is appropriate.

**Suggestion:** Add a concrete example of when to use each backend (BQ vs Firestore).

---

## 4. Architecture Feedback

### 4.1 Strengths

1. **Clear separation of control plane (BQ) vs data plane (GCS)** - follows Google best practices
2. **Logical checkpointing over heap snapshots** - pragmatic and maintainable
3. **Two-phase commit pattern** - ensures atomic visibility
4. **Authoritative reconciliation** - critical for BigQuery job scenarios
5. **Good competitive analysis** (Section 14)

### 4.2 Gaps / Missing Considerations

| Gap | Impact | Suggested Action |
|-----|--------|------------------|
| No mention of existing `ResumabilityConfig` | Misleading problem statement | Add section on existing capability |
| No cost estimates for BQ storage/queries | Budget planning | Add rough estimates |
| No mention of BQ quota limits | Operational risk | Document relevant quotas |
| Checkpoint versioning migration strategy | Future maintenance | Expand Section 16.2 |
| No monitoring/alerting design | Operability | Add observability section |
| No rollback strategy | Safety | Document how to rollback |

### 4.3 API Contract Review

The proposed `CheckpointableAgentState` interface is clean:

```python
class CheckpointableAgentState:
    def export_state(self) -> dict: ...
    def import_state(self, state: dict) -> None: ...
```

**Suggestion:** Consider alignment with existing ADK patterns:
- Existing `BaseAgentState` in `src/google/adk/agents/base_agent.py`
- Existing state patterns in `src/google/adk/sessions/state.py`

---

## 5. Specific Line-by-Line Feedback

### Section 0 (Executive Summary)
- Line 14: "12-minute barrier" - should cite source or clarify this is environment-specific
- Line 28: Cost estimate "< $0.01/session-day paused" - show calculation

### Section 2 (Problem Statement)
- **Major revision needed** - must acknowledge existing resumability

### Section 4.1 (States)
- Consider: should PAUSED be a first-class `Session.status` field or remain at `InvocationContext` level?

### Section 8 (API Extensions)
- `checkpoint_policy` options are good, but:
  - What triggers `superstep`?
  - How does `manual` interact with `long_running_tool_ids`?

### Section 13 (Moltbot Alignment)
- Moltbot reference is useful context
- Consider adding link/citation if public

### Section 18 (Open Questions)
- Good list, but add: "How does this integrate with existing `ResumabilityConfig`?"

---

## 6. Recommended Document Changes

### High Priority (Must Fix)

1. **Add Section 1.3: "Existing ADK Resumability"**
   - Document current `ResumabilityConfig` capability
   - Explain limitations this design addresses
   - Position proposal as extension, not replacement

2. **Revise Section 2 (Problem Statement)**
   - Remove/qualify claims about ADK lacking pause/resume
   - Focus on actual gaps: cross-process durability, external event triggers, enterprise audit

3. **Add explicit integration plan**
   - How does `CheckpointableAgentState` relate to `BaseAgentState`?
   - Migration path from current resumability to new design

### Medium Priority

4. Add cost estimation section
5. Add monitoring/observability design
6. Add rollback/recovery procedures
7. Fix SQL schema issues (PKs)

### Low Priority

8. Add Moltbot citation if available
9. Add BQ quota documentation links
10. Consider adding architecture diagram (beyond Mermaid sequence)

---

## 7. Summary Table

| Category | Status | Details |
|----------|--------|---------|
| External URLs | VALID | All 7 references work |
| SQL Syntax | VALID with issues | Missing PKs, edge cases |
| Python Code | VALID | Sound patterns |
| Problem Statement | INACCURATE | Ignores existing resumability |
| Architecture | SOUND | Good Google-scale patterns |
| Completeness | GAPS | Missing cost, monitoring, rollback |

---

## 8. Conclusion

This is a **solid technical design** for extending ADK's capabilities for long-running BigQuery workloads. The core architecture (BQ control plane, GCS data plane, two-phase commit, authoritative reconciliation) is well-reasoned.

**However, the document cannot be approved in its current form** because it misrepresents ADK's existing capabilities. Once the existing `ResumabilityConfig` is acknowledged and the document is repositioned as an extension rather than a new capability, it will be ready for technical review.

**Recommended Next Steps:**
1. Revise document to acknowledge existing resumability
2. Add cost/monitoring sections
3. Fix SQL schema issues
4. Re-submit for review

---

*Review generated by Claude Code on 2026-02-01*
