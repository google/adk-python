# ADK Testing Style Guide

## Core Principles

- **Test through the public interface** — call what users call, assert what users see.
- **Test behavior, not implementation** — verify outcomes (outputs, side effects, errors), not internal mechanics.
- **Refactor-proof** — if an internal refactor preserves the same behavior, all tests should still pass.

## Rules

### 1. Test names describe the behavior, not the mechanism

```python
# Good — describes what the caller observes
def test_empty_queue_returns_none():
def test_retry_stops_after_max_attempts():
def test_missing_key_raises_key_error():

# Bad — describes implementation details
def test_deque_popleft_called():
def test_retry_counter_incremented():
def test_dict_getitem_raises():
```

### 2. Docstring: one-line summary, then setup/act/assert

The first line describes the expected behavior from the caller's
perspective. For complex tests (multi-step, multi-invocation),
follow with a structured breakdown of Setup, Act, and Assert.

```python
# Good — simple test, one-liner is enough
"""Getting from an empty cache returns the default value."""

# Good — complex test with structured breakdown
"""Partial FR re-runs nested Workflow, resolved child completes
while unresolved stays interrupted.

Setup: outer_wf → inner_wf → (child_a, child_b) → join.
  Both children interrupt on first run.
Act:
  - Run 2: resolve only child_a's FR.
  - Run 3: resolve child_b's FR.
Assert:
  - Run 2: child_a produces output, invocation still interrupted.
  - Run 3: child_b produces output, join completes, no interrupts.
"""

# Bad — restates the implementation
"""LRUCache._store.get returns sentinel when key missing."""
"""ThreadPool._accept_tasks flag checked in submit()."""
```

### 3. Each test covers one behavior

If a test checks multiple unrelated behaviors, split it. If you can't
describe the test in one sentence, it's testing too much.

```python
# Bad — tests capacity AND eviction AND default in one test
def test_cache_behavior():
    assert cache.size == 0
    assert cache.get('x') is None
    cache.put('a', 1)
    assert cache.size == 1

# Good — split into focused tests
def test_new_cache_is_empty():
    """A freshly created cache has no entries."""

def test_cache_evicts_oldest_when_full():
    """Adding to a full cache removes the least recently used entry."""
```

### 4. Don't test internal state

```python
# Bad — reaches into private attributes
assert pool._workers[0].is_alive
assert parser._state == 'HEADER'
assert isinstance(router._handler, _FastHandler)

# Good — tests through the public interface
assert pool.active_count == 1
assert parser.parse('data') == expected
assert router.route('/api') == handler
```

### 5. Use real components, mock only boundaries

ADK tests should use real implementations as much as possible
instead of mocking.

- Mock external dependencies: LLM APIs, cloud services, session stores
- Use real ADK components: BaseNode subclasses, Event, Context
- Mock InvocationContext when testing NodeRunner (it's a boundary)

### 6. Test fixtures should be minimal

Define the simplest possible setup that triggers the behavior:

```python
# Good — minimal fixture, one purpose
def make_user(role='viewer'):
    return User(name='test', email='t@t.com', role=role)

# Bad — kitchen-sink fixture with unrelated setup
def make_full_test_env():
    db = create_database()
    user = create_user_with_billing()
    setup_notifications()
    ...
```

### 7. Keep arrange logic close to the test

When a helper class or fixture is used by only one test, define it
inline inside the test function. This keeps the setup visible at the
point of use and avoids scrolling to distant module-level definitions.
Extract to module level only when 3+ tests share the same helper.

```python
# Good — helper defined inline, right next to the test
@pytest.mark.asyncio
async def test_state_delta_bundled_with_output():
    """State set before yield is flushed onto the output event."""

    class _Node(BaseNode):
        async def _run_impl(self, *, ctx, node_input):
            ctx.state['color'] = 'blue'
            yield 'result'

    ctx, events = _make_ctx()

    await NodeRunner(node=_Node(name='n'), parent_ctx=ctx).run()

    assert events[0].output == 'result'
    assert events[0].actions.state_delta['color'] == 'blue'

# Bad — helper defined 300 lines above, reader must scroll
class _StateThenOutputNode(BaseNode):
    async def _run_impl(self, *, ctx, node_input):
        ctx.state['color'] = 'blue'
        yield 'result'

# ... 300 lines later ...
async def test_state_delta_bundled_with_output():
    node = _StateThenOutputNode(name='n')
    ...
```

### 8. Assertions tell a story

```python
# Good — reads like a specification
assert queue.size == 0
assert config.get('timeout') == 30
assert response.status_code == 404

# Bad — overly defensive, tests framework behavior
assert isinstance(queue, Queue)
assert hasattr(config, 'get')
assert len(response.headers) > 0
```

### 9. Structure tests as arrange, act, assert

Every test has three distinct steps:

- **Arrange** — set up the external state specific to the scenario.
  General setup shared by many tests belongs in fixtures.
- **Act** — call the system under test. Usually a single call.
- **Assert** — verify return values or visible state changes. No
  further calls to the system under test here.

Keep steps distinct. Separate with blank lines. In simple tests where
each step is a single statement, blank lines can be omitted. In
complex tests, use descriptive comments like "Given [situation]",
"When [action]", "Then [expectation]" — avoid bare labels that add
no information.

```python
# Good — clear visual separation
def test_cache_returns_stored_value():
    cache = Cache()
    cache.put('key', 'value')

    result = cache.get('key')

    assert result == 'value'

# Good — simple test, blank lines omitted
def test_new_cache_is_empty():
    assert Cache().size == 0

# Bad — steps interleaved
def test_cache_behavior():
    cache = Cache()
    cache.put('key', 'value')
    result = cache.get('key')
    assert result == 'value'
    cache.put('key2', 'value2')  # more setup after assert
    assert cache.size == 2
```

### Test Structure Template

```python
"""Tests for <ComponentName>.

Verifies that <component> correctly <high-level behavior>.
"""

# --- Fixtures (minimal, one purpose each) ---

def _make_service():
    ...

# --- Tests (one behavior per test) ---

def test_<behavior_description>():
    """<One sentence: what the system does from the outside.>"""
    # Given a service with default config
    service = _make_service()
    input_data = 'hello'

    # When the operation is performed
    result = service.do_something(input_data)

    # Then the result matches expectations
    assert result == expected
```
