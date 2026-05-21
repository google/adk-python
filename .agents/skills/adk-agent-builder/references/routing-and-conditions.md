# Routing and Conditional Branching Reference

Route workflow execution along different paths based on node outputs.

## 📋 Agent Verification Checklist (Routing)
Use this checklist when implementing routing logic:
- [ ] **Syntax**: Is the preferred dict syntax used for mapping routes to targets? (Avoid verbose individual edges)
- [ ] **Loops**: Are cycles (loops) routed? (Unconditional cycles are rejected during validation)
- [ ] **Triggering**: If a node has conditional routing, do ALL outgoing edges have routes? (To avoid unintended triggering by unconditional edges)

## 💡 Quick Reference
- **Dict Routing**: `(source_node, {"route_a": target_a, "route_b": target_b})`
- **Sequence**: `("START", step_a, step_b, step_c)`
- **Default**: `"__DEFAULT__"` (Fallback route)

## Basic Routing

A node emits an `Event` with a `route` value. Use **dict syntax** to map routes to target nodes:

```python
from google.adk import Event, Workflow

def classify(node_input: str):
  if "error" in node_input:
    return Event(output=node_input, route="error")
  return Event(output=node_input, route="success")

def handle_success(node_input: str) -> str:
  return f"Success: {node_input}"

def handle_error(node_input: str) -> str:
  return f"Error: {node_input}"

agent = Workflow(
    name="router",
    edges=[
        ('START', classify),
        (classify, {"success": handle_success, "error": handle_error}),
    ],
)
```

## Routing Map (Dict Syntax) — Preferred

The dict syntax is the idiomatic way to express routing. It maps route values to target nodes in a single edge tuple:

```python
edges = [
    ("START", process_input, classifier, route_on_category),
    (route_on_category, {
        "question": answer_question,
        "statement": comment_on_statement,
        "other": handle_other,
    }),
]
```

This replaces verbose individual routed edges:
```python
# ❌ Verbose — avoid
(classifier, answer_question, "question"),
(classifier, comment_on_statement, "statement"),
(classifier, handle_other, "other"),

# ✅ Preferred — dict syntax
(classifier, {"question": answer_question, "statement": comment_on_statement, "other": handle_other}),
```

## Sequence Shorthand (Tuple Chains)

A tuple with more than 2 elements creates a sequential chain:

```python
# Shorthand: tuple creates chain edges
edges = [("START", step_a, step_b, step_c)]
# Equivalent to: [("START", step_a), (step_a, step_b), (step_b, step_c)]
```

Combine with dict routing:
```python
edges = [
    ("START", process_input, classify, route_on_result),
    (route_on_result, {"approved": send, "rejected": discard}),
]
```

## Route Value Types

Routes can be `str`, `bool`, or `int`:

```python
# String routes (most common)
(decision_node, {"approve": path_a, "reject": path_b})

# Boolean routes
(decision_node, {True: yes_path, False: no_path})

# Integer routes
(decision_node, {0: path_0, 1: path_1})
```

## Default Route

Use `'__DEFAULT__'` as a fallback when no other route matches:

```python
edges = [
    ("START", classify),
    (classify, {
        "success": handler_a,
        "error": handler_b,
        "__DEFAULT__": fallback_handler,
    }),
]
```

Only one default route per node is allowed.

**No duplicate edges:** Two edges from the same source to the same target are rejected, even with different routes. If you need both a named route and `__DEFAULT__` to reach the same destination, use a thin wrapper function for the default path.

## Dynamic Routing with Functions

A function node that emits different routes based on runtime data:

```python
from google.adk import Context, Event

def route_on_score(ctx: Context, node_input: dict):
  score = node_input.get("score", 0)
  if score > 0.8:
    return Event(output=node_input, route="high")
  elif score > 0.5:
    return Event(output=node_input, route="medium")
  else:
    return Event(output=node_input, route="low")

agent = Workflow(
    name="scored_router",
    edges=[
        ("START", compute_score, route_on_score),
        (route_on_score, {
            "high": premium_handler,
            "medium": standard_handler,
            "low": basic_handler,
        }),
    ],
)
```

## Multi-Route (Fan-Out via Route)

A node can output multiple routes to trigger multiple downstream paths simultaneously:

```python
def fan_out_router(node_input: str):
  return Event(output=node_input, route=["path_a", "path_b"])

agent = Workflow(
    name="multi_route",
    edges=[
        ("START", fan_out_router),
        (fan_out_router, {"path_a": branch_a, "path_b": branch_b}),
    ],
)
```

## List of Routes on a Single Edge

An edge can match multiple routes by passing a list as the route value. The edge fires if the node output matches **any** route in the list:

```python
edges = [
    ("START", classifier),
    (classifier, {"route_z": handler_b}),
    # handler_a fires on either route_x or route_y
    (classifier, handler_a, ["route_x", "route_y"]),
]
```

This is useful when multiple route values should lead to the same downstream node without duplicating edges. Note: list-of-routes on a single edge uses the 3-tuple syntax since dict syntax maps one route to one target.

## Self-Loop

A node can route back to itself:

```python
def guess_number(target_number: int):
  guess = random.randint(0, 10)
  yield Event(message=f'Guessing {guess}...')
  if guess == target_number:
    yield Event(message='Correct!')
  else:
    yield Event(route='guessed_wrong')

agent = Workflow(
    name='root_agent',
    edges=[
        ('START', validate_input, guess_number),
        (guess_number, {'guessed_wrong': guess_number}),
    ],
)
```

## Revision Loop

A common pattern: route back to an earlier node for revision, or forward for approval:

```python
edges = [
    ("START", process_input, draft_email, human_review),
    (human_review, {
        "revise": draft_email,
        "approved": send,
        "rejected": discard,
    }),
]
```

**Important**: Cycles must have at least one routed edge (unconditional cycles are rejected during graph validation).

## Unconditional Edges

Edges without a route value are unconditional — they always fire:

```python
edges = [
    ('START', node_a),       # Unconditional
    (node_a, node_b),        # Unconditional (always fires)
]
```

**Important**: Unrouted edges always fire, regardless of whether the output event has a route. If a node has conditional routing, ALL outgoing edges should have routes to avoid unintended triggering.
