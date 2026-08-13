# Routing and Conditional Branching

A node emits `Event(route=...)`; the edges leaving that node decide which
successors fire.

## Dict routing — the default form

Map route values to targets in one edge tuple:

```python
from google.adk import Event, Workflow


def classify(node_input: str):
  route = 'error' if 'error' in node_input else 'success'
  return Event(output=node_input, route=route)


agent = Workflow(
    name='router',
    edges=[
        ('START', classify),
        (classify, {'success': handle_success, 'error': handle_error}),
    ],
)
```

The three-tuple form `(classify, handle_success, 'success')` does the same thing
one target at a time. Reach for it only when a single edge must match several
routes (see below), since the dict form maps one route to one target.

## Sequence shorthand

A tuple of more than two elements becomes a chain:

```python
edges = [('START', step_a, step_b, step_c)]
# equivalent to [('START', step_a), (step_a, step_b), (step_b, step_c)]
```

Chains and dict routing compose:

```python
edges = [
    ('START', process_input, classify),
    (classify, {'approved': send, 'rejected': discard}),
]
```

## Route values

A route is a `str`, `bool`, or `int`, or a list of those.

```python
(decision, {'approve': path_a, 'reject': path_b})   # strings
(decision, {True: yes_path, False: no_path})        # booleans
(decision, {0: path_0, 1: path_1})                  # integers
```

## Default route

`'__DEFAULT__'` (exported as `DEFAULT_ROUTE`) fires when no other route on that
node matches:

```python
edges = [
    ('START', classify),
    (classify, {
        'success': handler_a,
        'error': handler_b,
        '__DEFAULT__': fallback_handler,
    }),
]
```

One default per node. `'__DEFAULT__'` may not appear inside a list of routes on
one edge — give it its own edge.

## One edge, several routes

Passing a list matches any value in it:

```python
edges = [
    ('START', classifier),
    (classifier, {'route_z': handler_b}),
    (classifier, handler_a, ['route_x', 'route_y']),
]
```

## Several routes from one node

A node can emit a list of routes to fire multiple branches at once:

```python
def fan_out_router(node_input: str):
  return Event(output=node_input, route=['path_a', 'path_b'])


agent = Workflow(
    name='multi_route',
    edges=[
        ('START', fan_out_router),
        (fan_out_router, {'path_a': branch_a, 'path_b': branch_b}),
    ],
)
```

## Self-loop

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

## Revision loop

```python
edges = [
    ('START', process_input, draft_email, human_review),
    (human_review, {
        'revise': draft_email,
        'approved': send,
        'rejected': discard,
    }),
]
```

## Constraints the graph validator enforces

- **A cycle needs at least one routed edge.** An entirely unconditional cycle is
  rejected at construction time, because nothing could ever break out of it.
- **Edges leaving `START` may not carry a route.** `START` never runs, so it
  never emits one.
- **No two edges may share a source and a target**, even with different routes.
  To reach one destination from both a named route and `__DEFAULT__`, point the
  default at a thin wrapper function.

## Unrouted edges always fire

An edge with no route fires on every output event from its source, whatever
route that event carries. So if a node routes at all, give *every* one of its
outgoing edges a route — otherwise the unrouted one fires alongside the branch
you selected.

```python
edges = [
    ('START', node_a),  # unconditional
    (node_a, node_b),   # fires on every node_a output
]
```
