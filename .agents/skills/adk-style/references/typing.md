# Type Hints and Strong Typing

## General Rules

- **Annotate everything**: type hints on all function arguments and return
  types, including `-> None` and private functions.
- **Minimize `Any`**: use a specific type or a `TypeVar`. `Any` disables
  checking for every value that flows through it.
- **Precise types**: `Literal` for a fixed set of strings, `TypedDict` for a
  dict with known keys, `Protocol` for a structural interface.
- **`from __future__ import annotations` goes at the top of every module**
  under `src/google/adk/`, immediately after the license header and before any
  other import. `scripts/compliance_checks.py` fails the commit if it is
  missing. Exempt: `__init__.py`, `version.py`, `tests/`, and
  `contributing/samples/`.
- **No quoted type hints.** Deferred annotations make forward references work
  unquoted, so write `list[str]`, not `"list[str]"`.
- **Builtin generics** for new code: `list[str]`, `dict[str, int]`,
  `tuple[str, ...]`. `typing.List` / `typing.Dict` survive in older modules;
  don't add more, and don't churn existing ones.

## Mypy

Mypy runs in `strict` mode against `src/` with the Pydantic plugin, targeting
Python 3.11 (`[tool.mypy]` in `pyproject.toml`). `tests/` and
`contributing/samples/` are excluded.

```bash
mypy .
```

The CI job compares your branch's errors against the base branch and fails
only on **new** ones, so a pre-existing error in a file you touched is not
your problem — an error on a line you added is.

## Escape Hatches

Fix a type error rather than silencing it. When you can't:

- `# type: ignore[arg-type]  # <the external cause>`, never a bare ignore.
- `object`, not `Any`, for a value that can be anything.
- `cast(Foo, x)  # <why it holds>`, never a bare cast. On a typing-only
  change, use a `cast` rather than a new `isinstance` check that raises: test
  doubles and protobuf maps reach that check.

## `Optional[X]` vs `X | None`

Both appear in the codebase. Follow this convention:

- **New code** (especially in `workflow/`): prefer `X | None`.
- **Existing files**: match the style already in the file.
- Do not refactor one into the other without a reason.

## Agent Mode Constants and Type Aliases

Use `google.adk.utils._agent_mode` for agent execution and delegation modes
rather than scattering raw `'chat'`, `'task'`, and `'single_turn'` strings or
repeating `Literal[...]` unions:

- **Runtime checks and defaults**: use `AgentMode.CHAT`, `AgentMode.TASK`,
  `AgentMode.SINGLE_TURN`, and `DELEGATED_TASK_MODES`
  (`frozenset({AgentMode.TASK, AgentMode.SINGLE_TURN})`). `AgentMode`
  subclasses `(str, enum.Enum)`, so members compare equal to plain strings.
- **Type annotations**: use the shared type aliases from `utils/_agent_mode.py`
  (`LlmAgentMode`, `SingleTurnAgentMode`, `TaskAgentMode`,
  `DefaultLlmNodeMode`).

```python
# Bad — raw mode strings and duplicated Literal unions
def run_agent(
    agent: BaseAgent,
    default_mode: Literal['chat', 'single_turn'] = 'single_turn',
) -> None:
  if getattr(agent, 'mode', None) == 'single_turn':
    ...
  if agent.mode in ('task', 'single_turn'):
    ...

# Good — shared enum, constant set, and type alias
from ..utils._agent_mode import AgentMode
from ..utils._agent_mode import DefaultLlmNodeMode
from ..utils._agent_mode import DELEGATED_TASK_MODES

def run_agent(
    agent: BaseAgent,
    default_mode: DefaultLlmNodeMode = AgentMode.SINGLE_TURN,
) -> None:
  if getattr(agent, 'mode', None) == AgentMode.SINGLE_TURN:
    ...
  if agent.mode in DELEGATED_TASK_MODES:
    ...
```

## Path and Node Builders (`_NodePathBuilder`, `_BranchPath`, `build_node`)

Never manipulate workflow node paths or execution branch strings with ad-hoc
`str.split`, `str.startswith`, or f-string concatenation, and never manually
wrap or mutate `NodeLike` targets in place:

- **Workflow node paths (`'wf@1/node@2'`)**: use `_NodePathBuilder` from
  `google.adk.events._node_path_builder` (`.from_string()`, `.append()`,
  `.parent`, `.node_name`, `.run_id`, `.static_path`, `.is_descendant_of()`,
  `.is_direct_child_of()`, `.get_direct_child()`).
- **Execution branch paths (`'parent@1.child@2'`)**: use `_BranchPath` from
  `google.adk.events._branch_path` (`.from_string()`, `.create_sub_branch()`,
  `.append()`, `.parent`, `.segments`, `.run_ids`, `.is_descendant_of()`,
  `.common_prefix()`, `.is_tool_branch()`).
- **Workflow node construction**: use `build_node` from
  `google.adk.workflow.utils._workflow_graph_utils` to normalize any `NodeLike`
  (`BaseNode`, `BaseAgent`, `BaseTool`, or callable) into a `BaseNode` without
  mutating shared `LlmAgent` instances in place.

```python
# Bad — raw '/' / '.' / '@' string splitting, prefix matching, and manual wrapping
if event_path == node_path or event_path.startswith(f'{node_path}/'):
  ...
sub_branch = f'{ctx.branch}.{agent.name}' if ctx.branch else agent.name

# Good — structured builders handle run_ids and hierarchy checks
from ..events._branch_path import _BranchPath
from ..events._node_path_builder import _NodePathBuilder
from ..workflow.utils._workflow_graph_utils import build_node

self_path = _NodePathBuilder.from_string(node_path)
ev_path = _NodePathBuilder.from_string(event_path)
if ev_path == self_path or ev_path.is_descendant_of(self_path):
  ...
sub_branch = str(
    _BranchPath.create_sub_branch(ctx.branch, name=agent.name, run_id=run_id)
)
executable_node = build_node(target, default_llm_mode=AgentMode.SINGLE_TURN)
```

## Abstract Types for Function Parameters

Annotate parameters with abstract types from `collections.abc` so callers can
pass any compatible container; annotate returns with the concrete type so
callers know exactly what they get.

```python
from collections.abc import Mapping
from collections.abc import Sequence

def merge_labels(
    labels: Mapping[str, str], extra: Sequence[str]
) -> dict[str, str]:
  ...
```

## Keyword-Only Arguments

Put `*` before the parameters of any constructor or function where argument
order is easy to get wrong — two parameters of the same type is enough for a
silent bug.

```python
class NodeRunner:

  def __init__(
      self,
      *,
      node: BaseNode,
      parent_ctx: Context,
      run_id: str | None = None,
  ):
    ...
```

Use it for: constructors with 2+ non-`self` parameters, any function where
swapping two arguments would still typecheck, and methods taking several
`str` or `int` parameters.

## Mutable Default Arguments

A mutable default is evaluated once at definition time and shared by every
call, so one caller's mutation leaks into the next. Use `None` as a sentinel:

```python
# Bad — every caller shares one list.
def add(item: str, items: list[str] = []) -> list[str]:
  ...

# Good
def add(item: str, items: list[str] | None = None) -> list[str]:
  items = list(items) if items else []
  ...
```

This applies to `list`, `dict`, `set`, and any other mutable type.

## Runtime Type Discrimination with `isinstance()`

`isinstance()` is the codebase's standard way to handle polymorphic input.
Write exhaustive `if`/`elif` chains and always terminate them:

```python
if isinstance(node, FunctionNode):
  ...
elif isinstance(node, (JoinNode, ToolNode)):
  ...
else:
  raise TypeError(f'Unsupported node type: {type(node)}')
```

- Always include an `else` that raises `TypeError` or handles the unknown
  case, so a new subclass fails loudly instead of silently doing nothing.
- Prefer `isinstance(x, SomeType)` over `type(x) is SomeType` — it handles
  subclasses.
- Check several types at once with a tuple: `isinstance(x, (TypeA, TypeB))`.

## No Asserts in Production Code

`assert` is stripped when Python runs with `-O`, so an assertion is not a
runtime guarantee, and its failure message tells the caller nothing. Raise
`ValueError`, `TypeError`, or `RuntimeError` instead. Asserts in tests are
fine.
