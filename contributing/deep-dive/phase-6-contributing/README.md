# Phase 6: Contributing

**Duration:** Week 3+
**Outcome:** True mastery through contribution

---

## Overview

The ultimate test of understanding is contribution. This phase guides you through making your first meaningful contribution to ADK.

## Contribution Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONTRIBUTION FLOW                                 │
│                                                                          │
│  1. Find Issue    2. Discuss     3. Implement    4. Test    5. PR       │
│  ┌─────────┐     ┌─────────┐    ┌─────────┐    ┌────────┐  ┌────────┐  │
│  │ Browse  │ ──▶ │ Comment │ ──▶│  Code   │ ──▶│ pytest │──▶│ Submit │  │
│  │ Issues  │     │ on Issue│    │ Changes │    │ + E2E  │  │   PR   │  │
│  └─────────┘     └─────────┘    └─────────┘    └────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Finding Your First Issue

### Good First Issues

Look for labels:
- `good first issue` - Newcomer-friendly
- `help wanted` - Community contributions welcome

### Issue Types by Phase

| Your Phase | Suggested Issue Types |
|------------|----------------------|
| After Phase 1-2 | Documentation, typos |
| After Phase 3 | Bug fixes, small features |
| After Phase 4-5 | New integrations, major features |

### Where to Look

```bash
# GitHub Issues
https://github.com/google/adk-python/issues?q=is%3Aopen+label%3A%22good+first+issue%22

# Or search for specific areas
https://github.com/google/adk-python/issues?q=is%3Aopen+label%3A%22tools%22
```

---

## Step 2: Development Setup

### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/google/adk-python.git
cd adk-python

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment (Python 3.11+ recommended)
uv venv --python "python3.11" ".venv"
source .venv/bin/activate

# 4. Install all dependencies
uv sync --all-extras

# 5. Verify setup by running tests
pytest ./tests/unittests -x -v
```

### IDE Setup

For VSCode, create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.formatting.provider": "none",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  },
  "python.analysis.typeCheckingMode": "basic"
}
```

---

## Step 3: Understanding the Test Structure

### Test Organization

```
tests/
├── unittests/               # Unit tests (fast, isolated)
│   ├── agents/              # Agent tests
│   ├── tools/               # Tool tests
│   ├── sessions/            # Session tests
│   └── ...
└── integration/             # Integration tests
```

### Running Tests

```bash
# Run all unit tests
pytest ./tests/unittests

# Run specific module tests
pytest ./tests/unittests/tools/

# Run with verbose output
pytest ./tests/unittests -v

# Run with coverage
pytest ./tests/unittests --cov=src/google/adk

# Run only tests matching a pattern
pytest ./tests/unittests -k "test_function_tool"

# Stop on first failure
pytest ./tests/unittests -x
```

### Writing Tests

Follow existing patterns:

```python
# tests/unittests/tools/test_my_tool.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext


class TestMyTool:
    """Tests for my custom tool."""

    @pytest.fixture
    def mock_tool_context(self):
        """Create a mock tool context."""
        ctx = MagicMock(spec=ToolContext)
        ctx.state = {}
        return ctx

    @pytest.mark.asyncio
    async def test_basic_functionality(self, mock_tool_context):
        """Test that the tool works with valid input."""
        # Arrange
        def my_func(x: int) -> int:
            return x * 2

        tool = FunctionTool(func=my_func)

        # Act
        result = await tool.run_async(
            args={"x": 5},
            tool_context=mock_tool_context,
        )

        # Assert
        assert result == 10

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_tool_context):
        """Test that errors are handled properly."""
        def failing_func():
            raise ValueError("Expected error")

        tool = FunctionTool(func=failing_func)

        with pytest.raises(ValueError, match="Expected error"):
            await tool.run_async(
                args={},
                tool_context=mock_tool_context,
            )

    @pytest.mark.asyncio
    async def test_with_state_modification(self, mock_tool_context):
        """Test that state is properly modified."""
        async def stateful_func(value: str, tool_context: ToolContext):
            tool_context.state["saved"] = value
            return {"status": "ok"}

        tool = FunctionTool(func=stateful_func)

        await tool.run_async(
            args={"value": "test_value"},
            tool_context=mock_tool_context,
        )

        assert mock_tool_context.state["saved"] == "test_value"
```

---

## Step 4: Code Style

### Auto-formatting

```bash
# Run the auto-formatter before committing
./autoformat.sh
```

This runs:
- `isort` - Import sorting
- `pyink` - Code formatting (Google's Black fork)

### Style Guidelines

```python
# Good: Type hints everywhere
def create_agent(
    name: str,
    model: str,
    tools: list[BaseTool] | None = None,
) -> LlmAgent:
    ...

# Good: Docstrings for public APIs
def process_event(event: Event) -> Optional[Event]:
    """Process an event and optionally transform it.

    Args:
        event: The event to process.

    Returns:
        The transformed event, or None if the event should be dropped.

    Raises:
        ValueError: If the event is malformed.
    """
    ...

# Good: Async by default
async def fetch_data(url: str) -> dict:
    ...

# Good: Pydantic for data classes
class MyConfig(BaseModel):
    name: str
    value: int = 0
    options: list[str] = Field(default_factory=list)
```

---

## Step 5: Making Changes

### Branch Strategy

```bash
# Create a feature branch
git checkout -b feature/my-improvement

# Or for bug fixes
git checkout -b fix/issue-123-description
```

### Commit Messages

```bash
# Good commit messages
git commit -m "feat(tools): add retry logic to OpenAPI tool"
git commit -m "fix(sessions): handle empty state correctly"
git commit -m "docs: add example for custom plugin"
git commit -m "test: add unit tests for memory service"

# Format: type(scope): description
# Types: feat, fix, docs, test, refactor, chore
```

### Code Change Checklist

```markdown
Before submitting:
- [ ] Code follows existing patterns
- [ ] Type hints added for new code
- [ ] Docstrings for public APIs
- [ ] Unit tests written
- [ ] Existing tests still pass
- [ ] Auto-formatter run
- [ ] No new linting errors
```

---

## Step 6: Creating a Pull Request

### PR Structure

```markdown
## Summary
Brief description of what this PR does.

## Related Issue
Fixes #123

## Changes
- Added X feature
- Fixed Y bug
- Updated Z documentation

## Testing Plan
### Unit Tests
```
pytest ./tests/unittests/tools/test_my_change.py -v
# All tests pass
```

### E2E Testing
Tested with the following agent:
```python
agent = LlmAgent(
    name="test_agent",
    model="gemini-2.0-flash",
    tools=[MyNewTool()],
)
```

Screenshot/logs showing it works:
[Attach evidence]

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated (if needed)
- [ ] Auto-formatter run
- [ ] PR is focused on one concern
```

### PR Tips

1. **Keep PRs Small** - One concern per PR
2. **Include Evidence** - Screenshots, logs, test output
3. **Respond Promptly** - Address review feedback quickly
4. **Be Patient** - Reviews take time

---

## Example Contributions

### Example 1: Adding a New Tool

```python
# src/google/adk/tools/my_new_tool.py
"""My new tool implementation."""

from typing import Any
from google.adk.tools import BaseTool
from google.adk.tools import ToolContext


class MyNewTool(BaseTool):
    """A tool that does something useful.

    This tool provides functionality for...

    Example:
        >>> tool = MyNewTool(api_key="...")
        >>> result = await tool.run_async(args={"query": "test"}, tool_context=ctx)
    """

    def __init__(self, api_key: str):
        super().__init__(
            name="my_new_tool",
            description="Does something useful with the given query.",
        )
        self.api_key = api_key

    async def run_async(
        self, *, args: dict[str, Any], tool_context: ToolContext
    ) -> dict:
        query = args.get("query", "")
        # Implementation here
        return {"result": f"Processed: {query}"}
```

```python
# tests/unittests/tools/test_my_new_tool.py
import pytest
from google.adk.tools.my_new_tool import MyNewTool


class TestMyNewTool:
    @pytest.fixture
    def tool(self):
        return MyNewTool(api_key="test_key")

    @pytest.mark.asyncio
    async def test_basic_query(self, tool, mock_tool_context):
        result = await tool.run_async(
            args={"query": "hello"},
            tool_context=mock_tool_context,
        )
        assert "result" in result
        assert "hello" in result["result"]
```

### Example 2: Fixing a Bug

```python
# Before (buggy)
def parse_response(response: dict) -> str:
    return response["text"]  # KeyError if "text" missing

# After (fixed)
def parse_response(response: dict) -> str:
    """Parse the text from a response.

    Args:
        response: The response dictionary.

    Returns:
        The text content, or empty string if not present.
    """
    return response.get("text", "")
```

```python
# Test for the fix
def test_parse_response_handles_missing_text():
    """Ensure parse_response handles missing 'text' key."""
    result = parse_response({})
    assert result == ""

def test_parse_response_with_text():
    """Ensure parse_response extracts text correctly."""
    result = parse_response({"text": "hello"})
    assert result == "hello"
```

### Example 3: Adding Documentation

For documentation changes, update the [adk-docs](https://github.com/google/adk-docs) repository alongside your code PR.

---

## Common Pitfalls

| Mistake | How to Avoid |
|---------|--------------|
| Large PRs | Split into smaller, focused changes |
| No tests | Always add/update tests |
| Breaking changes | Discuss in issue first |
| Ignoring style | Run `./autoformat.sh` |
| Incomplete PR description | Use the template |

---

## Getting Help

- **GitHub Issues** - Ask questions on relevant issues
- **Discussions** - For general questions
- **AGENTS.md** - Context for LLM-assisted development

---

## Mastery Checklist

After contributing, you should be able to:

- [ ] Set up the development environment
- [ ] Navigate any module confidently
- [ ] Write tests following existing patterns
- [ ] Follow code style guidelines
- [ ] Create well-structured PRs
- [ ] Respond to code review feedback
- [ ] Explain ADK architecture to others

---

## Congratulations!

If you've completed all six phases:

1. You understand ADK's core abstractions
2. You can trace request flows
3. You've built real projects
4. You recognize the design patterns
5. You've specialized in an area
6. You've contributed back

**You are now an ADK expert.**

---

**Back to:** [Learning Path Overview](../README.md)
