# GitHub Issue: CodingAgent Feature Request

**Use this content to create an issue at: https://github.com/google/adk-python/issues/new?template=feature_request.md**

---

## Title

`feat(agents): Add CodingAgent - Agents that Think in Code with Sandboxed Execution`

---

## Is your feature request related to a problem? Please describe.

### The Fundamental Limitation of Tool-Calling Agents

Current ADK agents operate through a **predefined tool paradigm**: the agent receives a task, selects from a fixed set of tools, and chains tool calls to accomplish goals. While effective for well-scoped problems, this architecture imposes fundamental constraints that limit agent capabilities in real-world scenarios:

#### 1. **Constrained Action Space**

Tool-calling agents are restricted to a finite, pre-enumerated set of actions. As demonstrated in the [CodeAct paper (ICML 2024)](https://arxiv.org/abs/2402.01030), this creates a **combinatorial explosion problem**: complex tasks requiring composition of multiple operations become intractable when each combination must be explicitly defined as a tool. The paper shows that code-based actions achieve **up to 20% higher success rates** by allowing arbitrary composition.

#### 2. **Context Window Bottleneck**

Modern LLMs have context windows ranging from 8K to 200K tokens, but complex reasoning tasks can easily exceed these limits. Tool-calling agents must maintain entire conversation histories, tool schemas, and intermediate results in context. Code agents solve this by **offloading computation to the execution environment**—variables persist in the sandbox, not in the context window. This insight is central to systems like Claude Code and OpenCode that can work on entire codebases.

#### 3. **Inability to Dynamically Create Actions**

The [DynaSaur paper (COLM 2025)](https://arxiv.org/abs/2411.01747) identifies a critical flaw: "Existing LLM agent systems typically select actions from a fixed and predefined set at every step... this requires substantial human effort to enumerate and implement all possible actions, which is impractical in complex environments." Code agents can **generate novel actions on-the-fly**, adapting to unforeseen scenarios.

#### 4. **Competitive Gap**

The industry has converged on code-generating agents as the next evolution:
- **OpenAI Code Interpreter**: Code execution in sandbox
- **Anthropic Claude's Computer Use**: Code-based computer control
- **HuggingFace smolagents**: "Agents that think in code" ([25k+ GitHub stars](https://github.com/huggingface/smolagents))
- **OpenCode, Claude Code, Cursor**: Production coding agents using this pattern

ADK currently lacks this capability, forcing users to build complex workarounds or use competing frameworks.

### Concrete User Pain Points

```
"I want my agent to analyze a 50MB CSV and create visualizations"
→ Current: Build custom tools for every possible analysis operation
→ With CodingAgent: Agent writes pandas/matplotlib code dynamically

"I need multi-step calculations with intermediate results"
→ Current: Chain multiple tool calls, losing state between each
→ With CodingAgent: Variables persist in sandbox across iterations

"I want the agent to figure out HOW to solve a problem"
→ Current: Limited to predefined solution paths
→ With CodingAgent: Agent generates arbitrary solution code

"I need to work with long documents without hitting context limits"
→ Current: Complex chunking strategies, lost coherence
→ With CodingAgent: Load documents into sandbox, process incrementally
```

---

## Describe the solution you'd like

### CodingAgent: A ReAct Agent that Thinks in Code

Inspired by [HuggingFace's smolagents](https://github.com/huggingface/smolagents) and grounded in recent research ([CodeAct](https://arxiv.org/abs/2402.01030), [DynaSaur](https://arxiv.org/abs/2411.01747)), CodingAgent is a new agent type that:

1. **Generates Python code** as its action representation (in `tool_code` blocks)
2. **Executes code in sandboxed Docker containers** for security
3. **Calls ADK tools from generated code** via HTTP IPC
4. **Iterates using a ReAct loop** until producing a final answer
5. **Maintains state across iterations** in the execution environment

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│   User Query    │────▶│   CodingAgent    │────▶│   Docker Container      │
│                 │     │  (Gemini LLM)    │     │   (Python 3.11)         │
└─────────────────┘     └──────────────────┘     │                         │
                               │                 │  • pandas, numpy        │
                               │                 │  • matplotlib, seaborn  │
                               │ ReAct Loop      │  • Any pip package      │
                               │                 │  • Persistent state     │
                               ▼                 └───────────┬─────────────┘
                        ┌──────────────┐                    │
                        │ Tool Server  │◀───────────────────┘
                        │ (HTTP IPC)   │  Tool calls via HTTP POST
                        │ Port 8765    │  (fetch_url, save_chart, etc.)
                        └──────────────┘
```

### Why Code Actions Are Superior

From the CodeAct paper:
> "CodeAct can execute code actions and dynamically revise prior actions or emit new actions upon new observations through multi-turn interactions... CodeAct outperforms widely used alternatives (up to 20% higher success rate)."

From DynaSaur:
> "The agent interacts with its environment by generating and executing programs written in a general-purpose programming language. Moreover, generated actions are accumulated over time for future reuse."

Code provides:
- **Composability**: Combine operations arbitrarily (`for url in urls: data.append(fetch(url))`)
- **State persistence**: Variables survive across iterations
- **Dynamic tool creation**: Write new functions as needed
- **Error handling**: Try/except, retries, fallbacks in code
- **Computational offloading**: Process data in sandbox, not context

### API Design

```python
from google.adk.agents import CodingAgent
from google.adk.code_executors import ContainerCodeExecutor
from google.adk.code_executors.allowlist_validator import DEFAULT_SAFE_IMPORTS

def fetch_data(url: str) -> dict:
    """Fetch data from a URL - available to generated code."""
    # Implementation...

def save_chart(image_data: str, filename: str) -> dict:
    """Save chart to host filesystem - bridges container to host."""
    # Implementation...

root_agent = CodingAgent(
    name="data_analyst",
    model="gemini-2.5-flash",
    instruction="You are a data analyst. Analyze data and provide insights.",
    tools=[fetch_data, save_chart],  # Tools callable from generated code
    code_executor=ContainerCodeExecutor(image="python:3.11-slim"),
    authorized_imports=DEFAULT_SAFE_IMPORTS | {"pandas", "matplotlib", "numpy"},
    max_iterations=10,
    error_retry_attempts=2,
    stateful=False,  # Future: True for persistent state across turns
)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **CodingAgent** | ReAct loop orchestrator, code extraction, LLM interaction |
| **CodingAgentCodeExecutor** | Wraps underlying executor, injects tool stubs |
| **ToolCodeGenerator** | Generates Python function stubs for ADK tools |
| **ToolExecutionServer** | HTTP server enabling tool calls from container |
| **AllowlistValidator** | Security: validates imports against allowlist |

### Security Model

1. **Sandboxed execution**: All code runs in isolated Docker containers
2. **Import allowlisting**: Only explicitly authorized imports permitted
3. **Tool isolation**: Tools execute on host, not in container
4. **No filesystem access**: Container cannot access host filesystem
5. **Network isolation**: Container only reaches tool server
6. **Configurable**: Users control exactly what's permitted

---

## Describe alternatives you've considered

### Alternative 1: Extend LlmAgent with Code Execution Tool

Add code execution as just another tool that LlmAgent can call.

**Pros:**
- Minimal API changes
- Reuses existing agent infrastructure

**Cons:**
- No ReAct loop for code iteration
- Tool-calling overhead for every code snippet
- Mixes paradigms (tool-calling agent calling code execution tool)
- No state persistence between code executions
- Doesn't capture the "thinking in code" pattern

**Why rejected:** This approach treats code execution as an afterthought rather than a first-class paradigm. The power of code agents comes from the tight integration of code generation, execution, and iteration—not from occasionally executing snippets.

### Alternative 2: External Code Execution Service

Integrate with external services like E2B, Modal, or Blaxel for code execution.

**Pros:**
- Offloads security concerns to specialized providers
- Potentially more scalable
- No Docker dependency for users

**Cons:**
- External dependency and potential costs
- Latency for remote execution
- Less control over execution environment
- Requires API keys and network access
- Not self-contained

**Why rejected:** While external services are valuable for production deployments, ADK should provide a self-contained solution that works out-of-the-box. Users can later swap ContainerCodeExecutor for cloud-based alternatives.

### Alternative 3: Unsafe Local Execution

Execute code directly in the host Python process.

**Pros:**
- Simplest implementation
- Fastest execution
- No Docker dependency

**Cons:**
- **Critical security risk**: Arbitrary code execution
- Cannot safely use in production
- No isolation between agent code and host system

**Why rejected:** Security is non-negotiable for an agent framework. Even with import restrictions, local execution opens attack vectors through creative code generation.

### Chosen Approach: Dedicated CodingAgent with Docker Sandboxing

A purpose-built agent class that:
- Makes code generation a first-class paradigm
- Provides secure sandboxed execution via Docker
- Enables tool access through HTTP IPC
- Supports future extensions (stateful execution, alternative sandboxes)

This approach aligns with the architecture of smolagents while integrating cleanly with ADK's existing infrastructure.

---

## Additional context

### Implementation Status

I have a **complete, tested implementation** ready for PR submission.

**Production Code (~2,500 lines):**
| File | Lines | Description |
|------|-------|-------------|
| `src/google/adk/agents/coding_agent.py` | ~610 | Main agent class with ReAct loop |
| `src/google/adk/agents/coding_agent_config.py` | ~225 | Pydantic configuration |
| `src/google/adk/code_executors/coding_agent_code_executor.py` | ~505 | Executor wrapper with tool injection |
| `src/google/adk/code_executors/tool_code_generator.py` | ~475 | Python stub generation for tools |
| `src/google/adk/code_executors/tool_execution_server.py` | ~365 | HTTP IPC server for tool calls |
| `src/google/adk/code_executors/allowlist_validator.py` | ~355 | Import security validation |

**Sample Agent:**
| File | Description |
|------|-------------|
| `contributing/samples/coding_agent/agent.py` | Data Analysis Agent with 5 tools |
| `contributing/samples/coding_agent/README.md` | Comprehensive documentation |

**Unit Tests (~950 lines):**
| File | Tests |
|------|-------|
| `tests/unittests/agents/test_coding_agent.py` | Agent creation, code extraction, error handling |
| `tests/unittests/code_executors/test_allowlist_validator.py` | Import validation, patterns |
| `tests/unittests/code_executors/test_tool_code_generator.py` | Stub generation, system prompts |

### E2E Test Results

| Test Scenario | Status | Notes |
|--------------|--------|-------|
| Basic math ("What is 25 * 17?") | ✅ Passed | Generates code, executes, returns 425 |
| Data analysis (Titanic survival rate) | ✅ Passed | Fetches CSV, uses pandas, returns 38.38% |
| Visualization (bar chart by class) | ✅ Passed | Creates matplotlib chart, saves to host |
| Multi-step analysis | ✅ Passed | Stats → visualization → insights in one query |
| Tool calling via HTTP IPC | ✅ Passed | fetch_url, save_chart work correctly |
| Error handling (pip warnings) | ✅ Passed | Distinguishes warnings from real errors |

### Research Foundation

This implementation is grounded in peer-reviewed research:

1. **CodeAct (ICML 2024)**: [arXiv:2402.01030](https://arxiv.org/abs/2402.01030)
   - "Executable Code Actions Elicit Better LLM Agents"
   - Shows 20% improvement over JSON/text tool calling
   - Introduces the `tool_code` action format we adopt

2. **DynaSaur (COLM 2025)**: [arXiv:2411.01747](https://arxiv.org/abs/2411.01747)
   - "Large Language Agents Beyond Predefined Actions"
   - Demonstrates value of dynamically generated actions
   - Shows agents can create and accumulate actions over time

3. **smolagents (HuggingFace)**: [github.com/huggingface/smolagents](https://github.com/huggingface/smolagents)
   - "Agents that think in code" - 25k+ stars
   - Production-proven architecture
   - Supports multiple sandbox backends (E2B, Modal, Docker)

### Future Roadmap (Out of Scope for Initial PR)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Stateful execution** | Persist variables across conversation turns | High |
| **Alternative sandboxes** | E2B, Modal, Pyodide+Deno WebAssembly | High |
| **Custom container images** | Pre-installed packages for faster execution | Medium |
| **Jupyter integration** | Execute in Jupyter kernels | Medium |
| **Multi-agent orchestration** | CodingAgent as sub-agent in hierarchies | Medium |
| **Streaming output** | Stream stdout/stderr during execution | Low |
| **Additional languages** | JavaScript/TypeScript support | Low |

### Enabling New Use Cases

CodingAgent unlocks capabilities previously impossible or impractical in ADK:

1. **AI Data Scientists**: Analyze datasets, create visualizations, generate reports
2. **Code Review Agents**: Read codebases, run analysis, suggest improvements
3. **Automation Agents**: Generate scripts to accomplish arbitrary tasks
4. **Research Assistants**: Process papers, extract data, create summaries
5. **Sub-agent Architecture**: Build systems like Claude Code with specialized sub-agents

### Labels to Add

- `enhancement`
- `agents`
- `new-feature`
- `experimental`
