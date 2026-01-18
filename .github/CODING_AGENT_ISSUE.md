# GitHub Issue: CodingAgent Feature Request

Use this content to create an issue at:
https://github.com/google/adk-python/issues/new?template=feature_request.md

---

## Title

feat(agents): Add CodingAgent (agents that think in code)

---

## Is your feature request related to a problem? Please describe.

ADK’s current default agent interaction pattern is “tool selection from a fixed action set”. This is powerful, but it breaks down for two increasingly common workloads:

1) Long-context work beyond model context windows
- Many real tasks require operating over very large corpora: codebases, logs, datasets, multi-file configs, or long documents.
- If the agent must keep the relevant source text and intermediate results inside the LLM context, it becomes context-window bound and expensive.
- Recent work such as “Recursive Language Models” (arXiv:2512.24601) proposes treating long prompts as an external environment and letting the model programmatically examine/decompose/recursively process snippets. This suggests a practical direction for agents: move heavy inspection, decomposition, and intermediate state out of the prompt and into an execution environment.
  - https://arxiv.org/abs/2512.24601

2) Expressiveness and composability limits of pure tool-calling
- Tool-calling assumes we can enumerate actions up-front. In open-ended tasks, the agent needs to compose multiple operations, iterate, cache intermediate artifacts, and implement “one-off” transformations without requiring new bespoke tools each time.
- A code-based action space lets the agent compose operations naturally (loops, conditionals, helper functions), which reduces the need for an explosion of tools.

3) Developer experience gap for building “coding agents” and sub-agent architectures
- Users increasingly want agent systems like Claude Code / OpenCode: multi-step coding workflows with sub-agents (planner, tester, refactorer, etc.) and strong “think in code” execution.
- ADK has strong orchestration primitives; adding a first-class code-executing agent unlocks building these systems within ADK while keeping sandboxing and tool integration.

Related inspiration: HuggingFace “smolagents” positions CodeAgent as a first-class concept (“agents that think in code”) and supports sandbox backends (Docker, etc.).
- https://github.com/huggingface/smolagents

---

## Describe the solution you’d like

Add a new experimental agent type: CodingAgent.

CodingAgent should:
- Generate Python code as the primary action representation (in `tool_code` blocks).
- Execute that code in a sandboxed environment (Docker-based initially).
- Allow generated code to call ADK tools safely via an IPC bridge (e.g., HTTP) rather than exposing the host runtime directly.
- Support iterative execution (ReAct-style loop): generate → run → observe stdout/tool results → refine → final answer.

Why this solves the problem
- Long-context: aligns with the “external environment” framing in arXiv:2512.24601 by enabling the agent to iteratively inspect, decompose, and process large inputs using code and persisted artifacts, instead of forcing all content into the model context.
- Composability: code enables arbitrary composition (loops, conditionals, helper functions) without requiring every combination to be implemented as a first-class tool.
- Coding-agent architectures: makes it straightforward to build higher-level workflows and multi-agent hierarchies where sub-agents can generate/run code for specialized tasks.

High-level architecture

User → CodingAgent (LLM) → sandbox executor (Docker Python)
                     ↘ tool IPC server on host ↙

Proposed execution environments (progressive)
- v1: Docker Python sandbox (existing ContainerCodeExecutor integration)
- future: REPL / Jupyter-kernel style execution modes for interactive, stateful sessions (still sandboxed)

---

## Describe alternatives you’ve considered

1) “Just add a code-execution tool” to existing agents
- Pros: minimal surface-area change.
- Cons: code execution becomes an occasional tool call rather than the agent’s primary action space; harder to support tight generate→execute→iterate loops and long-context strategies that rely on an external environment.

2) Require users to write bespoke tools for every operation
- Pros: explicit and controlled.
- Cons: does not scale; real workflows need ad-hoc transformations and composition that explode the tool surface area.

3) Run code on the host interpreter
- Pros: simplest.
- Cons: unacceptable security risk; sandboxing is required for a general-purpose code agent.

---

## Additional context

Future directions enabled by CodingAgent
- Long-context scaffolds inspired by arXiv:2512.24601: treat large inputs (files, repo trees, logs) as an “environment” the agent queries/decomposes recursively using code, storing intermediate state outside the LLM context.
- Sub-agent coding workflows (Claude Code / OpenCode style): planner/tester/refactor sub-agents coordinated by ADK, each using code execution.
- Multiple sandbox backends (like smolagents): Docker initially, with optional future support for other sandboxes and interactive execution modes.

Links
- smolagents (inspiration): https://github.com/huggingface/smolagents
- Recursive Language Models (long-context framing): https://arxiv.org/abs/2512.24601

Labels to add
- enhancement
- agents
- new-feature
- experimental
