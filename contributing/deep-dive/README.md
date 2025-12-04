# ADK Deep Dive: Mastering the Agent Development Kit

A structured learning path for senior developers to master the ADK-Python codebase.

## Overview

This guide takes you from understanding core concepts to contributing production-ready code. The approach is **bottom-up with core flow first** - we start with the foundational abstractions and build up to complex multi-agent systems.

## Learning Philosophy

| Principle | Reasoning |
|-----------|-----------|
| **Core-first, not module-first** | 5 core files explain 80% of behavior |
| **Trace execution, don't just read** | Reveals implicit contracts and flows |
| **Build, don't just study** | Forces encounter with real edge cases |
| **Pattern recognition over memorization** | ADK follows highly consistent patterns |
| **Depth before completeness** | Better to master Runner than skim everything |

## Prerequisites

- Strong Python background (async/await, type hints, decorators)
- Familiarity with LLM concepts (prompts, function calling, streaming)
- Basic understanding of REST APIs and async programming

## Learning Path

### [Phase 1: Core Vocabulary](./phase-1-core-vocabulary/README.md) (Days 1-2)
Master the 5 foundational abstractions that define ADK:
- **Agent** - The blueprint
- **Event** - The data flow
- **Session** - The state container
- **Tool** - The capabilities
- **Runner** - The orchestrator

### [Phase 2: Request Flow Tracing](./phase-2-request-flow/README.md) (Days 2-3)
Trace a complete request through the system:
- From user input to LLM call
- Tool execution and response handling
- Event generation and session updates
- Understanding the "Reason-Act" loop

### [Phase 3: Mini-Projects](./phase-3-mini-projects/README.md) (Week 1)
Build three projects that cover 70% of the codebase:
1. **Custom Tool Agent** - Master tool binding
2. **Multi-Agent Workflow** - Learn orchestration
3. **Stateful Chatbot** - Understand persistence

### [Phase 4: Abstraction Layers](./phase-4-abstraction-layers/README.md) (Week 2)
Recognize the consistent service patterns:
- Base classes and their implementations
- Dependency injection patterns
- Plugin architecture

### [Phase 5: Deep Dive Tracks](./phase-5-deep-dive-tracks/README.md) (Week 2-3)
Choose a specialization:
- **LLM Integration** - Model adapters and flows
- **Tool Ecosystem** - OpenAPI, MCP, databases
- **Evaluation** - Testing AI agents systematically
- **Deployment** - Production readiness

### [Phase 6: Contributing](./phase-6-contributing/README.md) (Week 3+)
The mastery test:
- Finding and fixing issues
- Following existing patterns
- Writing comprehensive tests

## Time Investment

| Phase | Duration | Outcome |
|-------|----------|---------|
| 1-2 | 3-4 days | Can explain architecture to others |
| 3 | 4-5 days | Can build custom agents confidently |
| 4-5 | 1 week | Can extend/debug any module |
| 6 | Ongoing | True ownership |

**Total to mastery: ~3 weeks of focused work**

## Anti-Patterns to Avoid

1. **Don't start with `tools/`** - Too many files, you'll get lost
2. **Don't read tests first** - They test edge cases, not teach concepts
3. **Don't skip `runners.py`** - It's 56KB but it's THE core logic
4. **Don't ignore `events/`** - Events are the data model for everything

## Quick Reference: File Locations

```
src/google/adk/
├── agents/           # Agent types and configuration
│   ├── base_agent.py        # START HERE - Agent abstraction
│   ├── llm_agent.py         # Primary agent implementation
│   └── ...
├── runners.py        # THE CORE - Execution engine
├── events/           # Event model
│   └── event.py             # What flows through the system
├── sessions/         # State management
│   └── session.py           # Conversation state
├── tools/            # Tool ecosystem
│   ├── base_tool.py         # Tool abstraction
│   └── ...
├── models/           # LLM integrations
├── memory/           # Long-term recall
├── artifacts/        # File/media management
├── evaluation/       # Testing framework
├── cli/              # Command-line tools
└── plugins/          # Extension system
```

## Getting Started

1. Clone the repository and set up your environment:
   ```bash
   git clone https://github.com/google/adk-python.git
   cd adk-python
   pip install -e ".[dev]"
   ```

2. Start with [Phase 1: Core Vocabulary](./phase-1-core-vocabulary/README.md)

3. Run examples as you learn:
   ```bash
   cd contributing/samples/hello_world
   adk run .
   ```

---

**Next:** [Phase 1: Core Vocabulary](./phase-1-core-vocabulary/README.md)
