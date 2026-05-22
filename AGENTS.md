# AI Coding Assistant Context

This document provides context for AI coding assistants (Antigravity, Gemini CLI, etc.) to understand the ADK Python project and assist with development.

## ADK Knowledge, Architecture, and Style

For all matters regarding ADK development, please use the appropriate skill:

- **`adk-architecture`**: Use this skill whenever you need to understand the architecture, event flow, or state management of the ADK system, or when designing or modifying core components and public APIs.
  - Read `.agents/skills/adk-architecture/SKILL.md` for full instructions.
- **`adk-style`**: Use this skill whenever writing code, tests, or reviewing PRs for the ADK project to ensure compliance with styling and coding conventions. Also use it for committing, bug fixing, and testing rules.
  - Read `.agents/skills/adk-style/SKILL.md` for full instructions.
- **`adk-git`**: Use this skill for any git operation (commit, push, pull, rebase, etc.). It provides guidelines for Conventional Commits and branch naming.
  - Read `.agents/skills/adk-git/SKILL.md` for full instructions.
- **`adk-sample-creator`**: Use this skill when creating new samples demonstrating features or agent patterns, or when adding examples to subdirectories under `contributing/`.
  - Read `.agents/skills/adk-sample-creator/SKILL.md` for full instructions.

## Project Overview

The Agent Development Kit (ADK) is an open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents.

### Key Components

- **Agent**: Blueprint defining identity, instructions, and tools.
- **Runner**: Stateless execution engine that orchestrates agent execution.
- **Tool**: Functions/capabilities agents can call.
- **Session**: Conversation state management.
- **Memory**: Long-term recall across sessions.
- **Workflow** (ADK 2.0): Graph-based orchestration of complex, multi-step agent interactions.
- **BaseNode** (ADK 2.0): Contract for all nodes, supporting output streaming and human-in-the-loop steps.
- **Context** (ADK 2.0): Holds execution state and telemetry context mapped 1:1 to nodes.

For details on how the Runner works and the invocation lifecycle, please refer to the `adk-architecture` skill and the referenced documentation therein.

## Project Architecture

For detailed architecture patterns, component descriptions, and core interfaces, please refer to the **`adk-architecture`** skill at `.agents/skills/adk-architecture/SKILL.md`.

## Development Setup

The project uses `uv` for package management and Python 3.11+. Please refer to the **`adk-setup`** skill at `.agents/skills/adk-setup/SKILL.md` for detailed instructions.
