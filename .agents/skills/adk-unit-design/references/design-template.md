# Unit design document template

Copy this structure into `docs/design/{topic}/{unit}/index.md`. The bullets are
instructions for what to write in each section, not text to keep.

```markdown
# {unit_name} - Code Unit Design

Two-sentence summary of the code unit.

## Introduction

Prose covering:

- The purpose and application of the unit, including intended use cases.
- The developer problems it solves.
- The agent capabilities it enables.

## High-level architecture

- Where the unit sits in the wider ADK framework.
- Its general execution flow.
- Data flows it handles, including inputs and outputs.
- Cross-class dependencies, upstream and downstream.

### Extension points

How the unit is meant to be extended or customised, naming the surfaces that
actually exist in the code: abstract classes, interfaces, hooks, callbacks,
configurable parameters, plugin registration.

### Extension constraints

What must not be modified, and why — architectural constraint, implementation
limitation, or a dependency that would break.

## Limitations

Known limits: input and output constraints, data-structure constraints,
performance and memory limits.
```

Omit a section outright when the code gives you nothing to put in it. An empty
"Extension points" heading tells the reader less than its absence does.
