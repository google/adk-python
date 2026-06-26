---
name: adk-unit-design
description: Creates or updates code unit design documents for source code documentation.
---

# ADK Code Unit Design

This skill creates or updates a detailed software engineering design document for new or updated code file or specified code unit. The design document it generates is meant to explain the code to a developer who wants to modify or extend the code unit as part of the ADK development framework. Similar to a *unit test*, a *unit design* provides a generated software engineering design based on the *actual, implemented code* rather than any proposed code design or proposed software architecture.

## Input

- Code files containing new functionality
- Names of new methods and classes (optional)
- Code files for base classes or interfaces that the new functionality depends on (optional)
- Code unit tests (optional)
- Example code files (optional)

## Analysis

- Review specified code files for changes and named methods to determine:
  - Purpose and intended use of the new or updated code units
  - Any data flows handled by the new or updated code units
  - Dependencies required by the new or updated code units
  - Approaches for extending or customizing the code unit to add new capabilities
  - Classes that depend on the new or updated code units
  - Operational limitations of the new or updated code units

## Output

- Look for an existing design document in the `/docs/design/***` directory of this repository.
  - If a design already exists, update the existing design incrementally and prioritize preserving the previous content as much as possible.
  - If no design document exists, create a design file for the new code unit in the `/docs/design/***` directory of this repository, using the relative path of the code unit. For example, if the code unit is called `/topic/function/class.ext`, create a design document in the location `/docs/design/topic/function/class/index.md`.
- Any links to local code files should be translated to URL links to the `google/adk-python` repository on GitHub. For example, if the local code unit path is `***/adk-python/topic/function/class.ext#L93`, the URL to the code file should be `https://github.com/google/adk-python/blob/main/topic/function/class.ext#L93`.

### Design document structure and content

Use the following structure and instructions to create the design document for the code unit:

```
# (name of code unit or code file) - Code Unit Design

- 2-sentence summary of the code unit

## Introduction

- Paragraph(s) explaining:
  - The purpose and application of the code unit, including intended use cases
  - Developer problems solved by this code unit
  - Agent capabilities enabled by this code unit

## High-level architecture

- Describe the software architecture of this code unit and how it fits into the larger ADK framework
- Explain general execution flow of this code unit
- Describe any data flows handled by the code unit including inputs and outputs
- Explain any cross-class dependencies of the code unit, including upstream dependencies and downstream dependencies

### Extension points

- Describe how the code unit could be extended or customized to add new features or capabilities
- Note specific parts of the code unit that are designed to be extended or customized, including:
  - Abstract classes
  - Interfaces
  - Hooks
  - Callbacks
  - Configurable parameters
  - Plugin architecture
  - Other extension points

### Extension constraints

- Describe what parts of the code unit should not be modified, based on:
  - architectural constraints
  - implementation limitations
  - cross-class dependencies
  - other constraints

## Limitations

- Mention any limitations of the code unit, if known, such as:
  - input constraints
  - data structure constraints
  - output constraints
  - performance limitations
  - memory limitations
  - other limitations

```
