---
name: adk-unit-guide
description: Creates detailed code unit guides for source code documentation.
---

# ADK code unit guide
This skill creates a detailed developer guide for new or updated code file or direct code input. The guide it generates is meant to explain the code to a developer who wants to use it in an application, but with a higher level of technical detail than what would appear in published developer documentation. Similar to a *unit test*, a *unit guide* provides generated, granular-level documentation for a unit of code, without worrying about bloating the actual developer documentation with too many details.

## Input

- Code files containing new functionality
- Code unit tests (optional)
- Code design files (optional)
- Names of new methods and classes (optional)

## Analysis

- Review the code design files, if provided. Make note of:
  - Purpose and intended use of the new or updated code units
  - Classes that depend on the new or updated code units
  - Additional dependencies required by the new or updated code units
  - Limitations of the new or updated code units
- Review specified code file for changes and named methods, if provided.
- Determine what classes and code files may depend on the new or updated code units.

## Output

- Look for an existing guide in the `/docs/guides/***` directory of this repository.
  - If a guide already exists, update the existing guide incrementally and prioritize preserving the previous content as much as possible.
  - If no guide exists, create a guide file for the new code unit in the `/docs/guides/***` directory of this repository, using the relative path of the code unit. For example, if the code unit is called `/topic/function/class.ext`, create a guide in the location `/docs/guides/topic/function/class/index.md`.

### Guide structure and content

Use the following structure and instructions to create the guide for the code unit:

```
# Title: name of the code file or code unit

- 2-sentence summary of the code unit

## Introduction

- Paragraph(s) explaining:
  - The purpose and application of the code unit
  - Key classes that depend on this code unit
  - Developer problems solved by this code unit

## Get started

- Present a single, minimum implementation of the code unit to demonstrate its use.
- Show enough of the containing classes to make it clear where the code could be used.
- Use unit test code as a starting point for the code example, if available.

## How it works

- Explain how the code unit accomplishes its purpose or solves a problem.
- Mention key code classes that depend on this code unit.
- Mention code classes that this code unit depends on.
- Explain any cross-class dependencies of the code unit.

## Configuration options

- If the code unit has configuration options, document them in a table detailing parameters, types, default values, and descriptions.

## Advanced applications

- Determine if there are advanced use cases for the code unit.
- Add advanced applications of the code unit, including:
  - Problem solved
  - Implementations for special circumstances

## Limitations

- Mention any limitations of the code unit, if known.

```