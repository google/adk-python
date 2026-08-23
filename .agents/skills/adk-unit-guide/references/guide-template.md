# Unit guide template

Copy this structure into `docs/guides/{topic}/{unit}/index.md`. The bullets are
instructions for what to write in each section, not text to keep.

```markdown
# {unit_name}

Two-sentence summary of the code unit.

## Introduction

Prose covering the purpose and application of the unit, the key classes that
depend on it, and the developer problems it solves.

## Get started

A single minimal implementation demonstrating the unit, with enough of the
surrounding classes to show where the call belongs. Omit top-level imports and
main() runner boilerplate to keep the code snippet focused.

## How it works

How the unit accomplishes its purpose, the classes it depends on, the classes
that depend on it, and the cross-class interactions a caller will notice.

## Configuration options

A table of the options the unit itself introduces:

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `{option}` | `{type}` | `{default}` | What it controls. |

Follow the table with a paragraph per option covering real behaviour and usage
patterns, not a restatement of the type. Omit options inherited from a base
class, and do not enumerate every attribute and method — exhaustive API
reference belongs in the generated reference documentation.

## Advanced applications

Use cases beyond the minimum: the problem each solves and the implementation
for that circumstance. Omit the section when there are none.

## Limitations

Known limits of the unit.

## Related samples

Links to samples under `contributing/samples/` that exercise the unit, each
with a one-line description.
```

Omit a section outright when the code gives you nothing to put in it.
