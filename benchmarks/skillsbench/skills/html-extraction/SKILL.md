---
name: html-extraction
description: Extract structured data from HTML content using CSS-like selectors.
---

# HTML Extraction Skill

Parse HTML content and extract text, links, or table data using tag-based selectors.

## Available Scripts

### `extract.py`

Extracts content from embedded sample HTML based on a target selector.

**Usage**: `execute_skill_script(skill_name="html-extraction", script_name="extract.py", input_args="target=links")`

Supported targets:
- `target=links`: Extract all hyperlinks (text and href)
- `target=headings`: Extract all heading text
- `target=table`: Extract table data as CSV
- `target=text`: Extract all visible text content

**Output format**: One extracted item per line.

## References

- [sample-page.md](./references/sample-page.md) — Sample HTML page for testing

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `load_skill_resource` to see the sample HTML page.
3. Use `execute_skill_script` with the desired target to extract data.
4. Present the extracted content to the user.
