# Common Patterns for ADK Development

This document provides common patterns and best practices discovered while building ADK agents.

## Handling File Uploads in Agent Tools

When building agents that process files, you often need to handle both:
1. Files uploaded directly through the chat interface
2. File paths provided as text by the user

### Pattern: Flexible File Path Handling

Tools should accept both display names (from uploaded files) and full file paths:

```python
import os

def parse_document(doc_path: str = "uploaded_document") -> str:
  """Parses a document (uploaded file or file path).

  Args:
    doc_path: Display name of uploaded file or full file path

  Returns:
    Success message with output path
  """
  # Handle both uploaded file identifiers and full paths
  if '/' in doc_path:
    # Full path provided
    run_id = os.path.basename(os.path.dirname(doc_path))
  else:
    # Uploaded file display name or simple identifier
    run_id = doc_path.replace('.', '_')

  output_path = f"output/{run_id}.xml"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)

  # ... your parsing logic here

  return f"Successfully parsed document to {output_path}"
```

### Agent Instructions

Your agent instructions should explicitly handle both cases:

```python
from google.adk import Agent

agent = Agent(
  name="DocumentParser",
  model="gemini-2.0-flash",
  instruction="""
  You are a document parsing agent.

  When the user provides files:
  1. If files are uploaded directly in the chat:
     - Acknowledge them by their display names
     - Call parsing tools with their display names or identifiers
     - You can identify uploaded files by their presence as attachments
  2. If file paths are provided in text:
     - Call parsing tools with the exact paths provided

  Do not ask for file paths if files are already uploaded.
  """,
  tools=[parse_document]
)
```

### Why This Matters

Without explicit instructions, agents may:
- Ask for file paths even when files are already uploaded
- Fail to recognize uploaded files as attachments
- Pass incorrect parameters to parsing tools

This pattern ensures your agent gracefully handles both upload methods.

### Example Usage

**Uploaded File**:
```
User: [Uploads: report.pdf]
       Parse this document

Agent: [Calls parse_document("report.pdf")]
```

**File Path**:
```
User: Parse the document at /path/to/reports/quarterly_report.pdf

Agent: [Calls parse_document("/path/to/reports/quarterly_report.pdf")]
```

## Contributing More Patterns

Have a useful pattern to share? Consider contributing:
1. Add your pattern to this document
2. Include code examples
3. Explain why it matters
4. Submit a PR following [CONTRIBUTING.md](../CONTRIBUTING.md)
