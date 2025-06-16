# Structured Outputs in ADK

This example demonstrates how to implement structured outputs in the Agent Development Kit (ADK) using Pydantic models. The main agent in this example, `email_generator`, uses the `output_schema` parameter to ensure its responses conform to a specific structured format.

## What are Structured Outputs?

ADK allows you to define structured data formats for agent inputs and outputs using Pydantic models:

1. **Controlled Output Format**: Using `output_schema` ensures the LLM produces responses in a consistent JSON structure
2. **Data Validation**: Pydantic validates that all required fields are present and correctly formatted
3. **Improved Downstream Processing**: Structured outputs are easier to handle in downstream applications or by other agents

Use structured outputs when you need guaranteed format consistency for integration with other systems or agents.

## Email Generator Example

In this example, we've created an email generator agent that produces structured output with:

1. **Email Subject**: A concise, relevant subject line
2. **Email Body**: Well-formatted email content with greeting, paragraphs, and signature

The agent uses a Pydantic model called `EmailContent` to define this structure, ensuring every response follows the same format.

### Output Schema Definition

The Pydantic model defines exactly what fields are required and includes descriptions for each:

```python
class EmailContent(BaseModel):
    """Schema for email content with subject and body."""
    
    subject: str = Field(
        description="The subject line of the email. Should be concise and descriptive."
    )
    body: str = Field(
        description="The main content of the email. Should be well-formatted with proper greeting, paragraphs, and signature."
    )
```

### How It Works

1. The user provides a description of the email they need
2. The LLM agent processes this request and generates both a subject and body
3. The agent formats its response as a JSON object matching the `EmailContent` schema
4. ADK validates the response against the schema before returning it
5. The structured output is stored in the session state under the specified `output_key`

## Important Limitations

When using `output_schema`:

1. **No Tool Usage**: Agents with an output schema cannot use tools during their execution
2. **Direct JSON Response**: The LLM must produce a JSON response matching the schema as its final output
3. **Clear Instructions**: The agent's instructions must explicitly guide the LLM to produce properly formatted JSON

## Project Structure

```
4-structured-outputs/
│
├── email_agent/                   # Email Generator Agent package
│   └── agent.py                   # Agent definition with output schema
│
└── README.md                      # This documentation
```

## Getting Started

### Setup

1. Activate the virtual environment from the root directory:
```bash
# macOS/Linux:
source ../.venv/bin/activate
# Windows CMD:
..\.venv\Scripts\activate.bat
# Windows PowerShell:
..\.venv\Scripts\Activate.ps1
```

2. Create a `.env` file and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### Running the Example

```bash
cd 4-structured-outputs
adk web
```

Then select "email_generator" from the dropdown menu in the web UI.

## Example Interactions

Try these example prompts:

```
Write a professional email to my team about the upcoming project deadline that has been extended by two weeks.
```

```
Draft an email to a client explaining that we need additional information before we can proceed with their order.
```

```
Create an email to schedule a meeting with the marketing department to discuss the new product launch strategy.
```

## Key Concepts: Structured Data Exchange

Structured outputs are part of ADK's broader support for structured data exchange, which includes:

1. **input_schema**: Define expected input format (not used in this example)
2. **output_schema**: Define required output format (used in this example)
3. **output_key**: Store the result in session state for use by other agents (used in this example)

This pattern enables reliable data passing between agents and integration with external systems that expect consistent data formats.

## Additional Resources

- [ADK Structured Data Documentation](https://google.github.io/adk-docs/agents/llm-agents/#structuring-data-input_schema-output_schema-output_key)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/) 
