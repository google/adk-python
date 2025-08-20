# Output Schema with Tools Sample Agent

This sample demonstrates how to use structured output (`output_schema`) alongside other tools in an ADK agent. Previously, this combination was not allowed, but now it's supported through a special processor that handles the interaction.

## How it Works

The agent combines:
- **Tools**: `search_wikipedia` and `get_current_year` for gathering information
- **Structured Output**: `PersonInfo` schema to ensure consistent response format

When both `output_schema` and `tools` are specified:
1. ADK automatically adds a special `set_model_response` tool
2. The model can use the regular tools for information gathering
3. For the final response, the model uses `set_model_response` with structured data
4. ADK extracts and validates the structured response

## Expected Response Format

The agent will return information in this structured format for user query "Tell me about Albert Einstein":

```json
{
  "name": "Albert Einstein",
  "age": 76,
  "occupation": "Theoretical Physicist",
  "location": "Princeton, New Jersey, USA",
  "biography": "German-born theoretical physicist who developed the theory of relativity..."
}
```

## Key Features Demonstrated

1. **Tool Usage**: Agent can search Wikipedia and get current year
2. **Structured Output**: Response follows strict PersonInfo schema
3. **Validation**: ADK validates the response matches the schema
4. **Flexibility**: Works with any combination of tools and output schemas
