# Config-based Agent Sample - Learning Assistant

This sample demonstrates a minimal multi-agent setup with a learning assistant that delegates to specialized tutoring agents.

## Structure

- `root_agent.yaml` - Main learning assistant agent that routes questions to appropriate tutors
- `code_tutor_agent.yaml` - Specialized agent for programming and coding questions
- `math_tutor_agent.yaml` - Specialized agent for mathematical concepts and problems

## Usage

The root agent will automatically delegate:
- Coding/programming questions → `code_tutor_agent`
- Math questions → `math_tutor_agent`

This example shows how to create a simple multi-agent system without tools, focusing on clear delegation and specialized expertise.

## Sample Queries

### Coding Questions

```
"How do I create a for loop in Python?"
"Can you help me debug this function?"
"What are the best practices for variable naming?"
```

### Math Questions

```
"Can you explain the quadratic formula?"
"How do I solve this algebra problem: 2x + 5 = 15?"
"What's the difference between mean and median?"
```
