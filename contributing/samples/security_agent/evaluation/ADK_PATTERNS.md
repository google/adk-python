# ADK Evaluation Patterns Guide

This document explains how our evaluation framework follows Google ADK patterns exactly.

## Core ADK Principles

### 1. Evaluation Philosophy
> "Unlike traditional software testing, LLM agents require qualitative evaluations of both output and decision-making trajectory."

Our implementation:
- Evaluates both final responses AND tool usage patterns
- Uses fuzzy matching (ROUGE scores) instead of exact matching
- Supports multiple runs for statistical significance

### 2. File Formats

#### Test Files (.test.json)
For simple, single-turn evaluations:

```json
{
  "user_content": {
    "parts": [{"text": "User query here"}]
  },
  "final_response": {
    "parts": [{"text": "Expected response"}]
  },
  "expected_tool_use": [],
  "creation_timestamp": 1754020000.0
}
```

#### Evalsets (.evalset.json)
For complex, multi-turn conversations:

```json
{
  "eval_set_id": "unique_id",
  "name": "Evalset Name",
  "eval_cases": [
    {
      "eval_id": "case_001",
      "conversation": [
        {
          "invocation_id": "inv_001",
          "user_content": {...},
          "final_response": {...},
          "expected_tool_use": []
        }
      ]
    }
  ]
}
```

## ADK Evaluation Methods

### 1. Web UI Evaluation
```bash
adk web
```
- Visual interface for evaluation
- Interactive testing
- Real-time results

### 2. Programmatic Testing
```python
await AgentEvaluator.evaluate(
    agent_module="my_agent",
    eval_dataset_file_path_or_dir="tests/my_test.test.json"
)
```

### 3. CLI Evaluation
```bash
adk eval --agent my_agent --dataset tests/
```

## Metrics Explained

### Tool Trajectory Score
**Purpose:** Ensure agents use the right tools in the right order

**Calculation:**
1. Compare each tool invocation step-by-step
2. Matching step = 1 point
3. Mismatched step = 0 points
4. Final score = average

**Example:**
```
Expected: [search, analyze, report]
Actual:   [search, analyze, report]
Score:    1.0 (perfect match)

Expected: [search, analyze, report]
Actual:   [search, report]
Score:    0.67 (2/3 match)
```

### Response Match Score
**Purpose:** Validate response content while allowing variations

**Calculation:**
- Uses ROUGE-1 (word overlap)
- Default threshold: 0.8 (80% similarity)
- Allows paraphrasing and reordering

**Example:**
```
Expected: "The system has a critical SQL injection vulnerability"
Actual:   "A critical SQL injection issue was found in the system"
Score:    0.85 (high overlap, passes 0.8 threshold)
```

### Response Evaluation Score
**Purpose:** LLM-based quality assessment

**Calculation:**
- Uses another LLM to evaluate response quality
- Considers completeness, accuracy, relevance
- More flexible than text matching

## Best Practices

### 1. Test Organization
```
datasets/
├── unit/                    # Simple, focused tests
│   ├── auth.test.json
│   └── validation.test.json
├── integration/             # Complex scenarios
│   ├── workflow.evalset.json
│   └── e2e.evalset.json
└── regression/              # Previous bug fixes
    └── issues.test.json
```

### 2. Criteria Selection

**Development Phase:**
```python
# More lenient for iteration
criteria = EvaluationCriteria(
    tool_trajectory_avg_score=0.8,
    response_match_score=0.7
)
```

**Production Phase:**
```python
# Stricter for deployment
criteria = EvaluationCriteria(
    tool_trajectory_avg_score=0.95,
    response_match_score=0.85
)
```

### 3. Multiple Runs
```python
# Run multiple times for consistency
results = await evaluator.evaluate(
    agent_module="my_agent",
    eval_dataset_file_path_or_dir="tests/",
    num_runs=5  # Statistical significance
)
```

## Integration Patterns

### With Pytest
```python
@pytest.mark.asyncio
async def test_agent_evaluation():
    passed = await evaluate_agent(
        agent_module="my_agent",
        test_file="tests/my_test.test.json"
    )
    assert passed
```

### With CI/CD
```yaml
# GitHub Actions example
- name: Run ADK Evaluation
  run: |
    python -m pytest evaluation/examples/pytest_integration.py
```

### With Monitoring
```python
# Track evaluation metrics over time
results = await evaluator.evaluate(...)
metrics.record('eval_score', results.overall_score)
metrics.record('tool_accuracy', results.tool_trajectory_score)
```

## Common Patterns

### 1. Security Agent Testing
```python
# Test vulnerability detection
test_data = {
    "user_content": {
        "parts": [{"text": "Analyze this code for SQL injection"}]
    },
    "final_response": {
        "parts": [{"text": "SQL injection found in query construction"}]
    }
}
```

### 2. Compliance Testing
```python
# Test compliance checking
test_data = {
    "user_content": {
        "parts": [{"text": "Check SOC 2 compliance"}]
    },
    "expected_tool_use": ["audit_tool", "report_generator"]
}
```

### 3. Multi-Turn Conversations
```python
# Evalset for complex interactions
evalset = {
    "eval_cases": [{
        "conversation": [
            # Turn 1: Initial query
            {...},
            # Turn 2: Follow-up
            {...},
            # Turn 3: Resolution
            {...}
        ]
    }]
}
```

## Troubleshooting

### Low Tool Trajectory Scores
- Check tool naming consistency
- Verify tool parameter matching
- Consider order sensitivity

### Low Response Match Scores
- Review expected response format
- Check for overly specific expectations
- Consider using response_evaluation_score instead

### Inconsistent Results
- Increase num_runs for stability
- Check for non-deterministic agent behavior
- Review test data quality

## Migration from Custom Evaluators

If migrating from custom evaluation:

1. **Map metrics to ADK standards:**
   - Custom accuracy → response_match_score
   - Custom tool checks → tool_trajectory_avg_score

2. **Convert test formats:**
   - Custom JSON → ADK test.json format
   - Test suites → evalset.json format

3. **Update thresholds:**
   - Review ADK default thresholds
   - Adjust based on your requirements

## Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/evaluate/)
- [ADK GitHub Repository](https://github.com/google/adk)
- [Example Test Files](./datasets/)
- [Integration Examples](./examples/)