# ADK Agent Evaluation System - Usage Guide

## Quick Start

### 1. Basic Agent Evaluation

```python
import asyncio
from evaluation.runners.evaluation_runner import run_security_evaluation

# Evaluate a security agent
results = await run_security_evaluation(
    agent_module="agents.security_agent",
    dataset_dir="evaluation/datasets"
)

print(f"Overall Score: {results.overall_score:.3f}")
print(f"Status: {results.overall_status}")
```

### 2. Custom Evaluation Configuration

```python
from evaluation.runners.evaluation_runner import AgentEvaluationRunner, EvaluationConfig

config = EvaluationConfig(
    agent_module="agents.my_security_agent",
    eval_dataset_dir="my_datasets",
    num_runs=3,
    metrics={
        "security_accuracy_score": {"threshold": 0.85, "weight": 0.4},
        "vulnerability_detection_score": {"threshold": 0.9, "weight": 0.6}
    },
    output_formats=["json", "html", "csv"]
)

runner = AgentEvaluationRunner()
results = await runner.evaluate_agent(config=config)
```

### 3. Individual Evaluator Testing

```python
from evaluation.evaluators.security_evaluator import evaluate_security_response
from evaluation.evaluators.compliance_evaluator import evaluate_compliance_response

# Test security response
score, status = evaluate_security_response(
    query="Analyze this code for SQL injection vulnerabilities",
    actual_response="SQL injection found in line 42...",
    expected_response="Critical SQL injection vulnerability detected..."
)

# Test compliance response  
score, status = evaluate_compliance_response(
    query="Review for SOC 2 compliance",
    actual_response="SOC 2 analysis shows gaps in access controls...",
    expected_response="SOC 2 Trust Service Criteria assessment..."
)
```

## Evaluation Datasets

### Dataset Format (ADK EvalSet)

```json
{
  "eval_set_id": "unique_eval_set_id",
  "name": "Descriptive Name",
  "description": "Description of what this evaluates",
  "eval_cases": [
    {
      "eval_id": "unique_case_id",
      "conversation": [
        {
          "invocation_id": "unique_invocation_id",
          "user_content": {
            "parts": [{"text": "User query here..."}]
          },
          "final_response": {
            "parts": [{"text": "Expected agent response..."}]
          },
          "expected_tool_use": [],
          "creation_timestamp": 1754020000.0
        }
      ],
      "creation_timestamp": 1754020000.0
    }
  ],
  "creation_timestamp": 1754020000.0
}
```

### Creating Custom Datasets

1. **Security Vulnerability Dataset**
   - Focus on code analysis, vulnerability detection
   - Include various vulnerability types (SQL injection, XSS, etc.)
   - Provide detailed remediation guidance

2. **Compliance Framework Dataset**  
   - Cover major frameworks (SOC 2, PCI DSS, GDPR, HIPAA)
   - Include gap analysis scenarios
   - Focus on control mapping and requirements

3. **Incident Response Dataset**
   - Real-world incident scenarios
   - Include containment, investigation, recovery steps
   - Test decision-making under pressure

## Evaluation Metrics

### Standard ADK Metrics

- **Tool Trajectory Score**: Measures accuracy of tool usage (threshold: 0.9)
- **Response Match Score**: Rouge-1 text similarity (threshold: 0.8)  
- **Response Evaluation Score**: LLM-based quality assessment (threshold: 0.75)

### Security-Specific Metrics

- **Security Accuracy Score**: Overall security analysis accuracy (threshold: 0.85)
- **Vulnerability Detection Score**: Precision/recall for finding vulnerabilities (threshold: 0.9)
- **Risk Assessment Score**: Quality of risk analysis and prioritization (threshold: 0.8)

### Compliance Metrics

- **Compliance Coverage Score**: Framework requirement coverage (threshold: 0.9)
- **Control Mapping Score**: Accuracy of control identification (threshold: 0.85)
- **Gap Analysis Score**: Quality of gap identification and remediation (threshold: 0.8)

## Configuration Options

### Evaluation Config (evaluation_config.yaml)

```yaml
evaluation:
  num_runs: 3
  timeout_seconds: 300
  parallel_execution: true
  
  metrics:
    security_accuracy:
      enabled: true
      weight: 0.4
      threshold: 0.85
    
    vulnerability_detection:
      enabled: true
      weight: 0.3
      threshold: 0.9
    
    compliance_coverage:
      enabled: true
      weight: 0.3
      threshold: 0.9

reporting:
  output_formats: ["json", "html", "csv"]
  include_details: true
  export_path: "evaluation/results"
```

### Test Config (test_config.json)

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.9,
    "response_match_score": 0.8,
    "security_accuracy_score": 0.85
  },
  "evaluation_settings": {
    "num_runs": 3,
    "timeout_seconds": 300,
    "enable_parallel": true
  }
}
```

## Running Evaluations

### Command Line Usage

```bash
# Run all evaluations
python -m evaluation.runners.evaluation_runner \
    --agent-module agents.security_agent \
    --dataset-dir evaluation/datasets \
    --output-dir evaluation/results

# Run specific dataset
python -m evaluation.runners.evaluation_runner \
    --agent-module agents.security_agent \
    --dataset evaluation/datasets/vulnerability_assessment.test.json

# Run with custom config
python -m evaluation.runners.evaluation_runner \
    --config config/evaluation_config.yaml
```

### Programmatic Usage

```python
# Security agent evaluation
from evaluation.runners.evaluation_runner import run_security_evaluation

results = await run_security_evaluation(
    agent_module="agents.security_agent"
)

# Compliance agent evaluation  
from evaluation.runners.evaluation_runner import run_compliance_evaluation

results = await run_compliance_evaluation(
    agent_module="agents.compliance_agent"
)

# Custom evaluation
from evaluation.runners.evaluation_runner import AgentEvaluationRunner

runner = AgentEvaluationRunner("config/custom_eval.yaml")
results = await runner.evaluate_agent(
    agent_module="agents.custom_agent",
    eval_dataset_dir="custom_datasets"
)
```

## Integration with ADK

### Using Standard ADK Evaluator

```python
from google.adk.evaluation.agent_evaluator import AgentEvaluator

# Standard ADK evaluation
await AgentEvaluator.evaluate(
    agent_module="agents.security_agent",
    eval_dataset_file_path_or_dir="evaluation/datasets",
    num_runs=3
)
```

### Extending ADK Patterns

```python
from google.adk.evaluation.evaluator import Evaluator, EvaluationResult
from google.adk.evaluation.eval_set import EvalSet

class CustomSecurityEvaluator(Evaluator):
    def evaluate_invocations(self, actual, expected):
        # Custom evaluation logic
        return EvaluationResult(...)
```

## Best Practices

### Dataset Creation

1. **Realistic Scenarios**: Use real-world security and compliance scenarios
2. **Diverse Coverage**: Include various vulnerability types and compliance frameworks  
3. **Clear Expectations**: Provide detailed expected responses
4. **Progressive Complexity**: Range from basic to advanced scenarios

### Evaluation Configuration

1. **Appropriate Thresholds**: Set realistic performance thresholds
2. **Balanced Metrics**: Weight metrics according to importance
3. **Multiple Runs**: Use 3+ runs for reliable results
4. **Comprehensive Coverage**: Test all agent capabilities

### Performance Optimization

1. **Parallel Execution**: Enable for faster evaluation
2. **Batch Processing**: Group similar evaluations
3. **Caching**: Cache expensive operations
4. **Resource Management**: Monitor memory and CPU usage

### Result Analysis

1. **Trend Analysis**: Track performance over time
2. **Failure Analysis**: Investigate low scores
3. **Comparative Analysis**: Compare different models/approaches
4. **Continuous Improvement**: Use results to improve agents

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Dataset Validation**: Check JSON format and required fields
3. **Agent Module**: Verify agent module path and interface
4. **Permissions**: Check file system permissions

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
runner = AgentEvaluationRunner()
results = await runner.evaluate_agent(...)
```

### Performance Issues

1. **Reduce num_runs** for faster iteration
2. **Disable parallel_execution** if memory limited
3. **Use subset of datasets** for quick testing
4. **Implement caching** for expensive operations

## Advanced Usage

### Custom Metrics

```python
from evaluation.metrics.custom_metrics import CustomMetricEvaluator

class MyCustomMetric(CustomMetricEvaluator):
    def _evaluate_single_invocation(self, actual, expected):
        # Custom scoring logic
        score = self.calculate_custom_score(actual, expected)
        return PerInvocationResult(...)
```

### Batch Evaluation

```python
from evaluation.runners.batch_evaluator import BatchEvaluator

batch = BatchEvaluator()
results = await batch.evaluate_multiple_agents([
    "agents.security_agent_v1",
    "agents.security_agent_v2", 
    "agents.security_agent_v3"
])
```

### Integration Testing

```python
# Test evaluation system components
python evaluation/simple_test.py

# Full integration test
python evaluation/test_evaluation_system.py
```