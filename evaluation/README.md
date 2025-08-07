# ADK Agent Evaluation Framework

This directory contains a comprehensive agent evaluation system built using Google ADK evaluation patterns and frameworks.

## Overview

The evaluation framework provides:

- **Multi-metric evaluation**: Tool trajectory, response quality, and security-specific metrics
- **Standardized datasets**: Security-focused evaluation test cases
- **Automated benchmarking**: Performance and accuracy measurements
- **Integration with ADK**: Built on google.adk.evaluation patterns
- **Extensible architecture**: Easy to add new metrics and evaluators

## Structure

```
evaluation/
├── README.md
├── config/
│   ├── evaluation_config.yaml
│   └── test_config.json
├── datasets/
│   ├── security_agent_eval.json
│   ├── vulnerability_assessment.test.json
│   ├── compliance_check.test.json
│   └── incident_response.test.json
├── evaluators/
│   ├── __init__.py
│   ├── security_evaluator.py
│   ├── compliance_evaluator.py
│   └── performance_evaluator.py
├── metrics/
│   ├── __init__.py
│   ├── security_metrics.py
│   └── custom_metrics.py
├── runners/
│   ├── __init__.py
│   ├── evaluation_runner.py
│   └── batch_evaluator.py
└── results/
    └── reports/
```

## Usage

### Quick Start

```python
from evaluation.runners.evaluation_runner import AgentEvaluationRunner

# Run security agent evaluation
runner = AgentEvaluationRunner()
results = await runner.evaluate_agent(
    agent_module="agents.security_agent",
    eval_dataset_dir="evaluation/datasets"
)
```

### Custom Evaluation

```python
from evaluation.evaluators.security_evaluator import SecurityEvaluator

evaluator = SecurityEvaluator()
results = await evaluator.evaluate_security_response(
    query="Analyze this IAM policy for security vulnerabilities",
    response="Policy analysis with findings...",
    expected_findings=["overprivileged role", "missing MFA"]
)
```

## Metrics

### Core Metrics
- **Tool Trajectory Score**: Measures tool usage accuracy (ADK standard)
- **Response Match Score**: Rouge-1 text similarity (ADK standard)
- **Response Evaluation Score**: LLM-based quality assessment (ADK standard)

### Security-Specific Metrics
- **Vulnerability Detection Accuracy**: Precision/recall for security findings
- **Compliance Coverage**: Percentage of compliance requirements addressed
- **Risk Assessment Quality**: Accuracy of risk scoring and prioritization
- **Incident Response Time**: Speed and quality of security incident handling

### Performance Metrics
- **Response Latency**: Time to generate responses
- **Tool Execution Efficiency**: Tool usage optimization
- **Memory Usage**: Resource consumption tracking
- **Scalability**: Performance under load

## Configuration

Evaluation criteria are configured in `config/test_config.json`:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.9,
    "response_match_score": 0.8,
    "security_accuracy_score": 0.85,
    "compliance_coverage_score": 0.9
  }
}
```