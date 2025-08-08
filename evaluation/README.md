# 📊 ADK Agent Evaluation Framework

<div align="center">

[![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)]()
[![ADK](https://img.shields.io/badge/Built%20with-ADK-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)

**Comprehensive agent testing and benchmarking system**

[🚀 Quick Start](#-quick-start) • [📖 Usage](#-usage) • [📊 Metrics](#-metrics) • [⚙️ Configuration](#-configuration)

</div>

---

## 🎯 Overview

A comprehensive agent evaluation system built using Google ADK evaluation patterns and frameworks, providing standardized testing and benchmarking capabilities for intelligent agents.

### ✨ Key Features

- **📊 Multi-metric Evaluation** - Tool trajectory, response quality, and security-specific metrics
- **📋 Standardized Datasets** - Security-focused evaluation test cases  
- **🔄 Automated Benchmarking** - Performance and accuracy measurements
- **🧠 ADK Integration** - Built on google.adk.evaluation patterns
- **🔧 Extensible Architecture** - Easy to add new metrics and evaluators

## 🚀 Quick Start

```bash
# Navigate to evaluation framework
cd evaluation/

# Install dependencies  
pip install -r requirements.txt

# Run agent evaluation
python -m runners.evaluation_runner --agent security_agent
```

## 📁 Project Structure

<table>
<tr>
<th>Component</th>
<th>Location</th> 
<th>Purpose</th>
</tr>
<tr>
<td><strong>📋 Datasets</strong></td>
<td><code>datasets/</code></td>
<td>Test cases and evaluation scenarios</td>
</tr>
<tr>
<td><strong>🧪 Evaluators</strong></td>
<td><code>evaluators/</code></td>
<td>Metric calculation and assessment logic</td>
</tr>
<tr>
<td><strong>📊 Metrics</strong></td>
<td><code>metrics/</code></td>
<td>Custom metrics and scoring functions</td>
</tr>
<tr>
<td><strong>🏃 Runners</strong></td>
<td><code>runners/</code></td>
<td>Execution orchestration and batch processing</td>
</tr>
<tr>
<td><strong>⚙️ Config</strong></td>
<td><code>config/</code></td>
<td>Evaluation criteria and test configuration</td>
</tr>
</table>

## 📖 Usage

### Basic Agent Evaluation

```python
from evaluation.runners.evaluation_runner import AgentEvaluationRunner

# Run security agent evaluation
runner = AgentEvaluationRunner()
results = await runner.evaluate_agent(
    agent_module="agents.security_agent",
    eval_dataset_dir="evaluation/datasets"
)

print(f"Overall Score: {results.overall_score}")
print(f"Tool Trajectory: {results.tool_trajectory_score}")
```

### Custom Security Evaluation

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