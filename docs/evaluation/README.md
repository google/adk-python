# 📊 ADK Agent Evaluation Framework

<div align="center">

[![Status](https://img.shields.io/badge/Status-Production-green.svg)]()
[![ADK](https://img.shields.io/badge/Google%20ADK-Compliant-blue.svg)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)

**Google ADK-Compliant Agent Evaluation System**

[🚀 Quick Start](#-quick-start) • [📖 Usage](#-usage) • [📊 Metrics](#-metrics) • [🧪 Testing](#-testing)

</div>

---

## 🎯 Overview

A fully ADK-compliant evaluation framework that follows Google Agent Development Kit patterns for testing and benchmarking AI agents. This implementation provides standardized evaluation metrics, supports both test files and evalsets, and integrates seamlessly with existing ADK workflows.

### ✨ Key Features

- **🎯 ADK-Compliant** - Exact implementation of Google ADK evaluation patterns
- **📊 Standard Metrics** - Tool trajectory, response match, and response evaluation scores
- **📋 Dual Format Support** - Both `.test.json` and `.evalset.json` formats
- **🔄 Multiple Evaluation Modes** - Web UI, CLI, and programmatic (pytest) testing
- **🧪 Test & Evalset Patterns** - Simple tests for unit testing, evalsets for integration
- **⚡ Async Support** - Built with asyncio for efficient parallel evaluation

## 🚀 Quick Start

### ADK Pattern Usage

```python
# Following exact ADK documentation pattern
from adk_evaluator import ADKEvaluator

async def evaluate():
    await ADKEvaluator.evaluate(
        agent_module="security_agent",
        eval_dataset_file_path_or_dir="datasets/vulnerability_assessment.test.json"
    )
```

### Command Line

```bash
# Web UI evaluation (ADK standard)
adk web

# CLI evaluation
python examples/simple_agent_test.py

# Pytest integration
pytest examples/pytest_integration.py
```

## 📁 Project Structure (ADK-Compliant)

```
evaluation/
├── adk_evaluator.py          # Main ADK-compliant evaluator
├── datasets/                  # Evaluation datasets
│   ├── *.test.json           # Single test files (unit testing)
│   └── *.evalset.json        # Evalset files (integration testing)
├── examples/                  # Usage examples
│   ├── simple_agent_test.py  # Basic evaluation patterns
│   ├── pytest_integration.py # Pytest integration
│   └── web_ui_example.py     # Web UI simulation
├── evaluators/               # Extended evaluators
│   ├── security_evaluator.py # Security-specific metrics
│   └── compliance_evaluator.py # Compliance metrics
├── config/                   # Configuration
│   └── test_config.json     # Evaluation criteria
└── README.md                 # This file
```

## 📖 Usage Patterns

### 1. Test File Approach (Simple, Unit Testing)

```python
from adk_evaluator import ADKEvaluator

# Single test file evaluation
evaluator = ADKEvaluator()
results = await evaluator.evaluate(
    agent_module="security_agent",
    eval_dataset_file_path_or_dir="datasets/simple_test.test.json"
)
```

### 2. Evalset Approach (Complex, Integration Testing)

```python
# Multiple conversation evaluation
results = await evaluator.evaluate(
    agent_module="security_agent",
    eval_dataset_file_path_or_dir="datasets/complex.evalset.json",
    num_runs=3  # Multiple runs for consistency
)
```

### 3. Directory Evaluation (Comprehensive)

```python
# Evaluate all test files in directory
results = await evaluator.evaluate(
    agent_module="security_agent",
    eval_dataset_file_path_or_dir="datasets/"
)
```

### 4. Custom Criteria

```python
from adk_evaluator import EvaluationCriteria

criteria = EvaluationCriteria(
    tool_trajectory_avg_score=0.9,  # 90% tool accuracy
    response_match_score=0.8,        # 80% response similarity
    response_evaluation_score=0.75   # 75% quality score
)

evaluator = ADKEvaluator(criteria)
results = await evaluator.evaluate(...)
```

## 📊 ADK Standard Metrics

### Core Metrics (Required)

| Metric | Description | Default Threshold | Calculation |
|--------|-------------|-------------------|-------------|
| `tool_trajectory_avg_score` | Compares actual vs expected tool usage | 1.0 (100%) | Matching steps / Total steps |
| `response_match_score` | ROUGE-based text similarity | 0.8 (80%) | Word overlap ratio |
| `response_evaluation_score` | LLM-based quality assessment | 0.75 (75%) | Model evaluation score |

### How Metrics Work

**Tool Trajectory Score:**
- Each matching tool invocation = 1 point
- Each mismatch = 0 points
- Final score = average of all steps

**Response Match Score:**
- Uses ROUGE-1 metric for text similarity
- Default threshold allows for minor variations
- Compares semantic content, not exact text

**Key Principle:** *"Unlike traditional software testing, LLM agents require qualitative evaluations of both output and decision-making trajectory."* - Google ADK Docs

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