# Security Agent Evaluation Framework

This directory contains a comprehensive evaluation framework for the GCP Security Agent using ADK's evaluation capabilities.

## Overview

The evaluation framework tests the agent's ability to:
- Identify security vulnerabilities and misconfigurations
- Provide accurate security assessments
- Generate appropriate remediation recommendations
- Handle various security scenarios correctly

## Structure

```
evaluation/
├── config/
│   └── evaluation_config.yaml     # Evaluation configuration
├── datasets/                      # Test datasets
│   ├── iam_security.evalset.json         # IAM security tests
│   ├── storage_security.evalset.json     # Storage security tests  
│   ├── network_security.evalset.json     # Network security tests
│   ├── vulnerability_assessment.evalset.json # Vulnerability tests
│   └── compliance_check.evalset.json     # Compliance tests
├── runners/
│   └── evaluation_runner.py       # Evaluation orchestrator
└── results/                       # Generated reports (created at runtime)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install google-adk[eval]
```

### 2. Test Basic Setup

```bash
python test_adk_eval.py
```

### 3. Run Full Evaluation

```bash
# Run all evaluations
python run_evaluation.py --evaluation-type full

# Run security-focused evaluation only
python run_evaluation.py --evaluation-type security

# Run with custom config
python run_evaluation.py --config-file evaluation/config/evaluation_config.yaml
```

### 4. View Results

Results are saved in `evaluation/results/` in multiple formats:
- JSON: Machine-readable detailed results
- HTML: Human-readable report with visualizations  
- CSV: Tabular data for analysis

## Evaluation Datasets

### IAM Security (`iam_security.evalset.json`)
Tests the agent's ability to:
- Identify overprivileged users and service accounts
- Detect risky IAM configurations
- Recommend proper access controls

### Storage Security (`storage_security.evalset.json`)
Tests the agent's ability to:
- Find public storage buckets
- Assess encryption configurations
- Recommend storage security best practices

### Network Security (`network_security.evalset.json`)
Tests the agent's ability to:
- Identify overly permissive firewall rules
- Assess network segmentation
- Recommend network security improvements

### Vulnerability Assessment (`vulnerability_assessment.evalset.json`)
Tests the agent's ability to:
- Identify critical security findings
- Assess vulnerability severity
- Provide comprehensive remediation guidance

### Compliance Check (`compliance_check.evalset.json`)
Tests the agent's ability to:
- Assess SOC 2 compliance
- Evaluate GDPR compliance posture
- Provide framework-specific recommendations

## Configuration

### Metrics

The evaluation uses several metrics:

- **Tool Trajectory Score** (30%): Measures if the agent calls the right tools
- **Response Match Score** (20%): Measures response quality against expected responses
- **Security Accuracy Score** (30%): Custom metric for security assessment accuracy
- **Vulnerability Detection Score** (20%): Custom metric for vulnerability identification

### Thresholds

- Tool trajectory: 90% threshold
- Response match: 80% threshold  
- Security accuracy: 85% threshold
- Vulnerability detection: 90% threshold

## Adding New Test Cases

### 1. Create New Dataset

```json
{
  "eval_set_id": "my_test_evaluation",
  "name": "My Security Test",
  "description": "Tests specific security scenario",
  "eval_cases": [
    {
      "eval_id": "test_case_1",
      "conversation": [
        {
          "invocation_id": "test-1",
          "user_content": {
            "parts": [{"text": "Your test query"}],
            "role": "user"
          },
          "expected_final_response": {
            "parts": [{"text": "Expected agent response"}],
            "role": "assistant"
          },
          "expected_tool_calls": [
            {
              "name": "query_security_data",
              "args": {"query_type": "security_summary"}
            }
          ]
        }
      ],
      "session_input": {
        "app_name": "vertex_sqlite",
        "user_id": "evaluator",
        "state": {}
      }
    }
  ]
}
```

### 2. Save to datasets/ directory

Name the file with `.evalset.json` extension.

### 3. Run evaluation

The runner will automatically discover and include the new dataset.

## Custom Metrics

You can add custom security-specific metrics by:

1. Creating metric evaluators in `evaluators/`
2. Updating `evaluation_config.yaml`
3. Modifying the evaluation runner

## Best Practices

### Test Case Design
- Cover both positive and negative scenarios
- Include edge cases and error conditions
- Test multi-turn conversations
- Validate tool usage patterns

### Expected Responses
- Include specific security recommendations
- Cover remediation steps
- Include risk assessments
- Maintain consistent terminology

### Evaluation Criteria
- Set realistic but challenging thresholds
- Weight metrics based on importance
- Include both automated and manual validation
- Regular threshold calibration

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `google-adk[eval]` is installed
2. **Agent not found**: Check module path in configuration
3. **Dataset format errors**: Validate JSON syntax and schema
4. **Low scores**: Review expected responses and tool calls

### Debug Mode

Enable debug logging:

```bash
export GOOGLE_ADK_LOG_LEVEL=DEBUG
python run_evaluation.py
```

### Manual Testing

Test individual conversations:

```bash
cd agents/gcp_security
adk web
# Navigate to agent in browser and test manually
```

## CI/CD Integration

Add evaluation to your CI/CD pipeline:

```yaml
- name: Run Security Agent Evaluation
  run: |
    python run_evaluation.py --evaluation-type security
    # Fail if score below threshold
```

## Reporting

### HTML Reports
- Visual charts and graphs
- Color-coded results
- Detailed conversation traces
- Metric breakdowns

### JSON Reports  
- Machine-readable format
- API integration friendly
- Detailed metadata
- Timestamp tracking

### CSV Reports
- Spreadsheet analysis
- Trend tracking
- Comparative analysis
- Metric correlation

## Performance Monitoring

Track evaluation metrics over time:
- Score trends
- Regression detection
- Performance benchmarks
- Quality improvements

The evaluation framework provides comprehensive testing to ensure the security agent maintains high quality and accuracy in its security assessments and recommendations.