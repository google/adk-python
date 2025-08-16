# Migration Guide: From Custom to ADK-Compliant Evaluation

This guide helps migrate from the existing custom evaluation system to the ADK-compliant framework.

## Overview of Changes

### Before (Custom Implementation)
- Complex evaluator classes with custom metrics
- Security-specific scoring algorithms
- Custom invocation handling
- Non-standard test formats

### After (ADK-Compliant)
- Standard ADK evaluator pattern
- ADK metrics (tool trajectory, response match)
- Native ADK test formats
- Direct integration with ADK tools

## Migration Steps

### Step 1: Update Test File Formats

#### Old Format (Custom)
```json
{
  "eval_set_id": "vuln_assessment_v1",
  "eval_cases": [
    {
      "eval_id": "vuln_001",
      "conversation": [...]
    }
  ]
}
```

#### New Format (ADK-Compliant)
Already compatible! The existing format follows ADK evalset structure.

### Step 2: Replace Evaluators

#### Old Code
```python
from evaluators.security_evaluator import SecurityEvaluator

evaluator = SecurityEvaluator(threshold=0.8)
results = evaluator.evaluate_invocations(actual, expected)
```

#### New Code
```python
from adk_evaluator import ADKEvaluator, EvaluationCriteria

criteria = EvaluationCriteria(
    tool_trajectory_avg_score=0.9,
    response_match_score=0.8
)
evaluator = ADKEvaluator(criteria)
results = await evaluator.evaluate(
    agent_module="security_agent",
    eval_dataset_file_path_or_dir="datasets/"
)
```

### Step 3: Update Metrics Mapping

| Old Metric | ADK Metric | Notes |
|------------|------------|-------|
| `security_accuracy_score` | `response_match_score` | Use ROUGE-based matching |
| `vulnerability_detection_score` | `tool_trajectory_avg_score` | Track tool usage for detection |
| `compliance_coverage_score` | `response_evaluation_score` | LLM-based assessment |
| `risk_assessment_score` | Custom metric extension | Can be added as custom |

### Step 4: Update Runner Code

#### Old Runner
```python
from runners.evaluation_runner import AgentEvaluationRunner

runner = AgentEvaluationRunner()
results = await runner.evaluate_agent(
    agent_module="agents.security_agent",
    eval_dataset_dir="evaluation/datasets"
)
```

#### New ADK Pattern
```python
from adk_evaluator import ADKEvaluator

evaluator = ADKEvaluator()
results = await evaluator.evaluate(
    agent_module="security_agent",
    eval_dataset_file_path_or_dir="datasets/"
)
```

### Step 5: Update Test Scripts

#### Old Test Script
```python
async def test_security_evaluator():
    evaluator = SecurityEvaluator()
    score, status = evaluate_security_response(...)
```

#### New Test Script
```python
async def test_adk_evaluation():
    evaluator = ADKEvaluator()
    results = await evaluator.evaluate(...)
    assert all(r.passed for r in results)
```

## Preserving Custom Functionality

### Option 1: Extend ADK Evaluator

```python
from adk_evaluator import ADKEvaluator

class SecurityADKEvaluator(ADKEvaluator):
    def __init__(self, criteria=None):
        super().__init__(criteria)
        self.security_metrics = {}
    
    async def evaluate(self, agent_module, eval_dataset_file_path_or_dir, num_runs=1):
        # Run standard ADK evaluation
        results = await super().evaluate(
            agent_module, eval_dataset_file_path_or_dir, num_runs
        )
        
        # Add custom security metrics
        for result in results:
            result.custom_metrics = self._calculate_security_metrics(result)
        
        return results
    
    def _calculate_security_metrics(self, result):
        # Custom security scoring logic
        return {
            'vulnerability_detection': 0.9,
            'risk_assessment': 0.85
        }
```

### Option 2: Wrapper Pattern

```python
from adk_evaluator import ADKEvaluator
from evaluators.security_evaluator import SecurityEvaluator

class HybridEvaluator:
    def __init__(self):
        self.adk_evaluator = ADKEvaluator()
        self.security_evaluator = SecurityEvaluator()
    
    async def evaluate(self, agent_module, test_file):
        # Run ADK evaluation
        adk_results = await self.adk_evaluator.evaluate(
            agent_module, test_file
        )
        
        # Run security evaluation
        security_scores = self.security_evaluator.evaluate_security(
            test_file
        )
        
        # Combine results
        return {
            'adk_results': adk_results,
            'security_scores': security_scores
        }
```

## Configuration Migration

### Old Configuration (test_config.json)
```json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.9,
    "response_match_score": 0.8,
    "security_accuracy_score": 0.85,
    "vulnerability_detection_score": 0.9
  }
}
```

### New Configuration (ADK-Compliant)
```python
from adk_evaluator import EvaluationCriteria

criteria = EvaluationCriteria(
    tool_trajectory_avg_score=0.9,
    response_match_score=0.8,
    response_evaluation_score=0.75,
    custom_metrics={
        "security_accuracy": 0.85,
        "vulnerability_detection": 0.9
    }
)
```

## Testing the Migration

### 1. Parallel Testing
Run both old and new evaluators to compare:

```python
async def compare_evaluators():
    # Old evaluator
    old_runner = AgentEvaluationRunner()
    old_results = await old_runner.evaluate_agent(...)
    
    # New ADK evaluator
    adk_evaluator = ADKEvaluator()
    new_results = await adk_evaluator.evaluate(...)
    
    # Compare results
    print(f"Old score: {old_results.overall_score}")
    print(f"New score: {calculate_overall_score(new_results)}")
```

### 2. Gradual Migration
Migrate one test file at a time:

```python
# Phase 1: Migrate vulnerability tests
await test_with_adk("datasets/vulnerability_assessment.test.json")

# Phase 2: Migrate compliance tests
await test_with_adk("datasets/compliance_check.test.json")

# Phase 3: Migrate all remaining tests
await test_with_adk("datasets/")
```

### 3. Validation Checklist

- [ ] All test files load correctly with ADK evaluator
- [ ] Metrics produce comparable results
- [ ] Custom security metrics are preserved (if needed)
- [ ] Pytest integration works
- [ ] CI/CD pipeline updated
- [ ] Documentation updated

## Rollback Plan

If issues arise, you can temporarily use both systems:

```python
USE_ADK = os.getenv('USE_ADK_EVALUATOR', 'false').lower() == 'true'

if USE_ADK:
    evaluator = ADKEvaluator()
else:
    evaluator = SecurityEvaluator()  # Old system
```

## Benefits After Migration

1. **Standards Compliance**: Full compatibility with Google ADK ecosystem
2. **Simpler Code**: Less custom code to maintain
3. **Better Integration**: Works with ADK web UI and CLI tools
4. **Community Support**: Can use ADK community resources
5. **Future-Proof**: Automatic compatibility with ADK updates

## Common Issues and Solutions

### Issue: Different Score Ranges
**Problem**: Old system uses 0-100, ADK uses 0-1
**Solution**: Divide old scores by 100 or multiply ADK scores by 100

### Issue: Missing Custom Metrics
**Problem**: ADK doesn't have security-specific metrics
**Solution**: Use custom_metrics field or extend ADKEvaluator

### Issue: Tool Name Mismatches
**Problem**: Tool names differ between systems
**Solution**: Create mapping dictionary or update agent tool names

### Issue: Response Format Differences
**Problem**: Old system expects different response structure
**Solution**: Use response transformation in extended evaluator

## Support and Resources

- [ADK Documentation](https://google.github.io/adk-docs/evaluate/)
- [Migration Examples](./examples/)
- [Test Datasets](./datasets/)
- GitHub Issues: Report migration problems