# Log Probabilities Demo Agent

This sample demonstrates how to access and display log probabilities from language model responses using the new `avg_logprobs` and `logprobs_result` fields in `LlmResponse`.

## Overview

This simple example shows:

- **Log Probability Access**: How to extract `avg_logprobs` and `logprobs_result` from `LlmResponse`
- **After-Model Callback**: How to append log probability information to responses
- **Confidence Analysis**: How to interpret and display confidence metrics
- **Practical Usage**: Real-world example of accessing logprobs data

## How It Works

```
User Query → Agent Response → Log Probability Analysis Appended

1. User asks a question
2. Agent generates response with log probabilities enabled
3. After-model callback extracts avg_logprobs from LlmResponse
4. Callback appends log probability analysis to response content
5. User sees both the response and confidence information
```

## What You'll See

The agent response will include log probability analysis like:

```
[LOG PROBABILITY ANALYSIS]
📊 Average Log Probability: -0.23
🎯 Confidence Level: High
📈 Confidence Score: 79.4%
🔍 Top alternatives analyzed: 5
```

## Usage

### Basic Usage

```bash
# Run the agent in web UI
adk web contributing/samples

# Or run via CLI
adk run contributing/samples/logprobs
```

## Understanding Log Probabilities

- **Range**: -∞ to 0 (0 = 100% confident, -1 ≈ 37% confident, -2 ≈ 14% confident)
- **Confidence Levels**:
  - High: >= -0.5 (typically factual, straightforward responses)
  - Medium: -1.0 to -0.5 (reasonably confident responses)
  - Low: < -1.0 (uncertain or complex responses)
- **Use Cases**: Quality control, uncertainty detection, response filtering

## Key Fields in LlmResponse

- **`avg_logprobs`**: Average log probability across all tokens in the response
- **`logprobs_result`**: Detailed log probability information including top alternative tokens
