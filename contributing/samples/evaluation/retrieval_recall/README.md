# Retrieval recall (custom metric)

## Overview

This sample scores a retrieval step: did the agent's search tool return the
documents it needed to answer the question?

The built-in criteria check which tools were called
(`tool_trajectory_avg_score`) and whether the final answer is grounded
(`hallucinations_v1`). Neither checks what a retrieval tool *returned*. An agent
can call the right tool with a poor query, get the wrong documents, and still
pass a trajectory check.

`retrieval_recall.py` defines `retrieval_recall`, a custom metric that compares
the document ids in the agent's actual `search_docs` responses with the ids
listed in the eval case. The score for an invocation is the share of expected
documents that were retrieved. Extra documents do not lower the score; recall
only asks whether the needed ones were found.

The metric is deterministic. It needs no judge model.

## Sample Inputs

The agent (`../support_kb_agent/`) answers customer support questions from six
in-memory articles. Its `search_docs` tool ranks articles by word overlap with
the query and returns the best two.

The eval set (`support_kb.evalset.json`) contains four single-turn cases:

- `How many days do I have to return an item?` expects `returns-policy`
- `I want to return my headphones. What is the returns policy and when will my
  refund arrive?` expects `returns-policy` and `refund-timing`
- `How long does international shipping take?` expects
  `international-shipping`
- `How do I reset my password?` expects `account-password`

## How To

### Recording the expected documents

The expected document ids live in the eval case, in the `tool_responses` of the
expected invocation's `intermediate_data`:

```json
"intermediate_data": {
  "tool_uses": [
    {"name": "search_docs", "args": {"query": "return item days"}}
  ],
  "tool_responses": [
    {"name": "search_docs",
     "response": {"documents": [{"id": "returns-policy"}]}}
  ],
  "intermediate_responses": []
}
```

Only the `id` of each document is needed. List the documents that are required
to answer the question, not everything the tool might return.

### The metric function

`retrieval_recall` has the custom metric signature described in the
[`custom_metric`](../custom_metric/) sample. Inside it:

- `get_all_tool_responses(invocation.intermediate_data)` returns the
  `google.genai.types.FunctionResponse` objects of an invocation, for both the
  actual and the expected invocation.
- Responses from tools other than `search_docs` are ignored.
- A response without a `documents` list, such as a tool error, counts as nothing
  retrieved instead of raising.
- An invocation whose eval case expects no documents is left unscored, so turns
  that need no retrieval do not change the average.

### Registering the metric

`eval_config.json` wires the metric in through `custom_metrics`:

```json
{
  "criteria": {
    "retrieval_recall": 1.0
  },
  "custom_metrics": {
    "retrieval_recall": {
      "code_config": {"name": "retrieval_recall.retrieval_recall"},
      "description": "Share of the expected documents that search_docs actually returned."
    }
  }
}
```

### Running the sample

Run it from the workspace root, with this folder on `PYTHONPATH` so that
`adk eval` can import the metric module:

```bash
PYTHONPATH=contributing/samples/evaluation/retrieval_recall \
adk eval contributing/samples/evaluation/support_kb_agent \
    contributing/samples/evaluation/retrieval_recall/support_kb.evalset.json \
    --config_file_path contributing/samples/evaluation/retrieval_recall/eval_config.json \
    --print_detailed_results
```

Each case reports a line such as:

```
Metric: retrieval_recall, Status: PASSED, Score: 1.0, Threshold: 1.0
```

A case fails when the agent's query misses a needed document. For the two
document case, retrieving only `returns-policy` scores 0.5 (FAILED).

### Limits

- The metric measures recall only. It does not penalize irrelevant documents
  (precision) or reward ranking the best document first.
- It reads documents from a function tool's response. Results that a model-side
  search tool reports as grounding metadata are not covered.
- The tool name `search_docs` and the `documents` / `id` response shape are
  fixed in `retrieval_recall.py`. Change them to match your own retrieval tool.

## Related Guides

- Evaluation overview: https://adk.dev/evaluate/
- Evaluation criteria reference: https://adk.dev/evaluate/criteria/
