# LiteLLM Tests

This directory contains tests for the LiteLLM integration, including tests for the infinite loop fix.

## Test Files

- `test_litellm_patch.py`: Unit tests for the robust JSON parsing functionality and loop detection attributes
- `system_test_litellm.py`: System test for verifying the loop detection mechanism with a simulated conversation

## Running Tests

To run the unit tests:

```bash
python -m tests.litellm.test_litellm_patch
```

To run the system test:

```bash
python -m tests.litellm.system_test_litellm
```

## Test Description

These tests validate two key components of the LiteLLM infinite loop fix:

1. **Robust JSON Parsing**: Tests that malformed JSON in function call arguments can be properly parsed
2. **Loop Detection**: Tests that repeated calls to the same function are detected and broken after exceeding the threshold

For more information on the LiteLLM loop fix, see the [documentation](../../docs/litellm_loop_fix.md). 