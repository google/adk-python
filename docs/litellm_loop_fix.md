# LiteLLM Infinite Loop Fix

## Overview

This document describes a fix implemented to address an infinite loop issue that occurs when using ADK (Agent Development Kit) with Ollama/Gemma3 models via the LiteLLM integration.

## Problem Description

When using certain models like Ollama/Gemma3 through LiteLLM, the system could enter an infinite loop under the following conditions:

1. The model makes a function call with arguments
2. The function executes and returns a result
3. The model tries to make another function call, but with malformed JSON in the arguments
4. Due to the malformed JSON, the system gets stuck repeating the same function call

This issue caused the system to become unresponsive and waste resources, as the model would continuously attempt to call the same function without making progress.

## Solution

The fix addresses the issue through two main components:

### 1. Robust JSON Parsing

The enhanced `_model_response_to_generate_content_response` function now includes:

- Comprehensive validation for required fields with proper defaults
- Multiple strategies for parsing malformed JSON:
  - Standard JSON parsing
  - Single quote replacement
  - Regex-based fixes for common JSON formatting issues
- Graceful fallback to empty dictionaries when parsing fails
- Improved error handling to prevent crashes

### 2. Loop Detection Mechanism

The `generate_content_async` method in the `LiteLlm` class now includes:

- Tracking of consecutive calls to the same function
- Detection when the same function is called more than a threshold number of times (default: 5)
- Interruption of potential infinite loops when detected
- Generation of helpful user-facing messages that explain the issue
- Inclusion of relevant context from function calls to assist the user

## Implementation Details

The implementation preserves compatibility with all existing ADK functionality while adding the new safety mechanisms. The loop detection is efficient and adds minimal overhead to normal operation.

### Configuration

The loop detection threshold can be adjusted by modifying the `_loop_threshold` class variable in the `LiteLlm` class. The default value is 5, which strikes a balance between allowing legitimate repeated function calls and identifying problematic loops.

### Testing

The fix has been validated through:

1. Unit tests for robust JSON parsing
2. Integration tests to verify loop detection
3. Manual system testing to ensure compatibility with existing workflows

## Conclusion

This fix makes the LiteLLM integration more robust, particularly when using models that may produce malformed JSON or get stuck in repetitive patterns. It improves reliability and user experience by preventing infinite loops and providing helpful context when issues are detected. 