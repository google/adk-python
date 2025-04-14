#!/usr/bin/env python
"""
Unit tests for the LiteLLM infinite loop fix.

These tests verify:
1. The robust JSON parsing functionality for handling malformed function call arguments
2. The presence of loop detection attributes in the LiteLlm class
"""

import sys
import logging
from typing import Dict, Any
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_litellm_patch")

# Import LiteLLM components
from src.google.adk.models.lite_llm import (
    LiteLlm,
    _model_response_to_generate_content_response,
    ModelResponse
)

def test_robust_json_parsing():
    """Test the robust JSON parsing functionality."""
    logger.info("Testing robust JSON parsing...")
    
    # Test with malformed JSON (single quotes)
    response = ModelResponse(
        id="test_id",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_dealers",
                                "arguments": "{'param': 'value'}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ]
    )
    
    result = _model_response_to_generate_content_response(response)
    
    # Verify the result
    if result.content.parts[0].function_call.name == "get_dealers" and \
       isinstance(result.content.parts[0].function_call.args, dict) and \
       "param" in result.content.parts[0].function_call.args and \
       result.content.parts[0].function_call.args["param"] == "value":
        logger.info("✅ Successfully parsed single-quoted JSON")
    else:
        logger.error("❌ Failed to parse single-quoted JSON")

def test_loop_detection():
    """Test the loop detection mechanism."""
    # This would be a more complex test that requires setting up a mock conversation
    # For now, we'll just verify the class has the required attributes
    logger.info("Verifying loop detection attributes...")
    
    llm = LiteLlm(model="test_model")
    
    if hasattr(llm, "_consecutive_tool_calls") and \
       hasattr(llm, "_last_tool_call_name") and \
       hasattr(llm, "_loop_threshold"):
        logger.info("✅ Loop detection attributes are present")
    else:
        logger.error("❌ Loop detection attributes are missing")

if __name__ == "__main__":
    logger.info("=== LiteLLM Patch Verification ===")
    
    # Run tests
    test_robust_json_parsing()
    test_loop_detection()
    
    logger.info("=== Test Completed ===") 