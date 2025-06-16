"""
Tools for LinkedIn Post Reviewer Agent

This module provides tools for analyzing and validating LinkedIn posts.
"""

from typing import Any, Dict

from google.adk.tools.tool_context import ToolContext


def count_characters(text: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to count characters in the provided text and provide length-based feedback.
    Updates review_status in the state based on length requirements.

    Args:
        text: The text to analyze for character count
        tool_context: Context for accessing and updating session state

    Returns:
        Dict[str, Any]: Dictionary containing:
            - result: 'fail' or 'pass'
            - char_count: number of characters in text
            - message: feedback message about the length
    """
    char_count = len(text)
    MIN_LENGTH = 1000
    MAX_LENGTH = 1500

    print("\n----------- TOOL DEBUG -----------")
    print(f"Checking text length: {char_count} characters")
    print("----------------------------------\n")

    if char_count < MIN_LENGTH:
        chars_needed = MIN_LENGTH - char_count
        tool_context.state["review_status"] = "fail"
        return {
            "result": "fail",
            "char_count": char_count,
            "chars_needed": chars_needed,
            "message": f"Post is too short. Add {chars_needed} more characters to reach minimum length of {MIN_LENGTH}.",
        }
    elif char_count > MAX_LENGTH:
        chars_to_remove = char_count - MAX_LENGTH
        tool_context.state["review_status"] = "fail"
        return {
            "result": "fail",
            "char_count": char_count,
            "chars_to_remove": chars_to_remove,
            "message": f"Post is too long. Remove {chars_to_remove} characters to meet maximum length of {MAX_LENGTH}.",
        }
    else:
        tool_context.state["review_status"] = "pass"
        return {
            "result": "pass",
            "char_count": char_count,
            "message": f"Post length is good ({char_count} characters).",
        }


def exit_loop(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Call this function ONLY when the post meets all quality requirements,
    signaling the iterative process should end.

    Args:
        tool_context: Context for tool execution

    Returns:
        Empty dictionary
    """
    print("\n----------- EXIT LOOP TRIGGERED -----------")
    print("Post review completed successfully")
    print("Loop will exit now")
    print("------------------------------------------\n")

    tool_context.actions.escalate = True
    return {}
