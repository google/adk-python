#!/usr/bin/env python3
"""
Example demonstrating file loading with the completion function.
"""

from adk_rlm import completion

result = completion(
    files=["./plans/**/*"],
    prompt=(
        "What is this project about? Write a detailed summary of the project"
        " plan(s)."
    ),
)

print(result.response)
