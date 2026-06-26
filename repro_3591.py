# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reproduction script for issue #3591.

Demonstrates that `list[str] | None` (PEP 604 syntax) triggers a
"Failed to parse the parameter" ValueError in ADK's automatic function
calling parser, while `Optional[List[str]]` (typing module) succeeds.

This exercises `from_function_with_options` directly — the classic ADK
parameter-parsing path that is still the non-experimental default.

Run from the repo root (with ADK installed in dev mode):
  python repro_3591.py
"""

from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional

from google.adk.tools.tool_context import ToolContext


# ── BEFORE FIX: PEP 604 union syntax ──────────────────────────────────────────

async def cleanup_unused_files_broken(
    used_files: list[str],
    tool_context: ToolContext,
    file_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Original signature — triggers parser ValueError."""
    ...


# ── AFTER FIX: Optional[List[str]] (all sibling tools already use this) ──────

async def cleanup_unused_files_fixed(
    used_files: List[str],
    tool_context: ToolContext,
    file_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> dict[str, Any]:
    """Fixed signature — consistent with write_files, delete_files, read_files."""
    ...


def main():
    from google.adk.tools._automatic_function_calling_util import (
        from_function_with_options,
    )
    from google.adk.utils.variant_utils import GoogleLLMVariant

    variant = GoogleLLMVariant.GEMINI_API
    ignore = {"tool_context"}

    # Helper: strip ToolContext param, which the declaration builder also ignores
    import inspect
    from types import FunctionType

    def strip_tool_context(func):
        sig = inspect.signature(func)
        new_params = [p for n, p in sig.parameters.items() if n not in ignore]
        new_sig = sig.replace(parameters=new_params)
        wrapped = FunctionType(
            func.__code__, func.__globals__,
            func.__name__, func.__defaults__, func.__closure__
        )
        wrapped.__signature__ = new_sig
        wrapped.__doc__ = func.__doc__
        wrapped.__annotations__ = {
            k: v for k, v in func.__annotations__.items() if k not in ignore
        }
        return wrapped

    # ── BEFORE FIX ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("[BEFORE FIX]  list[str] | None  (PEP 604)")
    try:
        decl = from_function_with_options(
            strip_tool_context(cleanup_unused_files_broken), variant
        )
        print(f"  Unexpected success: {decl}")
    except ValueError as e:
        print(f"  ValueError (expected):\n    {e}\n")

    # ── AFTER FIX ─────────────────────────────────────────────────────────────
    print("[AFTER FIX]   Optional[List[str]]  (typing module)")
    try:
        decl = from_function_with_options(
            strip_tool_context(cleanup_unused_files_fixed), variant
        )
        props = decl.parameters.properties or {}
        fp = props.get("file_patterns")
        ep = props.get("exclude_patterns")
        print(f"  file_patterns    -> nullable={fp.nullable}, type={fp.type}")
        print(f"  exclude_patterns -> nullable={ep.nullable}, type={ep.type}")
        print("  OK — parsed without error\n")
    except Exception as e:
        import traceback
        print(f"  Unexpected error: {e}")
        traceback.print_exc()

    print("=" * 60)
    print("Conclusion: Replace `list[str] | None` with `Optional[List[str]]`")
    print("to fix issue #3591 without waiting for the parser to be updated.")


if __name__ == "__main__":
    main()
