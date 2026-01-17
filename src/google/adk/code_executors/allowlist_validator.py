# Copyright 2025 Google LLC
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

"""Import allowlist validation for code execution security."""

from __future__ import annotations

import ast
import fnmatch
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import FrozenSet
from typing import List
from typing import Set

logger = logging.getLogger("google_adk." + __name__)


# Default set of safe imports that are always allowed
DEFAULT_SAFE_IMPORTS: FrozenSet[str] = frozenset(
    {
        # Standard library - safe modules
        "json",
        "math",
        "re",
        "datetime",
        "collections",
        "collections.*",
        "itertools",
        "functools",
        "operator",
        "string",
        "textwrap",
        "unicodedata",
        "decimal",
        "fractions",
        "random",
        "statistics",
        "typing",
        "typing.*",
        "dataclasses",
        "enum",
        "abc",
        "copy",
        "pprint",
        "reprlib",
        "numbers",
        "cmath",
        "time",
        "calendar",
        "hashlib",
        "hmac",
        "base64",
        "binascii",
        "html",
        "html.*",
        "urllib.parse",
        "uuid",
        "struct",
        "codecs",
        "locale",
        "gettext",
        "bisect",
        "heapq",
        "array",
        "weakref",
        "types",
        "contextlib",
        "warnings",
        "traceback",
        "linecache",
        "difflib",
        "graphlib",
        "zoneinfo",
    }
)


class ImportValidationError(Exception):
    """Exception raised when import validation fails.

    Attributes:
      violations: List of import violations found.
      code: The code that was validated.
    """

    def __init__(
        self,
        violations: List[str],
        code: str,
    ):
        self.violations = violations
        self.code = code
        violation_str = "\n".join(f"  - {v}" for v in violations)
        super().__init__(
            f"Import validation failed. Unauthorized imports found:\n{violation_str}"
        )


@dataclass
class ImportInfo:
    """Information about an import statement.

    Attributes:
      module: The module being imported.
      names: Names being imported from the module (for 'from' imports).
      alias: Alias for the import (if any).
      line_number: Line number in the source code.
      is_from_import: Whether this is a 'from X import Y' style import.
    """

    module: str
    names: List[str] = field(default_factory=list)
    alias: str = ""
    line_number: int = 0
    is_from_import: bool = False


def extract_imports(code: str) -> List[ImportInfo]:
    """Extract all import statements from Python code using AST.

    Args:
      code: Python source code to analyze.

    Returns:
      List of ImportInfo objects describing each import.

    Raises:
      SyntaxError: If the code cannot be parsed.
    """
    imports = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.warning("Failed to parse code for import extraction: %s", e)
        raise

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=alias.name,
                        alias=alias.asname or "",
                        line_number=node.lineno,
                        is_from_import=False,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=module,
                        names=[alias.name],
                        alias=alias.asname or "",
                        line_number=node.lineno,
                        is_from_import=True,
                    )
                )

    return imports


def is_import_allowed(
    import_name: str,
    allowlist: FrozenSet[str],
) -> bool:
    """Check if an import is allowed by the allowlist.

    Supports wildcards:
      - 'collections.*' allows 'collections.abc', 'collections.defaultdict', etc.
      - 'numpy' allows only 'numpy', not 'numpy.array'
      - 'numpy.*' allows 'numpy.array', 'numpy.linalg', etc.

    Args:
      import_name: The full import name to check.
      allowlist: Set of allowed import patterns.

    Returns:
      True if the import is allowed, False otherwise.
    """
    # Direct match
    if import_name in allowlist:
        return True

    # Check wildcard patterns
    for pattern in allowlist:
        if "*" in pattern:
            if fnmatch.fnmatch(import_name, pattern):
                return True
            # Also check if the import is a submodule of an allowed module
            # e.g., 'collections.*' should allow 'collections.abc.Callable'
            base_pattern = pattern.rstrip(".*")
            if import_name.startswith(base_pattern + "."):
                return True

    # Check if parent module is allowed with wildcard
    parts = import_name.split(".")
    for i in range(len(parts)):
        parent = ".".join(parts[: i + 1])
        wildcard_pattern = parent + ".*"
        if wildcard_pattern in allowlist:
            return True

    return False


def validate_imports(
    code: str,
    allowlist: FrozenSet[str],
) -> List[str]:
    """Validate that all imports in code are in the allowlist.

    Args:
      code: Python source code to validate.
      allowlist: Set of allowed import patterns.

    Returns:
      List of violations (empty if all imports are valid).

    Raises:
      ImportValidationError: If unauthorized imports are found.
    """
    violations = []

    try:
        imports = extract_imports(code)
    except SyntaxError as e:
        # If we can't parse, we can't validate - return syntax error as violation
        violations.append(f"Syntax error in code: {e}")
        return violations

    for import_info in imports:
        module = import_info.module

        if import_info.is_from_import:
            # For 'from X import Y', check both the module and the full path
            for name in import_info.names:
                full_name = f"{module}.{name}" if module else name
                # Check if module is allowed OR full import path is allowed
                if not (
                    is_import_allowed(module, allowlist)
                    or is_import_allowed(full_name, allowlist)
                ):
                    violations.append(
                        f"Line {import_info.line_number}: "
                        f'Unauthorized import "from {module} import {name}"'
                    )
        else:
            # For 'import X', just check the module
            if not is_import_allowed(module, allowlist):
                violations.append(
                    f'Line {import_info.line_number}: Unauthorized import "{module}"'
                )

    return violations


def validate_imports_strict(
    code: str,
    allowlist: FrozenSet[str],
) -> None:
    """Validate imports and raise exception if any violations found.

    Args:
      code: Python source code to validate.
      allowlist: Set of allowed import patterns.

    Raises:
      ImportValidationError: If unauthorized imports are found.
    """
    violations = validate_imports(code, allowlist)
    if violations:
        raise ImportValidationError(violations, code)


class AllowlistValidator:
    """Validator for checking Python code imports against an allowlist.

    This class provides a stateful validator that can be reused for multiple
    validations with the same allowlist.

    Attributes:
      allowlist: The set of allowed import patterns.
    """

    def __init__(
        self,
        allowlist: FrozenSet[str] = DEFAULT_SAFE_IMPORTS,
        additional_imports: FrozenSet[str] = frozenset(),
    ):
        """Initialize the validator with an allowlist.

        Args:
          allowlist: Base set of allowed import patterns.
          additional_imports: Additional imports to allow beyond the base set.
        """
        self.allowlist = allowlist | additional_imports

    def validate(self, code: str) -> List[str]:
        """Validate imports in code.

        Args:
          code: Python source code to validate.

        Returns:
          List of violations (empty if all imports are valid).
        """
        return validate_imports(code, self.allowlist)

    def validate_strict(self, code: str) -> None:
        """Validate imports and raise if any violations found.

        Args:
          code: Python source code to validate.

        Raises:
          ImportValidationError: If unauthorized imports are found.
        """
        validate_imports_strict(code, self.allowlist)

    def is_allowed(self, import_name: str) -> bool:
        """Check if a single import is allowed.

        Args:
          import_name: The import name to check.

        Returns:
          True if allowed, False otherwise.
        """
        return is_import_allowed(import_name, self.allowlist)

    def add_allowed_imports(self, imports: Set[str]) -> None:
        """Add additional allowed imports.

        Args:
          imports: Set of import patterns to allow.
        """
        self.allowlist = self.allowlist | frozenset(imports)
