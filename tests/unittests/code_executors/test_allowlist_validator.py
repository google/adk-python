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

"""Tests for AllowlistValidator."""

from __future__ import annotations

from google.adk.code_executors.allowlist_validator import AllowlistValidator
from google.adk.code_executors.allowlist_validator import DEFAULT_SAFE_IMPORTS
from google.adk.code_executors.allowlist_validator import extract_imports
from google.adk.code_executors.allowlist_validator import ImportValidationError
from google.adk.code_executors.allowlist_validator import is_import_allowed
from google.adk.code_executors.allowlist_validator import validate_imports
import pytest


class TestExtractImports:
  """Tests for extract_imports function."""

  def test_simple_import(self):
    """Test extracting simple import statements."""
    code = "import json"
    imports = extract_imports(code)

    assert len(imports) == 1
    assert imports[0].module == "json"
    assert imports[0].is_from_import is False

  def test_multiple_imports(self):
    """Test extracting multiple imports."""
    code = """
import json
import math
import re
"""
    imports = extract_imports(code)

    assert len(imports) == 3
    modules = {i.module for i in imports}
    assert modules == {"json", "math", "re"}

  def test_from_import(self):
    """Test extracting from imports."""
    code = "from collections import defaultdict"
    imports = extract_imports(code)

    assert len(imports) == 1
    assert imports[0].module == "collections"
    assert imports[0].names == ["defaultdict"]
    assert imports[0].is_from_import is True

  def test_from_import_multiple(self):
    """Test extracting from imports with multiple names."""
    code = "from typing import List, Dict, Optional"
    imports = extract_imports(code)

    assert len(imports) == 3
    for imp in imports:
      assert imp.module == "typing"
      assert imp.is_from_import is True

  def test_import_with_alias(self):
    """Test extracting imports with aliases."""
    code = "import numpy as np"
    imports = extract_imports(code)

    assert len(imports) == 1
    assert imports[0].module == "numpy"
    assert imports[0].alias == "np"

  def test_submodule_import(self):
    """Test extracting submodule imports."""
    code = "import os.path"
    imports = extract_imports(code)

    assert len(imports) == 1
    assert imports[0].module == "os.path"

  def test_from_submodule_import(self):
    """Test extracting from submodule imports."""
    code = "from collections.abc import Mapping"
    imports = extract_imports(code)

    assert len(imports) == 1
    assert imports[0].module == "collections.abc"
    assert imports[0].names == ["Mapping"]

  def test_syntax_error(self):
    """Test handling of syntax errors."""
    code = "import json\nthis is not valid python"

    with pytest.raises(SyntaxError):
      extract_imports(code)

  def test_no_imports(self):
    """Test code with no imports."""
    code = "x = 1 + 2\nprint(x)"
    imports = extract_imports(code)

    assert len(imports) == 0


class TestIsImportAllowed:
  """Tests for is_import_allowed function."""

  def test_direct_match(self):
    """Test direct import match."""
    allowlist = frozenset({"json", "math"})

    assert is_import_allowed("json", allowlist) is True
    assert is_import_allowed("math", allowlist) is True
    assert is_import_allowed("os", allowlist) is False

  def test_wildcard_match(self):
    """Test wildcard pattern matching."""
    allowlist = frozenset({"collections.*"})

    assert is_import_allowed("collections.abc", allowlist) is True
    assert is_import_allowed("collections.defaultdict", allowlist) is True
    assert is_import_allowed("itertools", allowlist) is False

  def test_deep_wildcard_match(self):
    """Test wildcard matching for deep submodules."""
    allowlist = frozenset({"collections.*"})

    assert is_import_allowed("collections.abc.Mapping", allowlist) is True

  def test_exact_vs_wildcard(self):
    """Test that exact matches work without wildcard."""
    allowlist = frozenset({"numpy"})

    assert is_import_allowed("numpy", allowlist) is True
    # Without wildcard, submodules are not allowed
    assert is_import_allowed("numpy.array", allowlist) is False

  def test_multiple_patterns(self):
    """Test multiple patterns in allowlist."""
    allowlist = frozenset({"json", "typing.*", "collections"})

    assert is_import_allowed("json", allowlist) is True
    assert is_import_allowed("typing.List", allowlist) is True
    assert is_import_allowed("collections", allowlist) is True
    assert is_import_allowed("collections.abc", allowlist) is False


class TestValidateImports:
  """Tests for validate_imports function."""

  def test_all_allowed(self):
    """Test code with all imports allowed."""
    code = """
import json
import math
from typing import List
"""
    allowlist = frozenset({"json", "math", "typing.*"})

    violations = validate_imports(code, allowlist)
    assert len(violations) == 0

  def test_some_violations(self):
    """Test code with some unauthorized imports."""
    code = """
import json
import os
import subprocess
"""
    allowlist = frozenset({"json"})

    violations = validate_imports(code, allowlist)
    assert len(violations) == 2
    assert any("os" in v for v in violations)
    assert any("subprocess" in v for v in violations)

  def test_from_import_violations(self):
    """Test from import violations."""
    code = "from os import system"
    allowlist = frozenset({"json"})

    violations = validate_imports(code, allowlist)
    assert len(violations) == 1
    assert "os" in violations[0]

  def test_syntax_error_violation(self):
    """Test that syntax errors are reported as violations."""
    code = "import json\n$$$invalid"
    allowlist = frozenset({"json"})

    violations = validate_imports(code, allowlist)
    assert len(violations) == 1
    assert "Syntax error" in violations[0]


class TestImportValidationError:
  """Tests for ImportValidationError exception."""

  def test_error_message(self):
    """Test error message formatting."""
    violations = ["Unauthorized import: os", "Unauthorized import: subprocess"]
    code = "import os\nimport subprocess"

    error = ImportValidationError(violations, code)

    assert "Import validation failed" in str(error)
    assert "os" in str(error)
    assert "subprocess" in str(error)

  def test_error_attributes(self):
    """Test error attributes."""
    violations = ["violation1", "violation2"]
    code = "some code"

    error = ImportValidationError(violations, code)

    assert error.violations == violations
    assert error.code == code


class TestAllowlistValidator:
  """Tests for AllowlistValidator class."""

  def test_default_allowlist(self):
    """Test validator with default allowlist."""
    validator = AllowlistValidator()

    # These should be in the default safe imports
    assert validator.is_allowed("json") is True
    assert validator.is_allowed("math") is True
    assert validator.is_allowed("typing") is True

    # These should not be in the default safe imports
    assert validator.is_allowed("os") is False
    assert validator.is_allowed("subprocess") is False

  def test_custom_allowlist(self):
    """Test validator with custom allowlist."""
    custom = frozenset({"custom_module"})
    validator = AllowlistValidator(allowlist=custom)

    assert validator.is_allowed("custom_module") is True
    assert validator.is_allowed("json") is False

  def test_additional_imports(self):
    """Test adding additional imports to default."""
    additional = frozenset({"custom_module", "another_module"})
    validator = AllowlistValidator(additional_imports=additional)

    # Should have both default and additional
    assert validator.is_allowed("json") is True
    assert validator.is_allowed("custom_module") is True
    assert validator.is_allowed("another_module") is True

  def test_validate_method(self):
    """Test validate method returns violations."""
    validator = AllowlistValidator(allowlist=frozenset({"json"}))

    violations = validator.validate("import json\nimport os")
    assert len(violations) == 1
    assert "os" in violations[0]

  def test_validate_strict_raises(self):
    """Test validate_strict raises on violations."""
    validator = AllowlistValidator(allowlist=frozenset({"json"}))

    with pytest.raises(ImportValidationError):
      validator.validate_strict("import os")

  def test_validate_strict_passes(self):
    """Test validate_strict passes when no violations."""
    validator = AllowlistValidator(allowlist=frozenset({"json"}))

    # Should not raise
    validator.validate_strict("import json")

  def test_add_allowed_imports(self):
    """Test adding imports after construction."""
    validator = AllowlistValidator(allowlist=frozenset({"json"}))

    assert validator.is_allowed("os") is False

    validator.add_allowed_imports({"os"})

    assert validator.is_allowed("os") is True


class TestDefaultSafeImports:
  """Tests for the default safe imports list."""

  def test_common_safe_imports_included(self):
    """Test that common safe imports are in the default list."""
    assert "json" in DEFAULT_SAFE_IMPORTS
    assert "math" in DEFAULT_SAFE_IMPORTS
    assert "re" in DEFAULT_SAFE_IMPORTS
    assert "datetime" in DEFAULT_SAFE_IMPORTS
    assert "typing" in DEFAULT_SAFE_IMPORTS
    assert "collections" in DEFAULT_SAFE_IMPORTS

  def test_dangerous_imports_not_included(self):
    """Test that dangerous imports are not in the default list."""
    assert "os" not in DEFAULT_SAFE_IMPORTS
    assert "subprocess" not in DEFAULT_SAFE_IMPORTS
    assert "sys" not in DEFAULT_SAFE_IMPORTS
    assert "socket" not in DEFAULT_SAFE_IMPORTS
    assert "ctypes" not in DEFAULT_SAFE_IMPORTS

  def test_wildcard_patterns_included(self):
    """Test that wildcard patterns are included."""
    assert "collections.*" in DEFAULT_SAFE_IMPORTS
    assert "typing.*" in DEFAULT_SAFE_IMPORTS
