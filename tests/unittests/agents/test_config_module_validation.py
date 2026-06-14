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

"""Tests for module path validation in YAML agent config resolution."""

import pytest

from google.adk.agents.config_agent_utils import _validate_module_path


class TestValidateModulePath:
    """Tests for _validate_module_path blocklist enforcement."""

    def test_safe_module_passes(self):
        """User-defined modules should pass validation."""
        _validate_module_path("my_app.agents.my_agent")
        _validate_module_path("google.adk.agents")
        _validate_module_path("my_package")

    def test_os_module_blocked(self):
        """os module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("os")

    def test_os_path_blocked(self):
        """os.path should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("os.path")

    def test_subprocess_blocked(self):
        """subprocess module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("subprocess")

    def test_sys_blocked(self):
        """sys module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("sys")

    def test_shutil_blocked(self):
        """shutil module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("shutil")

    def test_pickle_blocked(self):
        """pickle module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("pickle")

    def test_importlib_blocked(self):
        """importlib module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("importlib")

    def test_builtins_blocked(self):
        """builtins module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("builtins")

    def test_socket_blocked(self):
        """socket module should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("socket")

    def test_empty_path_blocked(self):
        """Empty module path should be rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_module_path("")

    def test_invalid_characters_blocked(self):
        """Module paths with special characters should be rejected."""
        with pytest.raises(ValueError, match="invalid characters"):
            _validate_module_path("os;import sys")

    def test_dunder_segment_blocked(self):
        """Module paths with __dunder__ segments should be rejected."""
        with pytest.raises(ValueError, match="dunder segment"):
            _validate_module_path("my_app.__builtins__.evil")

    def test_google_adk_passes(self):
        """google.adk modules should pass (not blocked)."""
        _validate_module_path("google.adk.tools.my_tool")
        _validate_module_path("google.adk.agents.llm_agent")

    def test_multiprocessing_blocked(self):
        """multiprocessing should be blocked."""
        with pytest.raises(ValueError, match="blocked module"):
            _validate_module_path("multiprocessing.pool")
