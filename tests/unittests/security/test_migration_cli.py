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

import unittest
from unittest import mock

from click.testing import CliRunner
from google.adk.cli.cli_tools_click import main


class TestMigrationCLISecurity(unittest.TestCase):

  def setUp(self):
    self.runner = CliRunner()

  def test_migrate_session_blocks_remote_ip(self):
    """Verifies that the CLI blocks migration from a remote IP source."""
    # Using an external IP address
    untrusted_url = "sqlite://1.2.3.4/malicious.db"

    result = self.runner.invoke(
        main,
        [
            "migrate",
            "session",
            "--source_db_url",
            untrusted_url,
            "--dest_db_url",
            "sqlite:///local.db",
        ],
    )

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Untrusted source database URL detected", result.output)
    self.assertIn("--force-untrusted-source", result.output)

  def test_migrate_session_blocks_unc_path(self):
    """Verifies that the CLI blocks migration from a Windows UNC path."""
    # Using a Windows UNC path (Samba style)
    untrusted_url = "sqlite:///\\\\192.168.1.90\\lab_share\\malicious.db"

    result = self.runner.invoke(
        main,
        [
            "migrate",
            "session",
            "--source_db_url",
            untrusted_url,
            "--dest_db_url",
            "sqlite:///local.db",
        ],
    )

    self.assertNotEqual(result.exit_code, 0)
    self.assertIn("Untrusted source database URL detected", result.output)

  @mock.patch("google.adk.sessions.migration.migration_runner.upgrade")
  def test_migrate_session_allows_localhost(self, mock_upgrade):
    """Verifies that localhost URLs are trusted by default."""
    trusted_url = "sqlite:///local.db"

    result = self.runner.invoke(
        main,
        [
            "migrate",
            "session",
            "--source_db_url",
            trusted_url,
            "--dest_db_url",
            "sqlite:///dest.db",
        ],
    )

    # It should call upgrade (we mock it call because we don't want to run real migration)
    mock_upgrade.assert_called_once()
    self.assertEqual(result.exit_code, 0)

  @mock.patch("google.adk.sessions.migration.migration_runner.upgrade")
  def test_migrate_session_force_flag_works(self, mock_upgrade):
    """Verifies that the --force-untrusted-source flag bypasses the block."""
    untrusted_url = "sqlite://8.8.8.8/remote.db"

    result = self.runner.invoke(
        main,
        [
            "migrate",
            "session",
            "--source_db_url",
            untrusted_url,
            "--dest_db_url",
            "sqlite:///local.db",
            "--force-untrusted-source",
        ],
    )

    # Should call upgrade with force_untrusted_source=True
    mock_upgrade.assert_called_with(untrusted_url, "sqlite:///local.db", True)
    self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
  unittest.main()
