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

"""Tests for the release artifact import differential."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_SCRIPT = (
    pathlib.Path(__file__).parent.parent.parent
    / "scripts"
    / "verify_release_artifact.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "verify_release_artifact", _SCRIPT
)
verify = importlib.util.module_from_spec(_SPEC)
sys.modules["verify_release_artifact"] = verify
_SPEC.loader.exec_module(verify)


def _sweep(version: str, attempted, failures=None):
  return verify.Sweep(
      version=version,
      attempted=tuple(attempted),
      failures=dict(failures or {}),
  )


def test_module_names_skips_dist_info_and_dunder_main():
  names = verify.module_names_from_files([
      "google/adk/__init__.py",
      "google/adk/agents/llm_agent.py",
      "google/adk/__main__.py",
      "google_adk-2.6.1.dist-info/RECORD",
      "google_adk-2.6.1.dist-info/thing.py",
      "google/adk/py.typed",
  ])

  assert names == ["google.adk", "google.adk.agents.llm_agent"]


def test_module_names_includes_namespace_subpackages():
  # A subpackage with no __init__.py is exactly what a package-tree walk
  # silently skips, so it has to survive here.
  names = verify.module_names_from_files([
      "google/adk/integrations/thing/client.py",
  ])

  assert names == ["google.adk.integrations.thing.client"]


def test_module_names_rejects_paths_that_are_not_identifiers():
  assert not verify.module_names_from_files(["google/ad-k/mod.py"])


def test_module_names_deduplicates():
  names = verify.module_names_from_files(
      ["google/adk/__init__.py", "google/adk/__init__.py"]
  )

  assert names == ["google.adk"]


def test_compare_flags_a_module_that_stopped_importing():
  baseline = _sweep("2.6.0", ["a", "b"])
  candidate = _sweep("2.6.1", ["a", "b"], {"b": "ImportError: no name X"})

  result = verify.compare(baseline=baseline, candidate=candidate)

  assert result.regressed == ("b",)
  assert result.blocking == ("b",)
  assert not result.ok


def test_compare_reports_a_new_broken_module_without_failing():
  # A new module that does not import is almost always one sitting behind an
  # optional extra, so it is reported for a human but does not fail the gate.
  baseline = _sweep("2.6.0", ["a"])
  candidate = _sweep("2.6.1", ["a", "new"], {"new": "ImportError: boom"})

  result = verify.compare(baseline=baseline, candidate=candidate)

  assert result.newly_broken == ("new",)
  assert not result.blocking
  assert result.ok


def test_compare_still_fails_when_an_old_module_breaks_alongside_a_new_one():
  baseline = _sweep("2.6.0", ["a", "b"])
  candidate = _sweep(
      "2.6.1",
      ["a", "b", "new"],
      {"b": "ImportError: real", "new": "ImportError: needs an extra"},
  )

  result = verify.compare(baseline=baseline, candidate=candidate)

  assert result.blocking == ("b",)
  assert not result.ok


def test_compare_ignores_failures_that_were_already_there():
  # The signal is the delta. A healthy release carries a large stable set of
  # modules whose optional dependency is simply absent.
  baseline = _sweep("2.6.0", ["a", "b"], {"b": "ModuleNotFoundError: extra"})
  candidate = _sweep("2.6.1", ["a", "b"], {"b": "ModuleNotFoundError: extra"})

  result = verify.compare(baseline=baseline, candidate=candidate)

  assert result.ok
  assert not result.blocking


def test_compare_reports_repaired_and_dropped_without_failing():
  baseline = _sweep("2.6.0", ["a", "b", "gone"], {"b": "ImportError: x"})
  candidate = _sweep("2.6.1", ["a", "b"])

  result = verify.compare(baseline=baseline, candidate=candidate)

  assert result.repaired == ("b",)
  assert result.dropped == ("gone",)
  assert result.ok


def test_compare_honours_the_allowlist():
  baseline = _sweep("2.6.0", ["a", "b"])
  candidate = _sweep("2.6.1", ["a", "b"], {"b": "ImportError: on purpose"})

  result = verify.compare(
      baseline=baseline, candidate=candidate, allowlist={"b"}
  )

  assert result.ok
  assert result.suppressed == ("b",)
  assert not result.regressed


def test_load_allowlist_strips_comments_and_blanks():
  entries = verify.load_allowlist(
      "# a comment\n\ngoogle.adk.one  # why\n  google.adk.two\n"
  )

  assert entries == {"google.adk.one", "google.adk.two"}


def test_report_names_the_failing_modules_and_their_errors():
  baseline = _sweep("2.6.0", ["a", "b"])
  candidate = _sweep("2.6.1", ["a", "b"], {"b": "ImportError: cannot find X"})
  comparison = verify.compare(baseline=baseline, candidate=candidate)

  report = verify.render_report(
      baseline=baseline, candidate=candidate, comparison=comparison
  )

  assert "FAIL" in report
  assert "`b`" in report
  assert "ImportError: cannot find X" in report


def test_report_separates_new_broken_modules_from_regressions():
  baseline = _sweep("2.6.0", ["a"])
  candidate = _sweep("2.6.1", ["a", "new"], {"new": "ImportError: needs extra"})
  comparison = verify.compare(baseline=baseline, candidate=candidate)

  report = verify.render_report(
      baseline=baseline, candidate=candidate, comparison=comparison
  )

  assert "PASS" in report
  assert "New modules that do not import (1)" in report
  assert "Import regressions" not in report


def test_report_states_the_versions_it_compared():
  baseline = _sweep("2.6.0", ["a"])
  candidate = _sweep("2.6.1", ["a"])
  comparison = verify.compare(baseline=baseline, candidate=candidate)

  report = verify.render_report(
      baseline=baseline, candidate=candidate, comparison=comparison
  )

  assert "PASS" in report
  assert "`2.6.1`" in report and "`2.6.0`" in report


def test_baseline_target_auto_picks_the_release_below_the_candidate():
  # Not simply the newest release: a 1.x candidate must not be compared
  # against the newest 2.x while both lines are maintained.
  assert (
      verify.baseline_target("auto", candidate_version="1.36.0")
      == "google-adk>=1.0.0,<1.36.0"
  )


def test_baseline_target_auto_stays_inside_the_major_line():
  # Across a major boundary the comparison is restructuring noise, not signal.
  assert (
      verify.baseline_target("auto", candidate_version="2.6.1")
      == "google-adk>=2.0.0,<2.6.1"
  )


def test_baseline_target_accepts_an_explicit_version_or_path():
  assert (
      verify.baseline_target("2.6.0", candidate_version="2.6.1")
      == "google-adk==2.6.0"
  )
  assert (
      verify.baseline_target("dist/x.whl", candidate_version="2.6.1")
      == "dist/x.whl"
  )


def test_environment_commands_prefers_uv():
  commands = verify.environment_commands(
      venv_dir=pathlib.Path("/tmp/v"), target="x.whl", uv_available=True
  )

  assert commands[0][:2] == ["uv", "venv"]
  assert commands[1][-1] == "x.whl"


def test_environment_commands_falls_back_to_stdlib_venv():
  commands = verify.environment_commands(
      venv_dir=pathlib.Path("/tmp/v"), target="x.whl", uv_available=False
  )

  assert commands[0][1:3] == ["-m", "venv"]
  assert commands[1][1:] == ["install", "x.whl"]


def test_resolve_wheel_rejects_an_ambiguous_glob(tmp_path):
  (tmp_path / "one-1.0-py3-none-any.whl").write_text("")
  (tmp_path / "two-2.0-py3-none-any.whl").write_text("")

  with pytest.raises(verify.HarnessError, match="more than one wheel"):
    verify.resolve_wheel(str(tmp_path / "*.whl"))


def test_resolve_wheel_rejects_a_glob_matching_nothing(tmp_path):
  with pytest.raises(verify.HarnessError, match="no wheel matched"):
    verify.resolve_wheel(str(tmp_path / "*.whl"))


def test_main_exits_two_when_the_check_cannot_run(tmp_path, capsys):
  # Fail closed: a harness failure must never be reported as a pass.
  code = verify.main(["--wheel", str(tmp_path / "*.whl")])

  assert code == verify.EXIT_HARNESS_FAILURE
  assert "could not run" in capsys.readouterr().err


def test_check_rejects_a_baseline_equal_to_the_candidate(monkeypatch, tmp_path):
  wheel = tmp_path / "google_adk-2.6.1-py3-none-any.whl"
  wheel.write_text("")
  monkeypatch.setattr(
      verify,
      "sweep_target",
      lambda target, *, label, uv_available: _sweep("2.6.1", ["a"]),
  )

  args = verify.parse_args(["--wheel", str(wheel)])
  with pytest.raises(verify.HarnessError, match="nothing to compare"):
    verify.run_check(args)


def test_check_rejects_an_unexpected_version(monkeypatch, tmp_path):
  wheel = tmp_path / "google_adk-2.6.1-py3-none-any.whl"
  wheel.write_text("")
  versions = iter(["2.6.1", "2.6.0"])
  monkeypatch.setattr(
      verify,
      "sweep_target",
      lambda target, *, label, uv_available: _sweep(next(versions), ["a"]),
  )

  args = verify.parse_args(
      ["--wheel", str(wheel), "--expected-version", "2.7.0"]
  )
  with pytest.raises(verify.HarnessError, match="expected 2.7.0"):
    verify.run_check(args)


def test_check_rejects_an_empty_sweep(monkeypatch, tmp_path):
  wheel = tmp_path / "google_adk-2.6.1-py3-none-any.whl"
  wheel.write_text("")
  versions = iter(["2.6.1", "2.6.0"])
  monkeypatch.setattr(
      verify,
      "sweep_target",
      lambda target, *, label, uv_available: _sweep(next(versions), []),
  )

  args = verify.parse_args(["--wheel", str(wheel)])
  with pytest.raises(verify.HarnessError, match="no modules"):
    verify.run_check(args)


def test_sweep_installed_records_the_error_and_keeps_going(monkeypatch):
  monkeypatch.setattr(
      verify,
      "module_names_from_files",
      lambda paths: ["good", "bad", "also_good"],
  )

  class _Dist:
    version = "9.9.9"
    files = ["ignored.py"]

  monkeypatch.setattr(
      verify.importlib.metadata, "distribution", lambda name: _Dist()
  )

  def fake_import(name):
    if name == "bad":
      raise ImportError("cannot import name X")
    return object()

  monkeypatch.setattr(verify.importlib, "import_module", fake_import)

  sweep = verify.sweep_installed("google-adk")

  assert sweep.version == "9.9.9"
  assert sweep.attempted == ("good", "bad", "also_good")
  assert sweep.failures == {"bad": "ImportError: cannot import name X"}


def test_sweep_installed_survives_a_module_that_exits(monkeypatch):
  monkeypatch.setattr(
      verify, "module_names_from_files", lambda paths: ["quitter", "after"]
  )

  class _Dist:
    version = "9.9.9"
    files = ["ignored.py"]

  monkeypatch.setattr(
      verify.importlib.metadata, "distribution", lambda name: _Dist()
  )

  def fake_import(name):
    if name == "quitter":
      raise SystemExit(3)
    return object()

  monkeypatch.setattr(verify.importlib, "import_module", fake_import)

  sweep = verify.sweep_installed("google-adk")

  assert "quitter" in sweep.failures
  assert "after" not in sweep.failures


def test_check_explains_a_missing_same_major_baseline(monkeypatch, tmp_path):
  wheel = tmp_path / "google_adk-3.0.0-py3-none-any.whl"
  wheel.write_text("")

  def fake_sweep(target, *, label, uv_available):
    del target, uv_available
    if label == "baseline":
      raise verify.HarnessError("uv: no matching version")
    return _sweep("3.0.0", ["a"])

  monkeypatch.setattr(verify, "sweep_target", fake_sweep)

  args = verify.parse_args(["--wheel", str(wheel)])
  with pytest.raises(verify.HarnessError, match="same major"):
    verify.run_check(args)
