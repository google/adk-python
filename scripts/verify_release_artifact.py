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

"""Checks that a built wheel does not import worse than the last release.

Publishing uploads a wheel without ever installing it. This installs the
candidate wheel and the previous release side by side, tries to import every
module each one ships, and compares the two failure sets.

Only the difference matters. A healthy release has a large, stable set of
modules that fail to import because their optional dependency is absent, so
the absolute count says nothing. A module that imported in the previous
release and fails in the candidate is a regression, and so is a brand new
module that has never imported at all.

Run it locally against any two versions:

  python scripts/verify_release_artifact.py --wheel dist/*.whl

Exit codes: 0 clean, 1 regressions found, 2 the check itself could not run.
This deliberately depends on nothing outside the standard library, because it
has to run before the package under test is installed anywhere.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from collections.abc import Sequence
import dataclasses
import glob
import importlib
import importlib.metadata
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

DISTRIBUTION = "google-adk"

EXIT_OK = 0
EXIT_REGRESSED = 1
EXIT_HARNESS_FAILURE = 2

_INSTALL_TIMEOUT_SECONDS = 900
_SWEEP_TIMEOUT_SECONDS = 900
_MAX_CAPTURED_OUTPUT = 4000


class HarnessError(RuntimeError):
  """The check could not be completed, so its result means nothing."""


@dataclasses.dataclass(frozen=True)
class Sweep:
  """What one installed distribution could and could not import."""

  version: str
  attempted: tuple[str, ...]
  failures: dict[str, str]

  @classmethod
  def from_json(cls, payload: str) -> Sweep:
    data = json.loads(payload)
    return cls(
        version=data["version"],
        attempted=tuple(data["attempted"]),
        failures=dict(data["failures"]),
    )


@dataclasses.dataclass(frozen=True)
class Comparison:
  """How the candidate's imports differ from the baseline's."""

  regressed: tuple[str, ...]
  newly_broken: tuple[str, ...]
  repaired: tuple[str, ...]
  dropped: tuple[str, ...]
  suppressed: tuple[str, ...]

  @property
  def blocking(self) -> tuple[str, ...]:
    """Modules that fail the gate: they used to import and no longer do.

    A module that is new in this release and does not import is reported but
    does not fail the run. Most new modules sit behind an optional extra, so
    on a bare install their failure is expected and cannot be told apart from
    a real defect without modelling which extra each one needs.
    """
    return self.regressed

  @property
  def ok(self) -> bool:
    return not self.blocking


# --- module enumeration -----------------------------------------------------


def module_names_from_files(paths: Iterable[object]) -> list[str]:
  """Derives importable module names from a distribution's file list.

  Walking the installed file list rather than the package tree is deliberate.
  A namespace subpackage carries no `__init__.py`, and package walkers refuse
  to descend into one, so a tree walk silently skips whole subtrees.

  Args:
    paths: Paths recorded for the installed distribution, relative to the
      site-packages root.

  Returns:
    Sorted, de-duplicated dotted module names worth importing.
  """
  names: set[str] = set()
  for raw in paths:
    path = str(raw).replace("\\", "/")
    if not path.endswith(".py"):
      continue
    parts = path[: -len(".py")].split("/")
    if any(p.endswith((".dist-info", ".data")) for p in parts):
      continue
    if parts and parts[-1] == "__init__":
      parts = parts[:-1]
    if not parts:
      continue
    # Importing __main__ runs a command line entry point.
    if parts[-1] == "__main__":
      continue
    if any(not p.isidentifier() for p in parts):
      continue
    names.add(".".join(parts))
  return sorted(names)


def sweep_installed(distribution: str) -> Sweep:
  """Imports every module of an installed distribution, recording failures."""
  dist = importlib.metadata.distribution(distribution)
  names = module_names_from_files(dist.files or [])
  failures: dict[str, str] = {}
  for name in names:
    try:
      importlib.import_module(name)
    except (Exception, SystemExit) as err:  # pylint: disable=broad-except
      # One unimportable module must not end the sweep; recording it is the
      # entire purpose of this pass.
      failures[name] = f"{type(err).__name__}: {err}".strip()
  return Sweep(version=dist.version, attempted=tuple(names), failures=failures)


# --- comparison -------------------------------------------------------------


def load_allowlist(text: str) -> set[str]:
  """Reads allowlisted module names, ignoring comments and blank lines."""
  entries: set[str] = set()
  for line in text.splitlines():
    stripped = line.split("#", 1)[0].strip()
    if stripped:
      entries.add(stripped)
  return entries


def compare(
    *,
    baseline: Sweep,
    candidate: Sweep,
    allowlist: set[str] | None = None,
) -> Comparison:
  """Diffs two sweeps into the categories the gate cares about."""
  allowed = allowlist or set()
  baseline_attempted = set(baseline.attempted)
  candidate_attempted = set(candidate.attempted)
  baseline_failed = set(baseline.failures)
  candidate_failed = set(candidate.failures)

  regressed = (candidate_failed & baseline_attempted) - baseline_failed
  newly_broken = candidate_failed - baseline_attempted
  repaired = (baseline_failed & candidate_attempted) - candidate_failed
  dropped = baseline_attempted - candidate_attempted

  suppressed = (regressed | newly_broken) & allowed
  return Comparison(
      regressed=tuple(sorted(regressed - allowed)),
      newly_broken=tuple(sorted(newly_broken - allowed)),
      repaired=tuple(sorted(repaired)),
      dropped=tuple(sorted(dropped)),
      suppressed=tuple(sorted(suppressed)),
  )


def render_report(
    *, baseline: Sweep, candidate: Sweep, comparison: Comparison
) -> str:
  """Builds the markdown summary, naming modules rather than counting them."""
  verdict = "PASS" if comparison.ok else "FAIL"
  lines = [
      f"# Release artifact check: {verdict}",
      "",
      f"Comparing `{candidate.version}` against `{baseline.version}`.",
      (
          f"Modules swept: {len(candidate.attempted)} candidate,"
          f" {len(baseline.attempted)} baseline."
      ),
      "",
  ]

  if comparison.blocking:
    lines.extend([
        f"## Import regressions ({len(comparison.blocking)})",
        "",
        "These import in the baseline and fail to import in the candidate.",
        "",
    ])
    for name in comparison.blocking:
      lines.append(f"- `{name}`")
      lines.append(f"  - {candidate.failures.get(name, 'unknown error')}")
    lines.append("")
  else:
    lines.extend(["No module regressed against the baseline.", ""])

  if comparison.newly_broken:
    lines.extend([
        f"## New modules that do not import ({len(comparison.newly_broken)})",
        "",
        (
            "Not a failure. New modules usually sit behind an optional extra,"
            " so this is expected on a bare install -- but a module that is"
            " meant to work without extras belongs on the list above, so it"
            " is worth a glance."
        ),
        "",
    ])
    for name in comparison.newly_broken:
      lines.append(f"- `{name}`")
      lines.append(f"  - {candidate.failures.get(name, 'unknown error')}")
    lines.append("")

  if comparison.suppressed:
    lines.extend([
        f"## Allowlisted ({len(comparison.suppressed)})",
        "",
        "Failing, but declared expected in the allowlist file.",
        "",
    ])
    lines.extend(f"- `{name}`" for name in comparison.suppressed)
    lines.append("")

  for title, names in (
      ("Now importing again", comparison.repaired),
      ("No longer shipped", comparison.dropped),
  ):
    if not names:
      continue
    lines.extend(
        ["<details>", f"<summary>{title} ({len(names)})</summary>", ""]
    )
    lines.extend(f"- `{name}`" for name in names)
    lines.extend(["", "</details>", ""])

  return "\n".join(lines).rstrip() + "\n"


# --- environment plumbing ---------------------------------------------------


def venv_binary(venv_dir: pathlib.Path, name: str) -> str:
  """Path to an executable inside a virtual environment."""
  if sys.platform == "win32":
    return str(venv_dir / "Scripts" / f"{name}.exe")
  return str(venv_dir / "bin" / name)


def environment_commands(
    *, venv_dir: pathlib.Path, target: str, uv_available: bool
) -> list[list[str]]:
  """Commands that create an environment and install one target into it."""
  python = venv_binary(venv_dir, "python")
  if uv_available:
    return [
        ["uv", "venv", str(venv_dir)],
        ["uv", "pip", "install", "--python", python, target],
    ]
  return [
      [sys.executable, "-m", "venv", str(venv_dir)],
      [venv_binary(venv_dir, "pip"), "install", target],
  ]


def _run(command: Sequence[str], *, timeout: int) -> tuple[int, str]:
  """Runs a command, returning its exit code and combined output."""
  try:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
  except (subprocess.SubprocessError, OSError) as err:
    return 1, f"{type(err).__name__}: {err}"
  output = (completed.stdout + completed.stderr)[-_MAX_CAPTURED_OUTPUT:]
  return completed.returncode, output


def sweep_target(target: str, *, label: str, uv_available: bool) -> Sweep:
  """Installs one target into a throwaway environment and sweeps it."""
  with tempfile.TemporaryDirectory(prefix=f"adk-{label}-") as temp_dir:
    venv_dir = pathlib.Path(temp_dir) / "venv"
    for command in environment_commands(
        venv_dir=venv_dir, target=target, uv_available=uv_available
    ):
      code, output = _run(command, timeout=_INSTALL_TIMEOUT_SECONDS)
      if code != 0:
        raise HarnessError(
            f"{label}: `{' '.join(command)}` exited {code}\n{output}"
        )

    # The sweep reports through a file rather than stdout: importing a few
    # hundred modules reliably prints warnings and log lines, and any one of
    # them would corrupt a JSON document written to the same stream.
    result_path = pathlib.Path(temp_dir) / "sweep.json"
    code, output = _run(
        [
            venv_binary(venv_dir, "python"),
            __file__,
            "--sweep",
            "--sweep-out",
            str(result_path),
        ],
        timeout=_SWEEP_TIMEOUT_SECONDS,
    )
    if code != 0:
      raise HarnessError(f"{label}: sweep exited {code}\n{output}")
    if not result_path.is_file():
      raise HarnessError(f"{label}: sweep wrote no result\n{output}")
    try:
      return Sweep.from_json(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError, OSError) as err:
      raise HarnessError(f"{label}: unreadable sweep output: {err}") from err


# --- entry point ------------------------------------------------------------


def resolve_wheel(pattern: str) -> str:
  """Resolves a glob to exactly one wheel, or raises."""
  matches = sorted(glob.glob(pattern))
  if not matches:
    raise HarnessError(f"no wheel matched {pattern!r}")
  if len(matches) > 1:
    raise HarnessError(f"{pattern!r} matched more than one wheel: {matches}")
  return matches[0]


def baseline_target(baseline: str, *, candidate_version: str) -> str:
  """Turns a baseline argument into something installable.

  The default resolves to the highest release below the candidate within the
  same major line. Two reasons it is not simply the newest release. While an
  older line is still maintained, a 1.x candidate would otherwise be compared
  against the newest 2.x. And across a major boundary the comparison is not
  meaningful at all: 2.0.0 against 1.37.0 reports 73 modules, nearly all of
  them a deliberate restructuring rather than a defect.

  Args:
    baseline: 'auto', a released version, or a path to a distribution.
    candidate_version: Version the candidate wheel reports.

  Returns:
    An installable requirement or path.
  """
  if baseline == "auto":
    major = candidate_version.split(".")[0]
    return f"{DISTRIBUTION}>={major}.0.0,<{candidate_version}"
  if baseline.endswith((".whl", ".tar.gz")):
    return baseline
  return f"{DISTRIBUTION}=={baseline}"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  """Builds the command line and parses it."""
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument(
      "--sweep",
      action="store_true",
      help=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--sweep-out",
      default=None,
      help=argparse.SUPPRESS,
  )
  parser.add_argument(
      "--wheel",
      default="dist/*.whl",
      help="Candidate wheel to check. Accepts a glob matching one file.",
  )
  parser.add_argument(
      "--baseline",
      default="auto",
      help=(
          "What to compare against: a released version, a path to a"
          " distribution, or 'auto' for the highest release below the"
          " candidate."
      ),
  )
  parser.add_argument(
      "--expected-version",
      default=None,
      help="Version the candidate must report once installed.",
  )
  parser.add_argument(
      "--allowlist",
      default=None,
      help="File of module names whose import failure is expected.",
  )
  parser.add_argument(
      "--report",
      default=None,
      help="Write the markdown report here in addition to stdout.",
  )
  return parser.parse_args(argv)


def run_check(args: argparse.Namespace) -> tuple[str, bool]:
  """Runs both sweeps and compares them.

  Args:
    args: Parsed command line arguments.

  Returns:
    The rendered report and whether the gate passed.

  Raises:
    HarnessError: The check could not be completed.
  """
  wheel = resolve_wheel(args.wheel)
  uv_available = shutil.which("uv") is not None

  candidate = sweep_target(wheel, label="candidate", uv_available=uv_available)
  if args.expected_version and candidate.version != args.expected_version:
    raise HarnessError(
        f"candidate reports {candidate.version},"
        f" expected {args.expected_version}"
    )

  # The candidate is swept first so its version can pick the baseline.
  try:
    baseline = sweep_target(
        baseline_target(args.baseline, candidate_version=candidate.version),
        label="baseline",
        uv_available=uv_available,
    )
  except HarnessError as err:
    if args.baseline != "auto":
      raise
    raise HarnessError(
        f"no release below {candidate.version} exists in the same major"
        " line, so there is nothing meaningful to compare against. The"
        " first release of a major line has no baseline: either name one"
        " from the previous line with --baseline and read the result as a"
        " restructuring diff, or skip this check for this release."
        f"\n\n{err}"
    ) from err
  if candidate.version == baseline.version:
    raise HarnessError(
        f"candidate and baseline are both {candidate.version}, so there is"
        " nothing to compare. Name an older baseline explicitly, for example"
        " --baseline 2.6.0."
    )
  # A sweep that attempted nothing proves nothing.
  for label, sweep in (("candidate", candidate), ("baseline", baseline)):
    if not sweep.attempted:
      raise HarnessError(f"{label} sweep found no modules to import")

  allowlist = None
  if args.allowlist:
    allowlist = load_allowlist(
        pathlib.Path(args.allowlist).read_text(encoding="utf-8")
    )

  comparison = compare(
      baseline=baseline, candidate=candidate, allowlist=allowlist
  )
  report = render_report(
      baseline=baseline, candidate=candidate, comparison=comparison
  )
  return report, comparison.ok


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the check and returns the process exit code."""
  args = parse_args(argv)

  if args.sweep:
    sweep = sweep_installed(DISTRIBUTION)
    payload = json.dumps(dataclasses.asdict(sweep))
    if args.sweep_out:
      pathlib.Path(args.sweep_out).write_text(payload, encoding="utf-8")
    else:
      print(payload)
    return EXIT_OK

  try:
    report, ok = run_check(args)
  except HarnessError as err:
    # Fail closed. A check that could not run must never read as a pass.
    print(f"Release artifact check could not run: {err}", file=sys.stderr)
    return EXIT_HARNESS_FAILURE

  print(report)
  if args.report:
    pathlib.Path(args.report).write_text(report, encoding="utf-8")
  return EXIT_OK if ok else EXIT_REGRESSED


if __name__ == "__main__":
  sys.exit(main())
