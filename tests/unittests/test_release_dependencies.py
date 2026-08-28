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

"""Guard tests for the release-cut dependency contract.

These tests pin the public dependency surface so that release-blocking
regressions documented in the bare-install audit cannot silently re-emerge:

* ``packaging`` MUST be declared in main deps (used at import-time by
  ``utils/model_name_utils.py`` and ``cli/cli_deploy.py``; reachable from
  ``from google.adk import Runner`` and from ``adk --help``).
* ``ValidationError`` in ``environment_simulation_config`` MUST come from
  ``pydantic`` (which always installs alongside the package), NOT from the
  undeclared ``pydantic_core``.
* The LangGraph extras MUST exclude the releases that reconstruct unsafe
  objects while deserializing checkpoint data.
* ``google-genai`` MUST exclude 2.11 and floor at 2.12.1 or later, since that
  release is the first whose types module defers the optional MCP server stack
  instead of importing it at Agent startup.
* The ``all`` extra MUST stay the union of every extra that unlocks a runtime
  feature, so that ``pip install "google-adk[all]"`` cannot silently stop
  installing a feature's dependencies.
* Every ``<=`` upper bound MUST name the release the tests run against, so
  that raising one cannot claim support for a release nothing installed.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

try:
  import tomllib
except ImportError:
  import tomli as tomllib

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion
from packaging.version import Version
import pytest

# Releases that can reconstruct unsafe objects while deserializing checkpoint
# data, mapped to the first release of the same distribution without it.
_UNSAFE_CHECKPOINT_RELEASES = {
    'langgraph': (('0.2.60', '0.4.7', '1.0.9'), '1.0.10'),
    'langgraph-checkpoint': (('2.1.0', '3.0.0', '4.0.0', '4.1.0'), '4.1.1'),
}

# The first google-genai release whose types module defers the optional MCP
# server stack instead of importing it at Agent startup. The floor may be raised
# past it to pick up newer API surface, but never lowered below it.
_LAZY_MCP_GOOGLE_GENAI_RELEASE = Version('2.12.1')

# Extras that ``all`` deliberately leaves out, for the reason recorded in the
# comment above ``optional-dependencies.all`` in pyproject.toml. Every other
# extra is part of the ``all`` contract, so a new extra joins that contract
# unless it is also listed here.
_NON_RUNTIME_EXTRAS = frozenset({
    'benchmark',
    'community',
    'dev',
    'docs',
    'test',
})


def _find_pyproject() -> Path:
  """Locates pyproject.toml by walking up from this file's directory.

  Handles layouts where pyproject.toml is at an ancestor directory as well as
  layouts where it lives in a sibling build directory next to the package. The
  test tree may be symlinked, so the walk avoids ``.resolve()``.
  """
  start = Path(__file__).parent
  for candidate in [start, *start.parents]:
    direct = candidate / 'pyproject.toml'
    if direct.is_file():
      return direct
    try:
      children = sorted(p for p in candidate.iterdir() if p.is_dir())
    except OSError:
      continue
    for child in children:
      sibling = child / 'pyproject.toml'
      if sibling.is_file():
        return sibling
  raise FileNotFoundError(
      f'Could not find pyproject.toml walking up from {start}.'
  )


_PYPROJECT_PATH = _find_pyproject()


@pytest.fixture(scope='module')
def pyproject() -> dict:
  """Parses the project's pyproject.toml exactly once for the module."""
  with _PYPROJECT_PATH.open('rb') as fh:
    return tomllib.load(fh)


def _requirement_names(requirements: list[str]) -> set[str]:
  """Returns the lowercased PEP 508 distribution names from ``requirements``.

  Strips extras specifiers, version specifiers, and environment markers so the
  caller can do exact-name membership checks.
  """
  names: set[str] = set()
  for req in requirements:
    # Drop everything after a marker, version specifier, or extras block.
    head = req.split(';', 1)[0].strip()
    for sep in ('[', '>', '<', '=', '!', '~', ' '):
      head = head.split(sep, 1)[0]
    names.add(head.strip().lower())
  return names


def _requirement_specifier(
    requirements: list[str], distribution: str
) -> SpecifierSet | None:
  """Returns the version specifier ``requirements`` declares for a dependency.

  Returns ``None`` when the distribution is not declared at all, so callers can
  tell "unconstrained" apart from "absent".
  """
  wanted = canonicalize_name(distribution)
  for requirement in requirements:
    parsed = Requirement(requirement)
    if canonicalize_name(parsed.name) == wanted:
      return parsed.specifier
  return None


def _runtime_extra_requirements(
    pyproject: dict,
) -> dict[str, list[tuple[str, Requirement]]]:
  """Returns what the runtime extras require, keyed by distribution name.

  Each value lists the extras that ask for the distribution, paired with the
  requirement that extra declares, so a failure can name the extra that ``all``
  drifted away from.
  """
  contributors: dict[str, list[tuple[str, Requirement]]] = {}
  for extra, entries in pyproject['project']['optional-dependencies'].items():
    if extra == 'all' or extra in _NON_RUNTIME_EXTRAS:
      continue
    for entry in entries:
      requirement = Requirement(entry)
      key = canonicalize_name(requirement.name)
      contributors.setdefault(key, []).append((extra, requirement))
  return contributors


def _all_extra_requirements(pyproject: dict) -> dict[str, Requirement]:
  """Returns the ``all`` extra's requirements, keyed by distribution name."""
  entries = pyproject['project']['optional-dependencies']['all']
  requirements = (Requirement(entry) for entry in entries)
  return {canonicalize_name(req.name): req for req in requirements}


def _specifier_versions(specifier: SpecifierSet) -> set[Version]:
  """Returns the version literals a specifier mentions.

  Wildcard clauses such as ``==1.2.*`` name no single version and are skipped.
  """
  versions: set[Version] = set()
  for clause in specifier:
    try:
      versions.add(Version(clause.version))
    except InvalidVersion:
      continue
  return versions


def _expected_marker(requirements: list[Requirement]) -> str:
  """Returns the marker text the union of ``requirements`` should carry.

  An empty string means the union must be unmarked. Contributors that disagree
  about a marker install the distribution between them on every environment any
  of them names, so the union goes unmarked rather than under-installing.
  """
  markers = {str(req.marker) if req.marker else '' for req in requirements}
  return markers.pop() if len(markers) == 1 else ''


def _inclusive_upper_bounds(pyproject: dict) -> dict[str, Version]:
  """Returns the release each ``<=`` bound names, keyed by distribution.

  Covers the main dependencies and every extra. Exclusive ``<`` bounds are
  left out on purpose: they rule a release out without saying anything about
  any particular release below it. Where two extras disagree, the higher bound
  wins, because installing that extra on its own admits that release.
  """
  groups = [pyproject['project']['dependencies']]
  groups.extend(pyproject['project']['optional-dependencies'].values())

  bounds: dict[str, Version] = {}
  for group in groups:
    for entry in group:
      requirement = Requirement(entry)
      name = canonicalize_name(requirement.name)
      for clause in requirement.specifier:
        if clause.operator != '<=':
          continue
        # PEP 440 allows a wildcard only with == and !=, so a <= clause always
        # names a parseable release and a failure here is worth hearing about.
        bound = Version(clause.version)
        bounds[name] = max(bound, bounds.get(name, bound))
  return bounds


def test_main_deps_include_packaging(pyproject: dict) -> None:
  """``packaging`` is imported unguarded by core ADK; it must be a main dep."""
  main_deps = _requirement_names(pyproject['project']['dependencies'])
  assert 'packaging' in main_deps, (
      'packaging must be declared in [project] dependencies because '
      'src/google/adk/utils/model_name_utils.py and '
      'src/google/adk/cli/cli_deploy.py import it unguarded at module top '
      'level. Without this declaration, `pip install google-adk` is one '
      'transitive resolver change away from breaking on `import google.adk`.'
  )


@pytest.mark.parametrize('extra', ['extensions', 'test'])
@pytest.mark.parametrize('distribution', sorted(_UNSAFE_CHECKPOINT_RELEASES))
def test_langgraph_extras_exclude_unsafe_checkpoint_releases(
    pyproject: dict, extra: str, distribution: str
) -> None:
  """Both LangGraph extras resolve past the unsafe-deserialization releases.

  ``langgraph`` does not constrain ``langgraph-checkpoint`` tightly enough to
  rule the unsafe releases out on its own, so each extra must declare both.
  """
  unsafe_versions, first_safe = _UNSAFE_CHECKPOINT_RELEASES[distribution]
  specifier = _requirement_specifier(
      pyproject['project']['optional-dependencies'][extra], distribution
  )

  assert specifier is not None, (
      f'The {extra!r} extra must declare {distribution}; without it the '
      'resolver is free to install a release that can reconstruct unsafe '
      'objects from checkpoint data.'
  )
  admitted = [v for v in unsafe_versions if specifier.contains(v)]
  assert not admitted, (
      f'The {extra!r} extra admits {distribution} {admitted}, which can '
      'reconstruct unsafe objects from checkpoint data. Require '
      f'{distribution}>={first_safe}.'
  )
  assert specifier.contains(first_safe), (
      f'The {extra!r} extra excludes {distribution} {first_safe}, the first '
      'release without the unsafe behavior.'
  )


def test_main_deps_require_lazy_mcp_google_genai_release(
    pyproject: dict,
) -> None:
  """The google-genai floor preserves its lazy optional-MCP boundary."""
  requirements = [
      Requirement(raw) for raw in pyproject['project']['dependencies']
  ]
  google_genai = next(
      requirement
      for requirement in requirements
      if requirement.name == 'google-genai'
  )

  floor = min(
      Version(specifier.version)
      for specifier in google_genai.specifier
      if specifier.operator in ('>=', '==')
  )

  assert Version('2.11.0') not in google_genai.specifier
  assert floor >= _LAZY_MCP_GOOGLE_GENAI_RELEASE, (
      f'The google-genai floor {floor} predates'
      f' {_LAZY_MCP_GOOGLE_GENAI_RELEASE}, the first release whose types module'
      ' defers the optional MCP server stack.'
  )


def test_inclusive_upper_bounds_ignores_other_operators() -> None:
  """Only ``<=`` names a release as supported, and the highest one wins."""
  bounds = _inclusive_upper_bounds({
      'project': {
          'dependencies': [
              'named<=1.43',
              'floored>=1.39',
              'capped<2',
              'excluded!=3.1',
          ],
          'optional-dependencies': {
              'low': ['spread>=1,<=1.0'],
              'high': ['spread>=1,<=2.0'],
          },
      }
  })

  assert bounds == {'named': Version('1.43'), 'spread': Version('2.0')}


def test_inclusive_upper_bounds_name_an_installed_release(
    pyproject: dict,
) -> None:
  """Every ``<=`` bound names the release the tests are running against.

  ``opentelemetry-api<=1.43`` claims release 1.43 works. ``<1.44`` claims
  release 1.44 does not. The second is knowable from an upstream changelog;
  the first is only knowable by running it. So an inclusive bound that names a
  release the environment never installed is a claim no test stands behind,
  and raising one ships that claim to users on the next release.
  """
  bounds = _inclusive_upper_bounds(pyproject)

  checked = 0
  unexercised: list[str] = []
  for distribution, bound in sorted(bounds.items()):
    try:
      installed = Version(importlib.metadata.version(distribution))
    except importlib.metadata.PackageNotFoundError:
      # Declared by an extra this environment does not install.
      continue
    checked += 1
    if installed < bound:
      unexercised.append(f'{distribution}<={bound}, running {installed}')

  assert checked, (
      'No declared <= bound was compared against an installed release, so '
      'this test proved nothing. Either pyproject.toml stopped using '
      'inclusive upper bounds, or _inclusive_upper_bounds stopped finding '
      'them.'
  )
  assert not unexercised, (
      'These bounds name a release the tests never ran: '
      + '; '.join(unexercised)
      + '. Install the named release so the suite covers what the bound '
      'promises, or state what is actually known with an exclusive bound '
      '(<1.44 rules out the 1.44 line without claiming 1.43 was exercised).'
  )


def test_all_extra_covers_every_runtime_extra(pyproject: dict) -> None:
  """``all`` names exactly the distributions the runtime extras name.

  This is the guard against the failure that motivated the union: an extra
  gains a dependency, nobody mirrors it into ``all``, and users who installed
  ``google-adk[all]`` hit an ImportError for a feature they believed they had.
  """
  contributors = _runtime_extra_requirements(pyproject)
  all_extra = _all_extra_requirements(pyproject)

  missing = sorted(set(contributors) - set(all_extra))
  assert not missing, 'The all extra is missing ' + ', '.join(
      f'{name} (required by'
      f' {", ".join(sorted(e for e, _ in contributors[name]))})'
      for name in missing
  )

  orphaned = sorted(set(all_extra) - set(contributors))
  assert not orphaned, (
      f'The all extra requires {", ".join(orphaned)}, which no runtime extra '
      'declares. Every entry in all belongs to the extra that owns its '
      'feature, so either declare it there or drop it from all.'
  )


def test_all_extra_preserves_runtime_extra_constraints(pyproject: dict) -> None:
  """``all`` asks for each distribution on the same terms its extras do.

  Installing several extras at once yields the union of their distributions
  and the intersection of their version constraints, so ``all`` must request
  the union of the sub-extras named, admit a version exactly when every
  contributing extra admits it, and carry the environment marker its
  contributors agree on.
  """
  contributors = _runtime_extra_requirements(pyproject)
  all_extra = _all_extra_requirements(pyproject)
  problems: list[str] = []

  for name, sources in sorted(contributors.items()):
    combined = all_extra.get(name)
    if combined is None:
      continue  # Already reported as missing by the coverage test.
    extras = sorted(extra for extra, _ in sources)
    requirements = [requirement for _, requirement in sources]

    wanted_extras = set().union(*(req.extras for req in requirements))
    if combined.extras != wanted_extras:
      problems.append(
          f'{name}: all requests sub-extras {sorted(combined.extras)}, but '
          f'{extras} together require {sorted(wanted_extras)}'
      )

    wanted_marker = _expected_marker(requirements)
    actual_marker = str(combined.marker) if combined.marker else ''
    if actual_marker != wanted_marker:
      problems.append(
          f'{name}: all is gated on {actual_marker or "nothing"}, but '
          f'{extras} require {wanted_marker or "no marker"}'
      )

    candidates = _specifier_versions(combined.specifier)
    for requirement in requirements:
      candidates |= _specifier_versions(requirement.specifier)
    for version in sorted(candidates):
      admitted_by_all = combined.specifier.contains(version, prereleases=True)
      admitted_by_extras = all(
          req.specifier.contains(version, prereleases=True)
          for req in requirements
      )
      if admitted_by_all != admitted_by_extras:
        problems.append(
            f'{name}: all and {extras} disagree about version {version}. all '
            f'declares {combined.specifier or "no constraint"}, against '
            + ', '.join(
                f'{extra}: {req.specifier or "no constraint"}'
                for extra, req in sources
            )
        )

  assert not problems, 'The all extra diverges from its extras:\n' + '\n'.join(
      problems
  )


def test_environment_simulation_config_imports_validation_error_from_pydantic() -> (
    None
):
  """The ValidationError used by the config module must come from pydantic.

  pydantic-core is undeclared; importing from it directly is fragile. pydantic
  re-exports ValidationError, so use that.
  """
  # Use importlib to locate the source file so the test is independent of the
  # on-disk package layout.
  spec = importlib.util.find_spec(
      'google.adk.tools.environment_simulation.environment_simulation_config'
  )
  assert (
      spec is not None and spec.origin is not None
  ), 'environment_simulation_config module is not importable.'
  source_path = Path(spec.origin)
  source = source_path.read_text(encoding='utf-8')
  assert 'from pydantic import ValidationError' in source, (
      'environment_simulation_config.py must import ValidationError from '
      'pydantic, not pydantic_core. pydantic_core is undeclared as a main '
      'dep and pydantic re-exports the same class.'
  )
  assert 'from pydantic_core import ValidationError' not in source, (
      'environment_simulation_config.py must not import ValidationError '
      'from pydantic_core (undeclared dep).'
  )


def test_injection_config_validation_raises_pydantic_validation_error() -> None:
  """Behavioral check: invalid config raises the pydantic ValidationError."""
  # Local import keeps this test focused on the post-fix code path and
  # surfaces ImportError clearly if the module's import block regresses.
  from google.adk.tools.environment_simulation.environment_simulation_config import InjectedError
  from pydantic import ValidationError

  with pytest.raises(ValidationError):
    # Both required fields missing — pydantic must reject the construction.
    InjectedError()  # type: ignore[call-arg]
