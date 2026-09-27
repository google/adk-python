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

"""Unit tests for check_docs.py."""

from __future__ import annotations

import pathlib

from scripts import check_docs


def write(path: pathlib.Path, text: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding='utf-8')


def test_slugify_matches_rendered_heading_ids() -> None:
  assert check_docs.slugify('Choose an implementation') == (
      'choose-an-implementation'
  )
  assert check_docs.slugify('`ModelArmorConfig` fields') == (
      'modelarmorconfig-fields'
  )
  assert check_docs.slugify('What about *state*?') == 'what-about-state'
  assert check_docs.slugify('[Runner](runner.md) live') == 'runner-live'
  assert check_docs.slugify('inject_session_state') == 'inject_session_state'
  assert check_docs.slugify('The `to_a2a` helper') == 'the-to_a2a-helper'


def test_repeated_headings_get_numbered_anchors() -> None:
  anchors = check_docs.anchors_in('# Setup\n\n# Setup\n\n# Setup\n')
  assert anchors == {'setup', 'setup-1', 'setup-2'}


def test_html_anchor_is_a_link_target() -> None:
  assert 'custom' in check_docs.anchors_in('<a name="custom"></a>\n# Title\n')


def test_html_anchor_inside_a_code_fence_is_not_a_link_target() -> None:
  assert not check_docs.anchors_in('```html\n<a name="sample"></a>\n```\n')


def test_headings_and_links_inside_code_fences_are_ignored() -> None:
  text = '# Real\n\n```markdown\n# Fake\n[x](nowhere.md)\n```\n'
  assert check_docs.anchors_in(text) == {'real'}
  assert not check_docs.links_in(text)


def test_fence_quoted_in_prose_does_not_open_a_block() -> None:
  # Prose about markdown writes a fence inline; a naive parser reads it as the
  # start of a code block and stops seeing every heading after it.
  text = 'Delimiters are ```` ```python ```` here.\n\n# Later\n'
  assert 'later' in check_docs.anchors_in(text)


def test_broken_relative_link_is_reported(tmp_path: pathlib.Path) -> None:
  write(tmp_path / 'docs' / 'a.md', '[gone](missing.md)\n')

  problems = check_docs.check_links(
      str(tmp_path), [str(tmp_path / 'docs' / 'a.md')]
  )

  assert len(problems) == 1
  assert 'missing.md does not exist' in problems[0].detail


def test_resolvable_link_and_anchor_pass(tmp_path: pathlib.Path) -> None:
  write(tmp_path / 'docs' / 'b.md', '# Get started\n')
  write(
      tmp_path / 'docs' / 'a.md',
      '[b](b.md#get-started)\n\n[self](#own)\n\n## Own\n',
  )

  assert not check_docs.check_links(
      str(tmp_path), [str(tmp_path / 'docs' / 'a.md')]
  )


def test_unknown_fragment_is_reported(tmp_path: pathlib.Path) -> None:
  write(tmp_path / 'docs' / 'b.md', '# Get started\n')
  write(tmp_path / 'docs' / 'a.md', '[b](b.md#setup)\n')

  problems = check_docs.check_links(
      str(tmp_path), [str(tmp_path / 'docs' / 'a.md')]
  )

  assert len(problems) == 1
  assert 'names no heading' in problems[0].detail


def test_external_links_are_not_followed(tmp_path: pathlib.Path) -> None:
  write(
      tmp_path / 'docs' / 'a.md',
      '[site](https://adk.dev/nope)\n\n[mail](mailto:nobody@example.com)\n',
  )

  assert not check_docs.check_links(
      str(tmp_path), [str(tmp_path / 'docs' / 'a.md')]
  )


def test_reference_definitions_and_html_targets_are_checked(
    tmp_path: pathlib.Path,
) -> None:
  write(
      tmp_path / 'docs' / 'a.md',
      'Text [ref] and <img src="picture.png">\n\n[ref]: elsewhere.md\n',
  )

  problems = check_docs.check_links(
      str(tmp_path), [str(tmp_path / 'docs' / 'a.md')]
  )

  assert {'elsewhere.md does not exist', 'picture.png does not exist'} == {
      problem.detail.split(': ', 1)[1] for problem in problems
  }


def test_guide_missing_from_index_is_reported(tmp_path: pathlib.Path) -> None:
  guides = tmp_path / 'docs' / 'guides'
  write(guides / 'README.md', '# Index\n\n* [Listed](listed/index.md)\n')
  write(guides / 'listed' / 'index.md', '# Listed\n')
  write(guides / 'orphan' / 'index.md', '# Orphan\n')

  problems = check_docs.check_index_coverage(str(tmp_path))

  assert len(problems) == 1
  assert problems[0].detail.startswith('orphan/index.md is not listed')


def test_fully_indexed_guides_pass(tmp_path: pathlib.Path) -> None:
  guides = tmp_path / 'docs' / 'guides'
  write(
      guides / 'README.md',
      '# Index\n\n* [One](one/index.md)\n* [Two](two.md#section)\n',
  )
  write(guides / 'one' / 'index.md', '# One\n')
  write(guides / 'two.md', '# Two\n\n## Section\n')

  assert not check_docs.check_index_coverage(str(tmp_path))


def test_missing_index_is_reported(tmp_path: pathlib.Path) -> None:
  write(tmp_path / 'docs' / 'guides' / 'one.md', '# One\n')

  problems = check_docs.check_index_coverage(str(tmp_path))

  assert len(problems) == 1
  assert 'index is missing' in problems[0].detail


def test_main_returns_one_when_docs_have_problems(
    tmp_path: pathlib.Path,
) -> None:
  write(tmp_path / 'docs' / 'guides' / 'README.md', '# Index\n')
  write(tmp_path / 'docs' / 'a.md', '[gone](missing.md)\n')

  assert check_docs.main(['--root', str(tmp_path)]) == 1


def test_main_returns_two_when_root_has_no_docs_dir(
    tmp_path: pathlib.Path,
) -> None:
  assert check_docs.main(['--root', str(tmp_path)]) == 2


def test_shipped_docs_are_clean() -> None:
  root = pathlib.Path(check_docs.__file__).resolve().parent.parent

  problems = check_docs.check(str(root))

  assert not problems, '\n'.join(
      f'{problem.path}: {problem.detail}' for problem in problems
  )
