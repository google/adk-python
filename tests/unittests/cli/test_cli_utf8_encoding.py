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

"""Tests that the CLI reads and writes text files as UTF-8.

`open()` without an explicit `encoding` uses the platform's locale encoding,
which is not UTF-8 on a large share of Windows installs (`cp936` for zh-CN,
`cp932` for ja-JP, `cp1252` for much of the West). The CLI's JSON and Markdown
files carry agent prompts, model responses and display names, so they routinely
hold non-ASCII text -- and JSON is defined as UTF-8 for interchange (RFC 8259
§8.1). Reading those files under a non-UTF-8 locale either raises
`UnicodeDecodeError` or, when the UTF-8 bytes happen to form a valid sequence in
the locale codec, silently yields mojibake.

CI runs on a UTF-8 default, so a test that merely reads a UTF-8 file cannot
catch a missing `encoding=`. `_non_utf8_default_locale` therefore simulates the
locale by forcing a non-UTF-8 codec onto every `open()` call that omits
`encoding`, which makes these tests fail on any platform if the argument is
dropped again.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import pathlib
from typing import Iterator

import google.adk.cli as adk_cli
from google.adk.cli.cli_deploy import _get_ignore_patterns_func
import pytest

_NON_ASCII = "北京天气-café-🌤"


@contextlib.contextmanager
def _non_utf8_default_locale(codec: str = "ascii") -> Iterator[None]:
  """Forces `codec` onto text-mode `open()` calls that omit `encoding`.

  Patching `locale.getpreferredencoding` does not work: CPython resolves the
  default text encoding internally, so `open()` ignores it. Wrapping
  `builtins.open` is what actually reproduces a non-UTF-8 locale on a UTF-8 CI
  machine.

  `ascii` is the default because it rejects every non-ASCII byte, so a missing
  `encoding=` fails deterministically. A real locale codec is laxer: cp936
  accepts most UTF-8 byte pairs and yields mojibake instead of raising, which
  `test_locale_codec_can_corrupt_silently` covers separately.
  """
  real_open = builtins.open

  def patched_open(  # pylint: disable=redefined-builtin
      file, mode="r", buffering=-1, encoding=None, *args, **kwargs
  ):
    if "b" not in mode and encoding is None:
      encoding = codec
    return real_open(file, mode, buffering, encoding, *args, **kwargs)

  builtins.open = patched_open
  try:
    yield
  finally:
    builtins.open = real_open


def test_non_utf8_default_locale_helper_actually_bites(tmp_path):
  """Guards the guard: the helper must really change `open()`'s behavior."""
  path = tmp_path / "probe.txt"
  path.write_text(_NON_ASCII, encoding="utf-8")

  with _non_utf8_default_locale():
    with pytest.raises(UnicodeDecodeError):
      with open(path) as f:  # no encoding= -> forced cp936
        f.read()

    # An explicit encoding still wins over the simulated locale.
    with open(path, encoding="utf-8") as f:
      assert f.read() == _NON_ASCII


def test_locale_codec_can_corrupt_silently(tmp_path):
  """Documents the quieter failure mode: no exception, wrong text.

  cp936 decodes most UTF-8 byte pairs without complaint, so a missing
  `encoding=` on a zh-CN Windows box can persist mojibake into an eval set
  instead of raising.
  """
  path = tmp_path / "probe.txt"
  path.write_text(_NON_ASCII, encoding="utf-8")

  with _non_utf8_default_locale("cp936"):
    with open(path) as f:
      decoded = f.read()

  assert decoded != _NON_ASCII  # silently wrong, and no error was raised


def test_get_ignore_patterns_reads_utf8_ignore_file(tmp_path):
  """`.gitignore` entries with non-ASCII names survive a non-UTF-8 locale."""
  (tmp_path / ".gitignore").write_text(
      f"# comment\n{_NON_ASCII}/\n__pycache__\n", encoding="utf-8"
  )

  with _non_utf8_default_locale():
    ignore_func = _get_ignore_patterns_func(str(tmp_path))

  patterns = ignore_func(str(tmp_path), [f"{_NON_ASCII}", "__pycache__"])
  assert _NON_ASCII in patterns
  assert "__pycache__" in patterns


def _text_mode_open_calls_without_encoding(root: pathlib.Path) -> list[str]:
  """Returns `file:line` for each text-mode `open()` that omits `encoding`."""
  offenders = []
  for path in sorted(root.rglob("*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
      # Bare `open(...)` only: `ZipFile.open`/`z.open` are binary and take no
      # `encoding`, and `os.fdopen` sites here are binary too.
      if (
          not isinstance(node, ast.Call)
          or getattr(node.func, "id", None) != "open"
      ):
        continue
      if any(kw.arg == "encoding" for kw in node.keywords):
        continue
      mode = "r"
      if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
      for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
          mode = kw.value.value
      if isinstance(mode, str) and "b" not in mode:
        offenders.append(f"{path}:{node.lineno}")
  return offenders


def test_cli_package_has_no_implicit_encoding_open():
  """Every text-mode `open()` under `cli/` must pass `encoding` explicitly.

  This class of defect has been fixed piecemeal several times (#2049, #5820,
  #6288, #6298), so it is asserted for the whole package rather than per call
  site.
  """
  root = pathlib.Path(adk_cli.__file__).parent
  offenders = _text_mode_open_calls_without_encoding(root)
  assert not offenders, (
      "text-mode open() without encoding= (locale-dependent, breaks on"
      f" non-UTF-8 locales): {offenders}"
  )
