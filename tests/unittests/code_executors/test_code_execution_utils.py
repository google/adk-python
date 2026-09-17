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

import base64
import multiprocessing
import time
import traceback

from google.adk.code_executors import code_execution_utils
from google.genai import types

# The extraction itself must finish promptly. The join budget is far looser
# because it also covers spawning the child and importing this module there.
_REDOS_DEADLINE_SECONDS = 2.0
_CHILD_JOIN_TIMEOUT_SECONDS = 120.0


def _exercise_redos_candidate(result_conn) -> None:
  """Runs the ReDoS regression payload in an independently stoppable process."""
  failure = None
  try:
    ticks = "`" * 3
    long_invalid_payload = (
        ticks + "python\n" + "x = 1\n" * 5000 + "not_matching"
    )
    content = types.Content(
        role="model",
        parts=[types.Part(text=long_invalid_payload)],
    )
    delimiters = [(ticks + "python\n", "\n" + ticks)]

    started = time.perf_counter()
    code = code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
        content, delimiters
    )
    elapsed = time.perf_counter() - started

    if code is not None:
      failure = f"expected no code to be extracted, got {code!r}"
    elif elapsed > _REDOS_DEADLINE_SECONDS:
      failure = (
          f"extraction took {elapsed:.3f}s, over the"
          f" {_REDOS_DEADLINE_SECONDS}s deadline (possible ReDoS regression)"
      )
  except BaseException:  # pylint: disable=broad-except
    # Without this the parent only sees a bare exit code and has to dig the
    # traceback out of the child's captured stderr.
    failure = f"extraction raised in the child:\n{traceback.format_exc()}"
  result_conn.send(failure)
  result_conn.close()


def test_extract_code_and_truncate_content_basic():
  """Tests basic code extraction and content truncation."""
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text=(
                  "Here is some code:\n```python\nx = 1\n```\nAnd some text"
                  " after."
              )
          )
      ],
  )
  delimiters = [("```python\n", "\n```")]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "x = 1"
  assert len(content.parts) == 2
  assert content.parts[0].text == "Here is some code:\n"
  assert content.parts[1].executable_code.code == "x = 1"


def test_extract_code_and_truncate_content_multiple_blocks():
  """Tests that the first code block is extracted when multiple exist."""
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text=(
                  "First:\n"
                  "```python\n"
                  "x = 1\n"
                  "```\n"
                  "Second:\n"
                  "```python\n"
                  "y = 2\n"
                  "```"
              )
          )
      ],
  )
  delimiters = [("```python\n", "\n```")]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "x = 1"
  assert len(content.parts) == 2
  assert content.parts[0].text == "First:\n"
  assert content.parts[1].executable_code.code == "x = 1"


def test_extract_code_and_truncate_content_no_delimiter():
  """Tests when no delimiters are found in the content."""
  content = types.Content(
      role="model",
      parts=[types.Part(text="Just plain text without code.")],
  )
  delimiters = [("```python\n", "\n```")]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code is None
  # Content should be unmodified.
  assert len(content.parts) == 1
  assert content.parts[0].text == "Just plain text without code."


def test_extract_code_and_truncate_content_redos_vulnerability():
  """Tests that a string that would cause ReDoS behaves reasonably."""
  context = multiprocessing.get_context("spawn")
  receiver, sender = context.Pipe(duplex=False)
  process = context.Process(target=_exercise_redos_candidate, args=(sender,))
  process.start()
  sender.close()
  process.join(timeout=_CHILD_JOIN_TIMEOUT_SECONDS)
  hung = process.is_alive()
  if hung:
    process.kill()
    process.join()
  exitcode = process.exitcode
  try:
    # poll() is also true at EOF, so recv() has to carry the "child died
    # without reporting" case rather than a poll() guard.
    failure = receiver.recv()
  except EOFError:
    failure = "child exited without reporting a result"
  finally:
    receiver.close()
    process.close()

  assert not hung, "extraction never returned (possible ReDoS regression)"
  assert failure is None, failure
  assert exitcode == 0, f"extraction process exited with {exitcode}"


def test_extract_code_and_truncate_content_multiple_delimiter_pairs():
  """Tests code extraction when multiple different delimiter pairs are provided."""
  ticks = "`" * 3
  # Case 1: First delimiter pair matches first
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text="Here is tool code:\n"
              + ticks
              + "tool_code\nx = 1\n"
              + ticks
              + "\nAnd python code:\n"
              + ticks
              + "python\ny = 2\n"
              + ticks
          )
      ],
  )
  delimiters = [
      (ticks + "tool_code\n", "\n" + ticks),
      (ticks + "python\n", "\n" + ticks),
  ]
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "x = 1"
  assert len(content.parts) == 2
  assert content.parts[0].text == "Here is tool code:\n"
  assert content.parts[1].executable_code.code == "x = 1"

  # Case 2: Second delimiter pair matches first
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              text="Here is python code:\n"
              + ticks
              + "python\ny = 2\n"
              + ticks
              + "\nAnd tool code:\n"
              + ticks
              + "tool_code\nx = 1\n"
              + ticks
          )
      ],
  )
  code = (
      code_execution_utils.CodeExecutionUtils.extract_code_and_truncate_content(
          content, delimiters
      )
  )
  assert code == "y = 2"
  assert len(content.parts) == 2
  assert content.parts[0].text == "Here is python code:\n"
  assert content.parts[1].executable_code.code == "y = 2"


def test_get_encoded_file_content_encodes_raw_bytes():
  """Raw binary must come back base64-encoded, not verbatim."""
  encoded = code_execution_utils.CodeExecutionUtils.get_encoded_file_content(
      b"\x00\x01\x02"
  )
  # base64 of the three bytes 00 01 02 is "AAEC" (no padding needed).
  assert encoded == b"AAEC"


def test_get_encoded_file_content_encodes_payload_with_invalid_padding():
  """A payload that is not decodable base64 is encoded, not passed through."""
  encoded = code_execution_utils.CodeExecutionUtils.get_encoded_file_content(
      b"hello"
  )
  assert encoded == b"aGVsbG8="


def test_get_encoded_file_content_leaves_already_encoded_bytes_unchanged():
  """Double-encoding would corrupt the file for the executor that decodes it."""
  already_encoded = base64.b64encode(b"file,contents\n1,2\n")
  encoded = code_execution_utils.CodeExecutionUtils.get_encoded_file_content(
      already_encoded
  )
  assert encoded == already_encoded
  assert base64.b64decode(encoded) == b"file,contents\n1,2\n"


def test_get_encoded_file_content_is_idempotent():
  once = code_execution_utils.CodeExecutionUtils.get_encoded_file_content(
      b"\x00\x01\x02"
  )
  twice = code_execution_utils.CodeExecutionUtils.get_encoded_file_content(once)
  assert twice == once


def test_build_executable_code_part_carries_code_and_python_language():
  part = code_execution_utils.CodeExecutionUtils.build_executable_code_part(
      "print(1)"
  )
  assert part.executable_code.code == "print(1)"
  assert part.executable_code.language == types.Language.PYTHON


def test_build_code_execution_result_part_stderr_reports_failure():
  """stderr wins over stdout: a run that wrote to stderr did not succeed."""
  result = code_execution_utils.CodeExecutionResult(
      stdout="partial output", stderr="Traceback: boom"
  )
  part = (
      code_execution_utils.CodeExecutionUtils.build_code_execution_result_part(
          result
      )
  )
  assert part.code_execution_result.outcome == types.Outcome.OUTCOME_FAILED
  # The failure text is the stderr verbatim, so the model sees the real error.
  assert part.code_execution_result.output == "Traceback: boom"


def test_build_code_execution_result_part_stdout_only():
  result = code_execution_utils.CodeExecutionResult(stdout="42")
  part = (
      code_execution_utils.CodeExecutionUtils.build_code_execution_result_part(
          result
      )
  )
  assert part.code_execution_result.outcome == types.Outcome.OUTCOME_OK
  assert part.code_execution_result.output == "Code execution result:\n42\n"


def test_build_code_execution_result_part_empty_run_still_reports_result():
  """A silent successful run still gets a result header, not an empty string."""
  result = code_execution_utils.CodeExecutionResult()
  part = (
      code_execution_utils.CodeExecutionUtils.build_code_execution_result_part(
          result
      )
  )
  assert part.code_execution_result.outcome == types.Outcome.OUTCOME_OK
  assert part.code_execution_result.output == "Code execution result:\n\n"


def test_build_code_execution_result_part_files_only_omits_result_header():
  """With no stdout but saved files, only the artifact list is reported."""
  result = code_execution_utils.CodeExecutionResult(
      output_files=[
          code_execution_utils.File(name="a.csv", content=""),
          code_execution_utils.File(name="b.png", content=""),
      ]
  )
  part = (
      code_execution_utils.CodeExecutionUtils.build_code_execution_result_part(
          result
      )
  )
  assert part.code_execution_result.outcome == types.Outcome.OUTCOME_OK
  assert (
      part.code_execution_result.output == "Saved artifacts:\n`a.csv`,`b.png`"
  )


def test_build_code_execution_result_part_stdout_and_files():
  result = code_execution_utils.CodeExecutionResult(
      stdout="done",
      output_files=[code_execution_utils.File(name="a.csv", content="")],
  )
  part = (
      code_execution_utils.CodeExecutionUtils.build_code_execution_result_part(
          result
      )
  )
  assert part.code_execution_result.output == (
      "Code execution result:\ndone\n\n\nSaved artifacts:\n`a.csv`"
  )


def test_convert_code_execution_parts_rewrites_trailing_executable_code():
  content = types.Content(
      role="model",
      parts=[
          types.Part(text="here goes:"),
          code_execution_utils.CodeExecutionUtils.build_executable_code_part(
              "x = 1"
          ),
      ],
  )

  code_execution_utils.CodeExecutionUtils.convert_code_execution_parts(
      content, ("<code>", "</code>"), ("<out>", "</out>")
  )

  # The leading text part is left alone; only the trailing code part becomes
  # text, wrapped in the code delimiters.
  assert content.parts[0].text == "here goes:"
  assert content.parts[1].text == "<code>x = 1</code>"
  assert content.parts[1].executable_code is None
  assert content.role == "model"


def test_convert_code_execution_parts_rewrites_lone_execution_result_as_user():
  content = types.Content(
      role="model",
      parts=[
          types.Part.from_code_execution_result(
              outcome="OUTCOME_OK", output="42"
          )
      ],
  )

  code_execution_utils.CodeExecutionUtils.convert_code_execution_parts(
      content, ("<code>", "</code>"), ("<out>", "</out>")
  )

  assert content.parts[0].text == "<out>42</out>"
  # The execution result was produced by the executor, not the model, so the
  # rewritten turn is attributed to the user.
  assert content.role == "user"


def test_convert_code_execution_parts_keeps_multipart_execution_result():
  """A multi-part content came from the model, so its result is left as-is."""
  content = types.Content(
      role="model",
      parts=[
          types.Part(text="the answer is"),
          types.Part.from_code_execution_result(
              outcome="OUTCOME_OK", output="42"
          ),
      ],
  )

  code_execution_utils.CodeExecutionUtils.convert_code_execution_parts(
      content, ("<code>", "</code>"), ("<out>", "</out>")
  )

  assert content.parts[1].text is None
  assert content.parts[1].code_execution_result.output == "42"
  assert content.role == "model"


def test_convert_code_execution_parts_execution_result_without_output():
  content = types.Content(
      role="model",
      parts=[
          types.Part(
              code_execution_result=types.CodeExecutionResult(
                  outcome="OUTCOME_OK"
              )
          )
      ],
  )

  code_execution_utils.CodeExecutionUtils.convert_code_execution_parts(
      content, ("<code>", "</code>"), ("<out>", "</out>")
  )

  # No output means no delimiters either - an empty text part, not "<out></out>".
  assert content.parts[0].text == ""
  assert content.role == "user"


def test_convert_code_execution_parts_empty_parts_is_a_noop():
  content = types.Content(role="model", parts=[])

  code_execution_utils.CodeExecutionUtils.convert_code_execution_parts(
      content, ("<code>", "</code>"), ("<out>", "</out>")
  )

  assert content.parts == []
  assert content.role == "model"
