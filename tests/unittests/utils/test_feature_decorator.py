import os
import tempfile
import warnings

from google.adk.utils.feature_decorator import experimental
from google.adk.utils.feature_decorator import working_in_progress


@working_in_progress("in complete feature, don't use yet")
class IncompleteFeature:

  def run(self):
    return "running"


@working_in_progress("function not ready")
def wip_function():
  return "executing"


@experimental("api may have breaking change in the future.")
def experimental_fn():
  return "executing"


@experimental("class may change")
class ExperimentalClass:

  def run(self):
    return "running experimental"


# Test classes/functions for new usage patterns
@experimental
class ExperimentalClassNoParens:

  def run(self):
    return "running experimental without parens"


@experimental()
class ExperimentalClassEmptyParens:

  def run(self):
    return "running experimental with empty parens"


@experimental
def experimental_fn_no_parens():
  return "executing without parens"


@experimental()
def experimental_fn_empty_parens():
  return "executing with empty parens"


def test_working_in_progress_class_raises_error():
  """Test that WIP class raises RuntimeError by default."""
  # Ensure environment variable is not set
  if "ADK_ALLOW_WIP_FEATURES" in os.environ:
    del os.environ["ADK_ALLOW_WIP_FEATURES"]

  try:
    feature = IncompleteFeature()
    assert False, "Expected RuntimeError to be raised"
  except RuntimeError as e:
    assert "[WIP] IncompleteFeature:" in str(e)
    assert "don't use yet" in str(e)


def test_working_in_progress_function_raises_error():
  """Test that WIP function raises RuntimeError by default."""
  # Ensure environment variable is not set
  if "ADK_ALLOW_WIP_FEATURES" in os.environ:
    del os.environ["ADK_ALLOW_WIP_FEATURES"]

  try:
    result = wip_function()
    assert False, "Expected RuntimeError to be raised"
  except RuntimeError as e:
    assert "[WIP] wip_function:" in str(e)
    assert "function not ready" in str(e)


def test_working_in_progress_class_bypassed_with_env_var():
  """Test that WIP class works without warnings when env var is set."""
  # Set the bypass environment variable
  os.environ["ADK_ALLOW_WIP_FEATURES"] = "true"

  try:
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      feature = IncompleteFeature()
      result = feature.run()

      assert result == "running"
      # Should have no warnings when bypassed
      assert len(w) == 0
  finally:
    # Clean up environment variable
    if "ADK_ALLOW_WIP_FEATURES" in os.environ:
      del os.environ["ADK_ALLOW_WIP_FEATURES"]


def test_working_in_progress_function_bypassed_with_env_var():
  """Test that WIP function works without warnings when env var is set."""
  # Set the bypass environment variable
  os.environ["ADK_ALLOW_WIP_FEATURES"] = "true"

  try:
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      result = wip_function()

      assert result == "executing"
      # Should have no warnings when bypassed
      assert len(w) == 0
  finally:
    # Clean up environment variable
    if "ADK_ALLOW_WIP_FEATURES" in os.environ:
      del os.environ["ADK_ALLOW_WIP_FEATURES"]


def test_working_in_progress_env_var_case_insensitive():
  """Test that WIP bypass works with different case values."""
  test_cases = ["true", "True", "TRUE", "tRuE"]

  for case in test_cases:
    os.environ["ADK_ALLOW_WIP_FEATURES"] = case

    try:
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        result = wip_function()

        assert result == "executing"
        assert len(w) == 0
    finally:
      if "ADK_ALLOW_WIP_FEATURES" in os.environ:
        del os.environ["ADK_ALLOW_WIP_FEATURES"]


def test_working_in_progress_env_var_false_values():
  """Test that WIP still raises errors with false-like env var values."""
  false_values = ["false", "False", "FALSE", "0", "", "anything_else"]

  for false_val in false_values:
    os.environ["ADK_ALLOW_WIP_FEATURES"] = false_val

    try:
      result = wip_function()
      assert False, f"Expected RuntimeError with env var '{false_val}'"
    except RuntimeError as e:
      assert "[WIP] wip_function:" in str(e)
    finally:
      if "ADK_ALLOW_WIP_FEATURES" in os.environ:
        del os.environ["ADK_ALLOW_WIP_FEATURES"]


def test_working_in_progress_loads_from_dotenv_file():
  """Test that WIP decorator can load environment variables from .env file."""
  # Skip test if dotenv is not available
  try:
    from dotenv import load_dotenv
  except ImportError:
    import pytest

    pytest.skip("python-dotenv not available")

  # Ensure environment variable is not set in os.environ
  if "ADK_ALLOW_WIP_FEATURES" in os.environ:
    del os.environ["ADK_ALLOW_WIP_FEATURES"]

  # Create a temporary .env file in current directory
  dotenv_path = ".env.test"

  try:
    # Write the env file
    with open(dotenv_path, "w") as f:
      f.write("ADK_ALLOW_WIP_FEATURES=true\n")

    # Load the environment variables from the file
    load_dotenv(dotenv_path)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")

      # This should work because the .env file contains ADK_ALLOW_WIP_FEATURES=true
      result = wip_function()

      assert result == "executing"
      # Should have no warnings when bypassed via .env file
      assert len(w) == 0

  finally:
    # Clean up
    try:
      os.unlink(dotenv_path)
    except FileNotFoundError:
      pass
    if "ADK_ALLOW_WIP_FEATURES" in os.environ:
      del os.environ["ADK_ALLOW_WIP_FEATURES"]


def test_experimental_function_warns(monkeypatch):
  """Test that experimental function shows warnings (unchanged behavior)."""
  # Ensure environment variable is not set
  monkeypatch.delenv(
      "ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", raising=False
  )
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    result = experimental_fn()

    assert result == "executing"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[EXPERIMENTAL] experimental_fn:" in str(w[0].message)
    assert "breaking change in the future" in str(w[0].message)


def test_experimental_class_warns(monkeypatch):
  """Test that experimental class shows warnings (unchanged behavior)."""
  # Ensure environment variable is not set
  monkeypatch.delenv(
      "ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", raising=False
  )
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    exp_class = ExperimentalClass()
    result = exp_class.run()

    assert result == "running experimental"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[EXPERIMENTAL] ExperimentalClass:" in str(w[0].message)
    assert "class may change" in str(w[0].message)


def test_experimental_function_bypassed_with_env_var(monkeypatch):
  """Experimental function emits no warning when bypass env var is true."""
  true_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
  for true_val in true_values:
    monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", true_val)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      result = experimental_fn()
      assert result == "executing"
      assert len(w) == 0, f"Bypass failed for env value {true_val}"


def test_experimental_class_bypassed_with_env_var(monkeypatch):
  """Experimental class emits no warning when bypass env var is true."""
  true_values = ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]
  for true_val in true_values:
    monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", true_val)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      exp_class = ExperimentalClass()
      result = exp_class.run()
      assert result == "running experimental"
      assert len(w) == 0, f"Bypass failed for env value {true_val}"


def test_experimental_function_not_bypassed_for_false_env_var(monkeypatch):
  """Experimental function still warns for non-true bypass env var values."""
  false_values = ["false", "False", "FALSE", "0", "", "no", "off"]
  for false_val in false_values:
    monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", false_val)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      experimental_fn()
      assert len(w) == 1
      assert "[EXPERIMENTAL] experimental_fn:" in str(w[0].message)


def test_experimental_class_not_bypassed_for_false_env_var(monkeypatch):
  """Experimental class still warns for non-true bypass env var values."""
  false_values = ["false", "False", "FALSE", "0", "", "no", "off"]
  for false_val in false_values:
    monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", false_val)
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      ExperimentalClass()
      assert len(w) == 1
      assert "[EXPERIMENTAL] ExperimentalClass:" in str(w[0].message)


def test_experimental_class_no_parens_warns():
  """Test that experimental class without parentheses shows default warning."""
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    exp_class = ExperimentalClassNoParens()
    result = exp_class.run()

    assert result == "running experimental without parens"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[EXPERIMENTAL] ExperimentalClassNoParens:" in str(w[0].message)
    assert "This feature is experimental and may change or be removed" in str(
        w[0].message
    )


def test_experimental_class_empty_parens_warns():
  """Test that experimental class with empty parentheses shows default warning."""
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    exp_class = ExperimentalClassEmptyParens()
    result = exp_class.run()

    assert result == "running experimental with empty parens"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[EXPERIMENTAL] ExperimentalClassEmptyParens:" in str(w[0].message)
    assert "This feature is experimental and may change or be removed" in str(
        w[0].message
    )


def test_experimental_function_no_parens_warns():
  """Test that experimental function without parentheses shows default warning."""
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    result = experimental_fn_no_parens()

    assert result == "executing without parens"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[EXPERIMENTAL] experimental_fn_no_parens:" in str(w[0].message)
    assert "This feature is experimental and may change or be removed" in str(
        w[0].message
    )


def test_experimental_function_empty_parens_warns():
  """Test that experimental function with empty parentheses shows default warning."""
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    result = experimental_fn_empty_parens()

    assert result == "executing with empty parens"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[EXPERIMENTAL] experimental_fn_empty_parens:" in str(w[0].message)
    assert "This feature is experimental and may change or be removed" in str(
        w[0].message
    )
