import warnings

from google.adk.utils.feature_decorator import experimental
from google.adk.utils.feature_decorator import working_in_progress


@working_in_progress("in complete feature, don't use yet")
class IncompleteFeature:

  def run(self):
    return "running"


@experimental("api may have breaking change in the future.")
def experimental_fn():
  return "executing"


def test_working_in_progress_class_warns():
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    feature = IncompleteFeature()

    assert feature.run() == "running"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[WIP] IncompleteFeature:" in str(w[0].message)
    assert "don't use yet" in str(w[0].message)


def test_experimental_method_warns():
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    result = experimental_fn()

    assert result == "executing"
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "[EXPERIMENTAL] experimental_fn:" in str(w[0].message)
    assert "breaking change in the future" in str(w[0].message)
