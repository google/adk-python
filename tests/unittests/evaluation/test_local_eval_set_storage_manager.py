import os
import tempfile

from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.local_eval_set_storage_manager import LocalEvalSetStorageManager
import pytest


@pytest.fixture
def temp_agents_dir():
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


def test_init(temp_agents_dir):
  manager = LocalEvalSetStorageManager(temp_agents_dir)
  assert manager._agents_dir == temp_agents_dir


def test_get_eval_set_path(temp_agents_dir):
  manager = LocalEvalSetStorageManager(temp_agents_dir)
  path = manager.get_eval_set_path("test_app", "test_eval_set")
  expected_path = os.path.join(
      temp_agents_dir, "test_app", "test_eval_set.evalset.json"
  )
  assert path == expected_path


def test_list_eval_sets(temp_agents_dir):
  manager = LocalEvalSetStorageManager(temp_agents_dir)
  app_dir = os.path.join(temp_agents_dir, "test_app")
  os.makedirs(app_dir, exist_ok=True)
  with open(os.path.join(app_dir, "eval_set_1.evalset.json"), "w") as f:
    f.write('{"eval_set_id": "eval_set_1", "eval_cases": []}')
  with open(os.path.join(app_dir, "eval_set_2.evalset.json"), "w") as f:
    f.write('{"eval_set_id": "eval_set_2", "eval_cases": []}')
  with open(os.path.join(app_dir, "not_eval_set.txt.json"), "w") as f:
    f.write("ignore this")

  eval_sets = manager.list_eval_sets("test_app")
  assert eval_sets == ["eval_set_1", "eval_set_2"]


def test_list_eval_sets_empty(temp_agents_dir):
  manager = LocalEvalSetStorageManager(temp_agents_dir)
  app_dir = os.path.join(temp_agents_dir, "test_app")
  os.makedirs(app_dir, exist_ok=True)

  eval_sets = manager.list_eval_sets("test_app")
  assert eval_sets == []


def test_save_eval_set(temp_agents_dir):
  manager = LocalEvalSetStorageManager(temp_agents_dir)
  eval_set = EvalSet(eval_set_id="test_eval_set", eval_cases=[])
  path = os.path.join(temp_agents_dir, "test_app", "test_eval_set.evalset.json")
  manager.save_eval_set(path, eval_set)
  with open(path, "r") as f:
    loaded_content = f.read()
  assert loaded_content == eval_set.model_dump_json(indent=2)


def test_load_eval_set(temp_agents_dir):
  manager = LocalEvalSetStorageManager(temp_agents_dir)
  eval_set_data = '{"eval_set_id": "test_eval_set", "eval_cases": []}'
  path = os.path.join(temp_agents_dir, "test_app", "test_eval_set.evalset.json")
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    f.write(eval_set_data)
  loaded_eval_set = manager.load_eval_set(path)
  assert loaded_eval_set == EvalSet.model_validate_json(eval_set_data)


def test_load_eval_set_not_found(temp_agents_dir):
  manager = LocalEvalSetStorageManager(temp_agents_dir)
  path = os.path.join(temp_agents_dir, "test_app", "test_eval_set.evalset.json")
  loaded_eval_set = manager.load_eval_set(path)
  assert loaded_eval_set is None
