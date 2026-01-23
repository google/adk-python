"""
Tests for context passing to child agents.

This module tests the functionality of passing context objects (including
LazyFile and LazyFileCollection) to child agents via llm_query and llm_query_batched.
"""

from pathlib import Path
from typing import Any

from adk_rlm.files.lazy import LazyFile
from adk_rlm.files.lazy import LazyFileCollection
from adk_rlm.files.loader import FileLoader
from adk_rlm.repl.local_repl import LocalREPL
from adk_rlm.types import QueryMetadata
import pytest


class TestQueryMetadataWithLazyFiles:
  """Tests for QueryMetadata handling of LazyFile types."""

  @pytest.fixture
  def sample_lazy_file(self, tmp_path):
    """Create a sample LazyFile for testing."""
    subdir = tmp_path / "lazy_file_test"
    subdir.mkdir()
    test_file = subdir / "test.txt"
    test_file.write_text("This is test content with some text.")
    loader = FileLoader()
    return loader.create_lazy_file(str(test_file))

  @pytest.fixture
  def sample_lazy_collection(self, tmp_path):
    """Create a sample LazyFileCollection for testing."""
    subdir = tmp_path / "lazy_collection_test"
    subdir.mkdir()
    file1 = subdir / "file1.txt"
    file2 = subdir / "file2.txt"
    file3 = subdir / "file3.txt"
    file1.write_text("Content of file 1")
    file2.write_text("Content of file 2 with more text")
    file3.write_text("Short")
    loader = FileLoader()
    # Use explicit file list instead of glob to avoid cwd issues
    return loader.create_lazy_files([str(file1), str(file2), str(file3)])

  def test_query_metadata_with_string(self):
    """QueryMetadata handles string context."""
    metadata = QueryMetadata("This is a simple string context")
    assert metadata.context_type == "str"
    assert len(metadata.context_lengths) == 1
    assert metadata.context_total_length == 31

  def test_query_metadata_with_dict(self):
    """QueryMetadata handles dict context."""
    metadata = QueryMetadata({"key1": "value1", "key2": "value2"})
    assert metadata.context_type == "dict"
    assert len(metadata.context_lengths) == 2

  def test_query_metadata_with_list(self):
    """QueryMetadata handles list context."""
    metadata = QueryMetadata(["item1", "item2", "item3"])
    assert metadata.context_type == "list"
    assert len(metadata.context_lengths) == 3

  def test_query_metadata_with_lazy_file(self, sample_lazy_file):
    """QueryMetadata handles LazyFile context."""
    metadata = QueryMetadata(sample_lazy_file)
    assert metadata.context_type == "lazy_file"
    assert len(metadata.context_lengths) == 1
    # size_bytes may be 0 before file is loaded (lazy loading)
    assert metadata.context_total_length >= 0

  def test_query_metadata_with_lazy_collection(self, sample_lazy_collection):
    """QueryMetadata handles LazyFileCollection context."""
    metadata = QueryMetadata(sample_lazy_collection)
    assert metadata.context_type == "lazy_file_collection"
    assert len(metadata.context_lengths) == 3
    # size_bytes may be 0 before files are loaded (lazy loading)
    assert metadata.context_total_length >= 0

  def test_query_metadata_with_empty_list(self):
    """QueryMetadata handles empty list."""
    metadata = QueryMetadata([])
    assert metadata.context_type == "list"
    assert metadata.context_lengths == [0]


class TestREPLContextLoading:
  """Tests for REPL context loading with various types."""

  @pytest.fixture
  def sample_lazy_file(self, tmp_path):
    """Create a sample LazyFile for testing."""
    subdir = tmp_path / "repl_lazy_file_test"
    subdir.mkdir()
    test_file = subdir / "test.txt"
    test_file.write_text("This is test content.")
    loader = FileLoader()
    return loader.create_lazy_file(str(test_file))

  @pytest.fixture
  def sample_lazy_collection(self, tmp_path):
    """Create a sample LazyFileCollection for testing."""
    subdir = tmp_path / "repl_lazy_collection_test"
    subdir.mkdir()
    file_a = subdir / "a.txt"
    file_b = subdir / "b.txt"
    file_a.write_text("Content A")
    file_b.write_text("Content B")
    loader = FileLoader()
    # Use explicit file list instead of glob to avoid cwd issues
    return loader.create_lazy_files([str(file_a), str(file_b)])

  def test_load_string_context(self):
    """REPL loads string context correctly."""
    repl = LocalREPL()
    repl.load_context("test string context")

    result = repl.execute_code("print(context)")
    assert "test string context" in result.stdout

  def test_load_dict_context(self):
    """REPL loads dict context correctly."""
    repl = LocalREPL()
    repl.load_context({"key": "value", "number": 42})

    result = repl.execute_code("print(context['key'])")
    assert "value" in result.stdout

  def test_load_lazy_file_context(self, sample_lazy_file):
    """REPL loads LazyFile context correctly."""
    repl = LocalREPL()
    repl.load_context(sample_lazy_file)

    # Check that context is a LazyFile
    result = repl.execute_code("print(type(context).__name__)")
    assert "LazyFile" in result.stdout

    # Check that we can access file properties
    result = repl.execute_code("print(context.name)")
    assert "test.txt" in result.stdout

  def test_load_lazy_collection_context(self, sample_lazy_collection):
    """REPL loads LazyFileCollection context correctly."""
    repl = LocalREPL()
    repl.load_context(sample_lazy_collection)

    # Check that context is a LazyFileCollection
    result = repl.execute_code("print(type(context).__name__)")
    assert "LazyFileCollection" in result.stdout

    # Check that we can iterate over files
    result = repl.execute_code("print(len(list(context)))")
    assert "2" in result.stdout


class TestLLMQueryContextPassing:
  """Tests for llm_query context parameter."""

  @pytest.fixture
  def tracking_llm_query(self):
    """Create an llm_query function that tracks calls."""
    calls = []

    def _llm_query(
        prompt: str,
        context: Any = None,
        model: str | None = None,
        recursive: bool = True,
    ) -> str:
      calls.append({
          "prompt": prompt,
          "context": context,
          "model": model,
          "recursive": recursive,
      })
      return f"Response for: {prompt[:30]}..."

    _llm_query.calls = calls
    return _llm_query

  def test_llm_query_without_context(self, tracking_llm_query):
    """llm_query works without context."""
    repl = LocalREPL(llm_query_fn=tracking_llm_query)
    result = repl.execute_code(
        "response = llm_query('What is 2+2?')\nprint(response)"
    )

    assert len(tracking_llm_query.calls) == 1
    assert tracking_llm_query.calls[0]["prompt"] == "What is 2+2?"
    assert tracking_llm_query.calls[0]["context"] is None
    assert "Response for:" in result.stdout

  def test_llm_query_with_string_context(self, tracking_llm_query):
    """llm_query passes string context correctly."""
    repl = LocalREPL(llm_query_fn=tracking_llm_query)
    result = repl.execute_code(
        "response = llm_query('Summarize this', context='Some text to"
        " summarize')\nprint(response)"
    )

    assert len(tracking_llm_query.calls) == 1
    assert tracking_llm_query.calls[0]["context"] == "Some text to summarize"

  def test_llm_query_with_dict_context(self, tracking_llm_query):
    """llm_query passes dict context correctly."""
    repl = LocalREPL(llm_query_fn=tracking_llm_query)
    result = repl.execute_code(
        "ctx = {'data': 'important info'}\n"
        "response = llm_query('Analyze this', context=ctx)\n"
        "print(response)"
    )

    assert len(tracking_llm_query.calls) == 1
    assert tracking_llm_query.calls[0]["context"] == {"data": "important info"}

  def test_llm_query_with_lazy_file_context(self, tracking_llm_query, tmp_path):
    """llm_query passes LazyFile context correctly."""
    # Create a test file in isolated subdir
    subdir = tmp_path / "llm_query_lazy_test"
    subdir.mkdir()
    test_file = subdir / "data.txt"
    test_file.write_text("File content here")

    loader = FileLoader()
    lazy_file = loader.create_lazy_file(str(test_file))

    repl = LocalREPL(llm_query_fn=tracking_llm_query)
    repl.load_context(lazy_file)

    result = repl.execute_code(
        "response = llm_query('Summarize the file', context=context)\n"
        "print(response)"
    )

    assert len(tracking_llm_query.calls) == 1
    assert isinstance(tracking_llm_query.calls[0]["context"], LazyFile)

  def test_llm_query_with_model_override(self, tracking_llm_query):
    """llm_query passes model parameter correctly."""
    repl = LocalREPL(llm_query_fn=tracking_llm_query)
    result = repl.execute_code(
        "response = llm_query('Test', model='custom-model')\nprint(response)"
    )

    assert tracking_llm_query.calls[0]["model"] == "custom-model"

  def test_llm_query_with_recursive_false(self, tracking_llm_query):
    """llm_query passes recursive=False correctly."""
    repl = LocalREPL(llm_query_fn=tracking_llm_query)
    result = repl.execute_code(
        "response = llm_query('Test', recursive=False)\nprint(response)"
    )

    assert tracking_llm_query.calls[0]["recursive"] is False


class TestLLMQueryBatchedContextPassing:
  """Tests for llm_query_batched contexts parameter."""

  @pytest.fixture
  def tracking_llm_query_batched(self):
    """Create an llm_query_batched function that tracks calls."""
    calls = []

    def _llm_query_batched(
        prompts: list[str],
        contexts: list[Any] | None = None,
        model: str | None = None,
        recursive: bool = False,
    ) -> list[str]:
      calls.append({
          "prompts": prompts,
          "contexts": contexts,
          "model": model,
          "recursive": recursive,
      })
      return [f"Response {i}" for i in range(len(prompts))]

    _llm_query_batched.calls = calls
    return _llm_query_batched

  @pytest.fixture
  def tracking_llm_query(self):
    """Create an llm_query function for fallback."""

    def _llm_query(prompt, context=None, model=None, recursive=True):
      return f"Response for: {prompt[:20]}..."

    return _llm_query

  def test_batched_without_contexts(
      self, tracking_llm_query, tracking_llm_query_batched
  ):
    """llm_query_batched works without contexts."""
    repl = LocalREPL(
        llm_query_fn=tracking_llm_query,
        llm_query_batched_fn=tracking_llm_query_batched,
    )
    result = repl.execute_code(
        "prompts = ['Q1', 'Q2', 'Q3']\n"
        "responses = llm_query_batched(prompts)\n"
        "print(len(responses))"
    )

    assert len(tracking_llm_query_batched.calls) == 1
    assert tracking_llm_query_batched.calls[0]["contexts"] is None
    assert "3" in result.stdout

  def test_batched_with_contexts(
      self, tracking_llm_query, tracking_llm_query_batched
  ):
    """llm_query_batched passes contexts correctly."""
    repl = LocalREPL(
        llm_query_fn=tracking_llm_query,
        llm_query_batched_fn=tracking_llm_query_batched,
    )
    result = repl.execute_code(
        "prompts = ['Summarize A', 'Summarize B']\n"
        "contexts = ['Content A', 'Content B']\n"
        "responses = llm_query_batched(prompts, contexts=contexts)\n"
        "print(len(responses))"
    )

    assert len(tracking_llm_query_batched.calls) == 1
    assert tracking_llm_query_batched.calls[0]["contexts"] == [
        "Content A",
        "Content B",
    ]

  def test_batched_with_lazy_files(
      self, tracking_llm_query, tracking_llm_query_batched, tmp_path
  ):
    """llm_query_batched passes LazyFile contexts correctly."""
    # Create test files in isolated subdir
    subdir = tmp_path / "batched_lazy_test"
    subdir.mkdir()
    file1 = subdir / "file1.txt"
    file2 = subdir / "file2.txt"
    file1.write_text("Content 1")
    file2.write_text("Content 2")

    loader = FileLoader()
    # Use explicit file list instead of glob to avoid cwd issues
    files = loader.create_lazy_files([str(file1), str(file2)])
    file_list = list(files)

    repl = LocalREPL(
        llm_query_fn=tracking_llm_query,
        llm_query_batched_fn=tracking_llm_query_batched,
    )
    repl.locals["files"] = file_list

    result = repl.execute_code(
        "prompts = [f'Summarize {f.name}' for f in files]\n"
        "responses = llm_query_batched(prompts, contexts=files)\n"
        "print(len(responses))"
    )

    assert len(tracking_llm_query_batched.calls) == 1
    contexts = tracking_llm_query_batched.calls[0]["contexts"]
    assert len(contexts) == 2
    assert all(isinstance(c, LazyFile) for c in contexts)

  def test_batched_with_recursive_true(
      self, tracking_llm_query, tracking_llm_query_batched
  ):
    """llm_query_batched passes recursive=True correctly."""
    repl = LocalREPL(
        llm_query_fn=tracking_llm_query,
        llm_query_batched_fn=tracking_llm_query_batched,
    )
    result = repl.execute_code(
        "responses = llm_query_batched(['Q1', 'Q2'], recursive=True)\n"
        "print(len(responses))"
    )

    assert tracking_llm_query_batched.calls[0]["recursive"] is True

  def test_batched_fallback_without_batched_fn(self, tracking_llm_query):
    """llm_query_batched falls back to individual calls if no batched fn."""
    calls = []

    def tracking_query(prompt, context=None, model=None, recursive=True):
      calls.append({"prompt": prompt, "context": context})
      return f"Response for: {prompt}"

    repl = LocalREPL(llm_query_fn=tracking_query)
    result = repl.execute_code(
        "responses = llm_query_batched(['Q1', 'Q2'])\nprint(len(responses))"
    )

    # Should make 2 individual calls
    assert len(calls) == 2
    assert "2" in result.stdout

  def test_batched_fallback_with_contexts(self, tracking_llm_query):
    """llm_query_batched fallback passes contexts correctly."""
    calls = []

    def tracking_query(prompt, context=None, model=None, recursive=True):
      calls.append({"prompt": prompt, "context": context})
      return f"Response for: {prompt}"

    repl = LocalREPL(llm_query_fn=tracking_query)
    result = repl.execute_code(
        "responses = llm_query_batched(['Q1', 'Q2'], contexts=['C1', 'C2'])\n"
        "print(len(responses))"
    )

    assert len(calls) == 2
    assert calls[0]["context"] == "C1"
    assert calls[1]["context"] == "C2"


class TestCodeExecutorContextPassing:
  """Tests for RLMCodeExecutor context passing to child agents."""

  @pytest.fixture
  def sample_lazy_file(self, tmp_path):
    """Create a sample LazyFile for testing."""
    subdir = tmp_path / "executor_lazy_test"
    subdir.mkdir()
    test_file = subdir / "test.txt"
    test_file.write_text("Test file content for child agent.")
    loader = FileLoader()
    return loader.create_lazy_file(str(test_file))

  def test_executor_llm_query_signature(self):
    """Verify RLMCodeExecutor creates llm_query with correct signature."""
    import inspect

    from adk_rlm.code_executor import RLMCodeExecutor

    executor = RLMCodeExecutor()
    llm_query_fn = executor._create_llm_query_fn()

    sig = inspect.signature(llm_query_fn)
    params = list(sig.parameters.keys())

    assert "prompt" in params
    assert "context" in params
    assert "model" in params
    assert "recursive" in params

  def test_executor_llm_query_batched_signature(self):
    """Verify RLMCodeExecutor creates llm_query_batched with correct signature."""
    import inspect

    from adk_rlm.code_executor import RLMCodeExecutor

    executor = RLMCodeExecutor()
    llm_query_batched_fn = executor._create_llm_query_batched_fn()

    sig = inspect.signature(llm_query_batched_fn)
    params = list(sig.parameters.keys())

    assert "prompts" in params
    assert "contexts" in params
    assert "model" in params
    assert "recursive" in params
