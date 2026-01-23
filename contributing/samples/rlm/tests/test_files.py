"""
Tests for the file handling module.

Tests cover:
- LocalFileSource loading and glob patterns
- TextParser for various text formats
- PDFParser (when pdfplumber is available)
- LazyFile progressive disclosure
- LazyFileCollection filtering
- FileLoader orchestration
"""

import json
import os
from pathlib import Path
import tempfile

from adk_rlm.files import FileLoader
from adk_rlm.files import FileMetadata
from adk_rlm.files import LazyFile
from adk_rlm.files import LazyFileCollection
from adk_rlm.files import LoadedFile
from adk_rlm.files import LocalFileSource
from adk_rlm.files import ParsedContent
from adk_rlm.files import TextParser
import pytest

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
  """Create a temporary directory for test files."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir: Path):
  """Create a sample text file."""
  path = temp_dir / "sample.txt"
  path.write_text("Hello, world!\nThis is a test file.")
  return path


@pytest.fixture
def sample_json_file(temp_dir: Path):
  """Create a sample JSON file."""
  path = temp_dir / "data.json"
  data = {"name": "Test", "values": [1, 2, 3], "nested": {"key": "value"}}
  path.write_text(json.dumps(data))
  return path


@pytest.fixture
def sample_csv_file(temp_dir: Path):
  """Create a sample CSV file."""
  path = temp_dir / "data.csv"
  path.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago")
  return path


@pytest.fixture
def sample_markdown_file(temp_dir: Path):
  """Create a sample Markdown file."""
  path = temp_dir / "readme.md"
  path.write_text(
      "# Title\n\nThis is a **markdown** file.\n\n- Item 1\n- Item 2"
  )
  return path


@pytest.fixture
def multiple_files(temp_dir: Path):
  """Create multiple files of various types."""
  files = {}

  # Text files
  (temp_dir / "doc1.txt").write_text("Document 1 content")
  (temp_dir / "doc2.txt").write_text("Document 2 content")
  files["txt"] = [temp_dir / "doc1.txt", temp_dir / "doc2.txt"]

  # Markdown files
  (temp_dir / "readme.md").write_text("# README\n\nProject documentation")
  (temp_dir / "notes.md").write_text("# Notes\n\nImportant notes")
  files["md"] = [temp_dir / "readme.md", temp_dir / "notes.md"]

  # JSON files
  (temp_dir / "config.json").write_text('{"setting": "value"}')
  files["json"] = [temp_dir / "config.json"]

  # Subdirectory with files
  subdir = temp_dir / "subdir"
  subdir.mkdir()
  (subdir / "nested.txt").write_text("Nested file content")
  files["nested"] = [subdir / "nested.txt"]

  return files


# ============================================================================
# LocalFileSource Tests
# ============================================================================


class TestLocalFileSource:
  """Tests for LocalFileSource."""

  def test_source_type(self):
    """Test source type identifier."""
    source = LocalFileSource()
    assert source.source_type == "local"

  def test_load_text_file(self, sample_text_file: Path):
    """Test loading a text file."""
    source = LocalFileSource()
    loaded = source.load(str(sample_text_file))

    assert isinstance(loaded, LoadedFile)
    assert loaded.metadata.name == "sample.txt"
    assert loaded.as_text() == "Hello, world!\nThis is a test file."
    assert loaded.metadata.size_bytes > 0
    assert loaded.metadata.source_type == "local"

  def test_load_with_base_path(self, temp_dir: Path, sample_text_file: Path):
    """Test loading with base path."""
    source = LocalFileSource(base_path=temp_dir)
    loaded = source.load("sample.txt")

    assert loaded.metadata.name == "sample.txt"
    assert loaded.as_text() == "Hello, world!\nThis is a test file."

  def test_resolve_single_file(self, sample_text_file: Path):
    """Test resolving a single file path."""
    source = LocalFileSource()
    paths = source.resolve(str(sample_text_file))

    assert len(paths) == 1
    assert paths[0] == str(sample_text_file)

  def test_resolve_glob_pattern(self, temp_dir: Path, multiple_files):
    """Test resolving glob patterns."""
    source = LocalFileSource(base_path=temp_dir)

    # Match all txt files
    paths = source.resolve("*.txt")
    assert len(paths) == 2
    assert all(p.endswith(".txt") for p in paths)

  def test_resolve_recursive_glob(self, temp_dir: Path, multiple_files):
    """Test resolving recursive glob patterns."""
    source = LocalFileSource(base_path=temp_dir)

    # Match all txt files including subdirectories
    paths = source.resolve("**/*.txt")
    assert len(paths) == 3  # 2 in root + 1 in subdir

  def test_get_metadata_efficient(self, sample_text_file: Path):
    """Test getting metadata without loading content."""
    source = LocalFileSource()
    metadata = source.get_metadata(str(sample_text_file))

    assert isinstance(metadata, FileMetadata)
    assert metadata.name == "sample.txt"
    assert metadata.size_bytes > 0
    assert metadata.last_modified is not None

  def test_exists_true(self, sample_text_file: Path):
    """Test exists returns True for existing file."""
    source = LocalFileSource()
    assert source.exists(str(sample_text_file)) is True

  def test_exists_false(self, temp_dir: Path):
    """Test exists returns False for non-existing file."""
    source = LocalFileSource()
    assert source.exists(str(temp_dir / "nonexistent.txt")) is False

  def test_load_nonexistent_raises(self, temp_dir: Path):
    """Test loading non-existent file raises error."""
    source = LocalFileSource()
    with pytest.raises(FileNotFoundError):
      source.load(str(temp_dir / "nonexistent.txt"))


# ============================================================================
# TextParser Tests
# ============================================================================


class TestTextParser:
  """Tests for TextParser."""

  def test_supported_extensions(self):
    """Test that common text extensions are supported."""
    parser = TextParser()
    exts = parser.supported_extensions
    assert ".txt" in exts
    assert ".md" in exts
    assert ".json" in exts
    assert ".csv" in exts
    assert ".py" in exts

  def test_parse_plain_text(self, sample_text_file: Path):
    """Test parsing plain text file."""
    source = LocalFileSource()
    parser = TextParser()

    loaded = source.load(str(sample_text_file))
    assert parser.can_parse(loaded)

    parsed = parser.parse(loaded)
    assert isinstance(parsed, ParsedContent)
    assert "Hello, world!" in parsed.text
    assert parsed.metadata["format"] == ".txt"

  def test_parse_json(self, sample_json_file: Path):
    """Test parsing JSON file."""
    source = LocalFileSource()
    parser = TextParser()

    loaded = source.load(str(sample_json_file))
    parsed = parser.parse(loaded)

    assert "Test" in parsed.text
    assert parsed.metadata["format"] == ".json"
    assert parsed.metadata["json_type"] == "dict"

  def test_parse_csv(self, sample_csv_file: Path):
    """Test parsing CSV file with table extraction."""
    source = LocalFileSource()
    parser = TextParser()

    loaded = source.load(str(sample_csv_file))
    parsed = parser.parse(loaded)

    assert parsed.tables is not None
    assert len(parsed.tables) == 3  # 3 data rows
    assert parsed.tables[0]["name"] == "Alice"
    assert parsed.metadata["row_count"] == 3
    assert "name" in parsed.metadata["columns"]

  def test_parse_markdown(self, sample_markdown_file: Path):
    """Test parsing Markdown file."""
    source = LocalFileSource()
    parser = TextParser()

    loaded = source.load(str(sample_markdown_file))
    parsed = parser.parse(loaded)

    assert "# Title" in parsed.text
    assert "**markdown**" in parsed.text
    assert parsed.metadata["format"] == ".md"


# ============================================================================
# LazyFile Tests
# ============================================================================


class TestLazyFile:
  """Tests for LazyFile progressive disclosure."""

  def test_level_0_no_io(self, sample_text_file: Path):
    """Test Level 0 properties don't trigger I/O."""
    source = LocalFileSource()
    parser = TextParser()
    lazy = LazyFile(path=str(sample_text_file), source=source, parser=parser)

    # Level 0 access - no loading
    assert lazy.name == "sample.txt"
    assert lazy.extension == ".txt"
    assert lazy.is_loaded is False
    assert lazy.level == 0

  def test_level_1_metadata(self, sample_text_file: Path):
    """Test Level 1 metadata access."""
    source = LocalFileSource()
    parser = TextParser()
    lazy = LazyFile(path=str(sample_text_file), source=source, parser=parser)

    # Level 1 access - metadata only
    assert lazy.size > 0
    assert lazy.level == 1
    assert lazy.is_loaded is False  # Full content not loaded
    assert lazy.mime_type == "text/plain"

  def test_level_2_content(self, sample_text_file: Path):
    """Test Level 2 content access."""
    source = LocalFileSource()
    parser = TextParser()
    lazy = LazyFile(path=str(sample_text_file), source=source, parser=parser)

    # Level 2 access - full content
    content = lazy.content
    assert "Hello, world!" in content
    assert lazy.level == 2
    assert lazy.is_loaded is True
    assert lazy.is_parsed is True

  def test_size_properties(self, sample_text_file: Path):
    """Test size conversion properties."""
    source = LocalFileSource()
    parser = TextParser()
    lazy = LazyFile(path=str(sample_text_file), source=source, parser=parser)

    assert lazy.size_kb == lazy.size / 1024
    assert lazy.size_mb == lazy.size / (1024 * 1024)

  def test_preload_metadata(self, sample_text_file: Path):
    """Test preload_metadata method."""
    source = LocalFileSource()
    parser = TextParser()
    lazy = LazyFile(path=str(sample_text_file), source=source, parser=parser)

    result = lazy.preload_metadata()
    assert result is lazy  # Returns self for chaining
    assert lazy.level >= 1

  def test_preload_full(self, sample_text_file: Path):
    """Test preload method."""
    source = LocalFileSource()
    parser = TextParser()
    lazy = LazyFile(path=str(sample_text_file), source=source, parser=parser)

    result = lazy.preload()
    assert result is lazy
    assert lazy.level == 2

  def test_read_method(self, sample_text_file: Path):
    """Test read method for raw text access."""
    source = LocalFileSource()
    parser = TextParser()
    lazy = LazyFile(path=str(sample_text_file), source=source, parser=parser)

    text = lazy.read()
    assert "Hello, world!" in text
    # Note: read() may trigger full load depending on source

  def test_repr(self, sample_text_file: Path):
    """Test string representation."""
    source = LocalFileSource()
    lazy = LazyFile(path=str(sample_text_file), source=source)

    repr_str = repr(lazy)
    assert "LazyFile" in repr_str
    assert "sample.txt" in repr_str
    assert "level=0" in repr_str


# ============================================================================
# LazyFileCollection Tests
# ============================================================================


class TestLazyFileCollection:
  """Tests for LazyFileCollection."""

  def test_empty_collection(self):
    """Test empty collection."""
    collection = LazyFileCollection([])
    assert len(collection) == 0
    assert bool(collection) is False

  def test_names_property(self, temp_dir: Path, multiple_files):
    """Test names property without loading."""
    source = LocalFileSource(base_path=temp_dir)
    parser = TextParser()

    lazy_files = [
        LazyFile(path=str(f), source=source, parser=parser)
        for f in multiple_files["txt"]
    ]
    collection = LazyFileCollection(lazy_files)

    names = collection.names
    assert len(names) == 2
    assert "doc1.txt" in names
    assert "doc2.txt" in names

  def test_by_extension(self, temp_dir: Path, multiple_files):
    """Test filtering by extension."""
    source = LocalFileSource(base_path=temp_dir)
    parser = TextParser()

    all_files = (
        multiple_files["txt"] + multiple_files["md"] + multiple_files["json"]
    )
    lazy_files = [
        LazyFile(path=str(f), source=source, parser=parser) for f in all_files
    ]
    collection = LazyFileCollection(lazy_files)

    # Filter by extension
    txt_files = collection.by_extension(".txt")
    assert len(txt_files) == 2
    assert all(f.extension == ".txt" for f in txt_files)

    md_files = collection.by_extension("md")  # Without leading dot
    assert len(md_files) == 2

  def test_by_name_pattern(self, temp_dir: Path, multiple_files):
    """Test filtering by name pattern."""
    source = LocalFileSource(base_path=temp_dir)
    parser = TextParser()

    all_files = multiple_files["txt"] + multiple_files["md"]
    lazy_files = [
        LazyFile(path=str(f), source=source, parser=parser) for f in all_files
    ]
    collection = LazyFileCollection(lazy_files)

    # Filter by pattern
    docs = collection.by_name("doc*.txt")
    assert len(docs) == 2

    readme = collection.by_name("readme*")
    assert len(readme) == 1

  def test_search(self, temp_dir: Path, multiple_files):
    """Test keyword search."""
    source = LocalFileSource(base_path=temp_dir)
    parser = TextParser()

    all_files = multiple_files["txt"] + multiple_files["md"]
    lazy_files = [
        LazyFile(path=str(f), source=source, parser=parser) for f in all_files
    ]
    collection = LazyFileCollection(lazy_files)

    # Case-insensitive search
    results = collection.search("DOC")
    assert len(results) == 2

  def test_loaded_count(self, temp_dir: Path, multiple_files):
    """Test loaded count tracking."""
    source = LocalFileSource(base_path=temp_dir)
    parser = TextParser()

    lazy_files = [
        LazyFile(path=str(f), source=source, parser=parser)
        for f in multiple_files["txt"]
    ]
    collection = LazyFileCollection(lazy_files)

    assert collection.loaded_count == 0

    # Load first file
    _ = collection[0].content
    assert collection.loaded_count == 1

  def test_extensions_property(self, temp_dir: Path, multiple_files):
    """Test extensions set property."""
    source = LocalFileSource(base_path=temp_dir)
    parser = TextParser()

    all_files = (
        multiple_files["txt"] + multiple_files["md"] + multiple_files["json"]
    )
    lazy_files = [
        LazyFile(path=str(f), source=source, parser=parser) for f in all_files
    ]
    collection = LazyFileCollection(lazy_files)

    extensions = collection.extensions
    assert ".txt" in extensions
    assert ".md" in extensions
    assert ".json" in extensions

  def test_summary(self, temp_dir: Path, multiple_files):
    """Test summary output."""
    source = LocalFileSource(base_path=temp_dir)
    parser = TextParser()

    all_files = multiple_files["txt"] + multiple_files["md"]
    lazy_files = [
        LazyFile(path=str(f), source=source, parser=parser) for f in all_files
    ]
    collection = LazyFileCollection(lazy_files)

    summary = collection.summary()
    assert "LazyFileCollection" in summary
    assert ".txt: 2" in summary
    assert ".md: 2" in summary


# ============================================================================
# FileLoader Tests
# ============================================================================


class TestFileLoader:
  """Tests for FileLoader orchestrator."""

  def test_default_sources_and_parsers(self):
    """Test default configuration."""
    loader = FileLoader()
    assert "local" in loader.sources
    assert len(loader.parsers) >= 2  # At least TextParser and PDFParser

  def test_load_single_file(self, sample_text_file: Path):
    """Test loading a single file."""
    loader = FileLoader()
    parsed = loader.load_single(str(sample_text_file))

    assert isinstance(parsed, ParsedContent)
    assert "Hello, world!" in parsed.text

  def test_load_multiple_files(self, temp_dir: Path, multiple_files):
    """Test loading multiple files."""
    loader = FileLoader(base_path=temp_dir)
    files = ["doc1.txt", "doc2.txt"]
    results = loader.load_files(files)

    assert len(results) == 2
    assert all(isinstance(r, ParsedContent) for r in results)

  def test_load_with_glob(self, temp_dir: Path, multiple_files):
    """Test loading with glob pattern."""
    loader = FileLoader(base_path=temp_dir)
    results = loader.load_files(["*.txt"])

    assert len(results) == 2

  def test_create_lazy_files(self, temp_dir: Path, multiple_files):
    """Test creating lazy file collection."""
    loader = FileLoader(base_path=temp_dir)
    collection = loader.create_lazy_files(["*.txt"])

    assert isinstance(collection, LazyFileCollection)
    assert len(collection) == 2
    assert collection.loaded_count == 0  # Not loaded yet

  def test_create_lazy_file_single(self, sample_text_file: Path):
    """Test creating single lazy file."""
    loader = FileLoader()
    lazy = loader.create_lazy_file(str(sample_text_file))

    assert isinstance(lazy, LazyFile)
    assert lazy.name == "sample.txt"

  def test_build_context_lazy(self, temp_dir: Path, multiple_files):
    """Test building context with lazy loading."""
    loader = FileLoader(base_path=temp_dir)
    context = loader.build_context(["*.txt"], lazy=True)

    assert "files" in context
    assert "file_count" in context
    assert "file_names" in context
    assert isinstance(context["files"], LazyFileCollection)

  def test_build_context_eager(self, temp_dir: Path, multiple_files):
    """Test building context with eager loading."""
    loader = FileLoader(base_path=temp_dir)
    context = loader.build_context(["*.txt"], lazy=False)

    assert "files" in context
    assert "file_count" in context
    # Files are already parsed dicts
    assert isinstance(context["files"], list)

  def test_register_parser(self, sample_text_file: Path):
    """Test registering custom parser."""
    loader = FileLoader()
    initial_count = len(loader.parsers)

    # Register another TextParser (for testing)
    loader.register_parser(TextParser())
    assert len(loader.parsers) == initial_count + 1

  def test_nonexistent_file_raises(self, temp_dir: Path):
    """Test loading non-existent file raises error."""
    loader = FileLoader()
    with pytest.raises(FileNotFoundError):
      loader.load_single(str(temp_dir / "nonexistent.txt"))


# ============================================================================
# FileMetadata Tests
# ============================================================================


class TestFileMetadata:
  """Tests for FileMetadata dataclass."""

  def test_size_properties(self):
    """Test size conversion properties."""
    metadata = FileMetadata(
        name="test.txt",
        path="/path/to/test.txt",
        source_type="local",
        size_bytes=1024 * 1024,  # 1 MB
    )

    assert metadata.size_kb == 1024
    assert metadata.size_mb == 1.0

  def test_extension_property(self):
    """Test extension extraction."""
    metadata = FileMetadata(
        name="document.PDF",
        path="/path/document.PDF",
        source_type="local",
        size_bytes=100,
    )

    assert metadata.extension == ".pdf"  # Lowercase

  def test_to_dict(self):
    """Test serialization to dict."""
    from datetime import datetime

    metadata = FileMetadata(
        name="test.txt",
        path="/path/test.txt",
        source_type="local",
        size_bytes=100,
        mime_type="text/plain",
        last_modified=datetime(2024, 1, 1, 12, 0, 0),
        extra={"key": "value"},
    )

    d = metadata.to_dict()
    assert d["name"] == "test.txt"
    assert d["size_bytes"] == 100
    assert "2024" in d["last_modified"]


# ============================================================================
# ParsedContent Tests
# ============================================================================


class TestParsedContent:
  """Tests for ParsedContent dataclass."""

  def test_has_tables(self):
    """Test has_tables property."""
    content_with_tables = ParsedContent(
        text="data",
        tables=[{"col": "val"}],
    )
    content_without = ParsedContent(text="data")

    assert content_with_tables.has_tables is True
    assert content_without.has_tables is False

  def test_has_chunks(self):
    """Test has_chunks property."""
    content_with_chunks = ParsedContent(
        text="data",
        chunks=["chunk1", "chunk2"],
    )
    content_without = ParsedContent(text="data")

    assert content_with_chunks.has_chunks is True
    assert content_without.has_chunks is False

  def test_counts(self):
    """Test count properties."""
    content = ParsedContent(
        text="data",
        chunks=["a", "b", "c"],
        tables=[{"x": 1}, {"x": 2}],
    )

    assert content.chunk_count == 3
    assert content.table_count == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
  """Integration tests for the file handling system."""

  def test_full_workflow_lazy(self, temp_dir: Path, multiple_files):
    """Test complete workflow with lazy loading."""
    loader = FileLoader(base_path=temp_dir)

    # Create lazy collection
    files = loader.create_lazy_files(["**/*.txt", "**/*.md"])

    # Level 0 - no I/O
    assert len(files) >= 4
    names = files.names
    assert all(isinstance(n, str) for n in names)

    # Filter without loading
    txt_files = files.by_extension(".txt")
    assert len(txt_files) == 3  # Including nested

    # Level 2 - load specific files
    for f in txt_files[:2]:
      content = f.content
      assert len(content) > 0
      assert f.level == 2

    # Check stats
    assert txt_files.loaded_count >= 2

  def test_full_workflow_eager(self, temp_dir: Path, multiple_files):
    """Test complete workflow with eager loading."""
    loader = FileLoader(base_path=temp_dir)

    # Eager load all files
    results = loader.load_files(["*.txt", "*.md"])

    assert len(results) == 4
    assert all(isinstance(r, ParsedContent) for r in results)
    assert all(len(r.text) > 0 for r in results)

  def test_build_context_for_rlm(self, temp_dir: Path, multiple_files):
    """Test building context suitable for RLM consumption."""
    loader = FileLoader(base_path=temp_dir)

    context = loader.build_context(["doc1.txt", "doc2.txt"], lazy=True)

    # Context should have expected structure
    assert "files" in context
    assert "file_count" in context
    assert context["file_count"] == 2

    # Files are lazy - can filter without loading
    files = context["files"]
    assert files.by_extension(".txt")


# ============================================================================
# PDF Parser Tests (conditional on pdfplumber availability)
# ============================================================================


class TestPDFParser:
  """Tests for PDFParser (requires pdfplumber)."""

  @pytest.fixture
  def sample_pdf(self, temp_dir: Path):
    """Create a minimal PDF file for testing."""
    try:
      from reportlab.pdfgen import canvas

      pdf_path = temp_dir / "test.pdf"
      c = canvas.Canvas(str(pdf_path))
      c.drawString(100, 750, "Hello PDF World!")
      c.drawString(100, 700, "This is a test PDF.")
      c.save()
      return pdf_path
    except ImportError:
      pytest.skip("reportlab not installed for PDF generation")

  @pytest.mark.skipif(
      not os.path.exists("/usr/bin/python3"), reason="Test requires PDF library"
  )
  def test_pdf_parser_import(self):
    """Test PDFParser can be imported."""
    from adk_rlm.files import PDFParser

    parser = PDFParser()
    assert parser.supported_extensions == [".pdf"]

  def test_pdf_can_parse(self, temp_dir: Path):
    """Test can_parse identifies PDF files."""
    from adk_rlm.files import PDFParser

    parser = PDFParser()

    # Create fake loaded file with PDF extension
    fake_metadata = FileMetadata(
        name="document.pdf",
        path=str(temp_dir / "document.pdf"),
        source_type="local",
        size_bytes=1000,
        mime_type="application/pdf",
    )
    fake_file = LoadedFile(metadata=fake_metadata, content=b"fake pdf")

    assert parser.can_parse(fake_file) is True
