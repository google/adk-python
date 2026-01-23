"""
PDF file parser implementation.

Extracts text and tables from PDF files using pdfplumber or pypdf.
"""

from io import BytesIO
from typing import Any

from adk_rlm.files.base import LoadedFile
from adk_rlm.files.base import ParsedContent
from adk_rlm.files.parsers.base import FileParser


class PDFParser(FileParser):
  """
  Parse PDF files.

  Uses pdfplumber for better table extraction when available,
  falls back to pypdf for basic text extraction.

  Example:
      ```python
      parser = PDFParser()
      if parser.can_parse(loaded_file):
          content = parser.parse(loaded_file)
          print(content.text)
          if content.tables:
              print(f"Found {len(content.tables)} table rows")
      ```
  """

  @property
  def supported_extensions(self) -> list[str]:
    """Return list of supported file extensions."""
    return [".pdf"]

  @property
  def supported_mime_types(self) -> list[str]:
    """Return list of supported MIME types."""
    return ["application/pdf"]

  def parse(self, file: LoadedFile) -> ParsedContent:
    """
    Extract text from PDF.

    Tries pdfplumber first (better for tables), falls back to pypdf.

    Args:
        file: LoadedFile with PDF content

    Returns:
        ParsedContent with text, optional tables, and page chunks
    """
    # Try pdfplumber first (better table extraction)
    try:
      import pdfplumber

      return self._parse_with_pdfplumber(file)
    except ImportError:
      pass

    # Fall back to pypdf
    try:
      import pypdf

      return self._parse_with_pypdf(file)
    except ImportError:
      pass

    # Neither library available
    raise ImportError(
        "PDF parsing requires either 'pdfplumber' or 'pypdf'. "
        "Install with: pip install pdfplumber  or  pip install pypdf"
    )

  def _parse_with_pdfplumber(self, file: LoadedFile) -> ParsedContent:
    """
    Parse PDF using pdfplumber (better for tables).

    Args:
        file: LoadedFile with PDF content

    Returns:
        ParsedContent with text, tables, and page chunks
    """
    import pdfplumber

    text_parts: list[str] = []
    tables: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"parser": "pdfplumber"}

    with pdfplumber.open(BytesIO(file.content)) as pdf:
      metadata["page_count"] = len(pdf.pages)

      for i, page in enumerate(pdf.pages):
        # Extract text
        page_text = page.extract_text() or ""
        text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

        # Extract tables
        page_tables = page.extract_tables()
        for table in page_tables:
          if table and len(table) > 1:
            # Use first row as headers
            headers = [
                str(h) if h else f"col_{j}" for j, h in enumerate(table[0])
            ]
            for row in table[1:]:
              if row:
                tables.append(dict(zip(headers, row)))

    metadata["table_count"] = len(tables)

    return ParsedContent(
        text="\n\n".join(text_parts),
        metadata=metadata,
        chunks=text_parts,  # Pre-chunked by page
        tables=tables if tables else None,
        images=None,
    )

  def _parse_with_pypdf(self, file: LoadedFile) -> ParsedContent:
    """
    Parse PDF using pypdf (simpler, no table extraction).

    Args:
        file: LoadedFile with PDF content

    Returns:
        ParsedContent with text and page chunks
    """
    import pypdf

    text_parts: list[str] = []
    metadata: dict[str, Any] = {"parser": "pypdf"}

    reader = pypdf.PdfReader(BytesIO(file.content))
    metadata["page_count"] = len(reader.pages)

    for i, page in enumerate(reader.pages):
      page_text = page.extract_text() or ""
      text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

    return ParsedContent(
        text="\n\n".join(text_parts),
        metadata=metadata,
        chunks=text_parts,
        tables=None,
        images=None,
    )
