from __future__ import annotations

import warnings

from google.adk.integrations.bigquery.query_tool import *

warnings.warn(
    "google.adk.tools.bigquery.query_tool is moved to"
    " google.adk.integrations.bigquery.query_tool",
    DeprecationWarning,
    stacklevel=2,
)
