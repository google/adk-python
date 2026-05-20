from __future__ import annotations

import warnings

from google.adk.integrations.bigquery.config import *

warnings.warn(
    "google.adk.tools.bigquery.config is moved to"
    " google.adk.integrations.bigquery.config",
    DeprecationWarning,
    stacklevel=2,
)
