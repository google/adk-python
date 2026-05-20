from __future__ import annotations

import warnings

from google.adk.integrations.bigquery.client import *

warnings.warn(
    "google.adk.tools.bigquery.client is moved to"
    " google.adk.integrations.bigquery.client",
    DeprecationWarning,
    stacklevel=2,
)
