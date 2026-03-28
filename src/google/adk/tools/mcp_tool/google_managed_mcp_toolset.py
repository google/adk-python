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

from __future__ import annotations

import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TextIO
from typing import Union
import sys

from mcp import SamplingCapability
from mcp.client.session import SamplingFnT
from mcp.shared.session import ProgressFnT

from ...agents.readonly_context import ReadonlyContext
from ..base_toolset import ToolPredicate
from .mcp_session_manager import StreamableHTTPConnectionParams
from .mcp_tool import ProgressCallbackFactory
from .mcp_toolset import McpToolset

logger = logging.getLogger("google_adk." + __name__)


# Maps product names to (endpoint_url, default_scopes).
# New Google Managed MCP products can be added here as they become available.
_GOOGLE_MCP_PRODUCTS: Dict[str, tuple[str, list[str]]] = {
    "bigquery": (
        "https://bigquery.googleapis.com/mcp",
        ["https://www.googleapis.com/auth/bigquery"],
    ),
}


def _create_adc_header_provider(
    scopes: list[str],
    project_id: Optional[str] = None,
) -> Callable[[ReadonlyContext], Dict[str, str]]:
  """Creates a header_provider that returns fresh ADC Bearer tokens.

  The returned callable refreshes the credential on every call. The
  ``google-auth`` library internally caches the token and only contacts
  the metadata server when the token is actually expired, so calling
  ``credentials.refresh()`` on every request is both safe and efficient.

  Args:
    scopes: OAuth2 scopes to request when obtaining credentials.
    project_id: Optional Google Cloud project ID. When provided it is
      sent as the ``x-goog-user-project`` header for quota attribution.

  Returns:
    A callable suitable for the ``header_provider`` parameter of
    :class:`McpToolset`.
  """

  try:
    import google.auth
    import google.auth.transport.requests
  except ImportError as e:
    raise ImportError(
        "google-auth is required for GoogleManagedMcpToolset. "
        "Install it with: pip install google-auth"
    ) from e

  credentials, default_project = google.auth.default(scopes=scopes)
  auth_request = google.auth.transport.requests.Request()

  def _provide_headers(readonly_context: ReadonlyContext) -> Dict[str, str]:
    """Returns Authorization and optional project headers."""
    credentials.refresh(auth_request)
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {credentials.token}",
    }
    resolved_project = project_id or default_project
    if resolved_project:
      headers["x-goog-user-project"] = resolved_project
    return headers

  return _provide_headers


class GoogleManagedMcpToolset(McpToolset):
  """McpToolset subclass for connecting to Google Managed MCP servers.

  This toolset simplifies authentication and connection setup by:

  * Automatically resolving the MCP server URL for a Google product.
  * Using Application Default Credentials (ADC) with automatic token
    refresh — no manual token management required.

  The developer only needs to specify the product name (e.g.
  ``"bigquery"``) and, optionally, a project ID and custom scopes.
  All other ``McpToolset`` parameters (``tool_filter``,
  ``tool_name_prefix``, etc.) are forwarded to the parent class.

  Supported products:

  * ``"bigquery"`` — Google BigQuery MCP server.

  Usage::

    # Minimal — uses ADC and default BigQuery scopes
    toolset = GoogleManagedMcpToolset(product="bigquery")

    # With custom project and scopes
    toolset = GoogleManagedMcpToolset(
        product="bigquery",
        project_id="my-project",
        scopes=["https://www.googleapis.com/auth/bigquery.readonly"],
        tool_filter=["list_datasets", "execute_sql"],
    )

    agent = LlmAgent(
        model="gemini-2.5-flash",
        name="bq_agent",
        instruction="Help users query BigQuery.",
        tools=[toolset],
    )
  """

  def __init__(
      self,
      *,
      product: str,
      project_id: Optional[str] = None,
      scopes: Optional[list[str]] = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      tool_name_prefix: Optional[str] = None,
      errlog: TextIO = sys.stderr,
      require_confirmation: Union[bool, Callable[..., bool]] = False,
      progress_callback: Optional[
          Union[ProgressFnT, ProgressCallbackFactory]
      ] = None,
      use_mcp_resources: Optional[bool] = False,
      sampling_callback: Optional[SamplingFnT] = None,
      sampling_capabilities: Optional[SamplingCapability] = None,
  ):
    """Initializes the GoogleManagedMcpToolset.

    Args:
      product: The Google product to connect to. Must be one of the
        supported product names (e.g. ``"bigquery"``). Case-insensitive.
      project_id: Optional Google Cloud project ID for quota attribution.
        If not provided, the project associated with the Application
        Default Credentials is used.
      scopes: Optional list of OAuth2 scopes. If not provided, product-
        specific default scopes are used.
      tool_filter: Optional filter to select specific tools. Can be either
        a list of tool names to include or a ``ToolPredicate`` function
        for custom filtering logic.
      tool_name_prefix: A prefix to be added to the name of each tool in
        this toolset.
      errlog: TextIO stream for error logging.
      require_confirmation: Whether tools in this toolset require
        confirmation. Can be a single boolean or a callable to apply to
        all tools.
      progress_callback: Optional callback to receive progress
        notifications from the MCP server during long-running tool
        execution.
      use_mcp_resources: Whether the agent should have access to MCP
        resources.
      sampling_callback: Optional callback to handle sampling requests
        from the MCP server.
      sampling_capabilities: Optional capabilities for sampling.

    Raises:
      ValueError: If the product name is not recognized.
      ImportError: If ``google-auth`` is not installed.
    """
    product_key = product.lower()
    if product_key not in _GOOGLE_MCP_PRODUCTS:
      supported = ", ".join(sorted(_GOOGLE_MCP_PRODUCTS.keys()))
      raise ValueError(
          f"Unknown Google MCP product: '{product}'. "
          f"Supported products: [{supported}]"
      )

    endpoint_url, default_scopes = _GOOGLE_MCP_PRODUCTS[product_key]
    resolved_scopes = scopes or default_scopes

    header_provider = _create_adc_header_provider(
        scopes=resolved_scopes,
        project_id=project_id,
    )

    connection_params = StreamableHTTPConnectionParams(url=endpoint_url)

    super().__init__(
        connection_params=connection_params,
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
        errlog=errlog,
        require_confirmation=require_confirmation,
        header_provider=header_provider,
        progress_callback=progress_callback,
        use_mcp_resources=use_mcp_resources,
        sampling_callback=sampling_callback,
        sampling_capabilities=sampling_capabilities,
    )
