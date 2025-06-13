# Copyright 2025 Google LLC
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

"""Utils package for ADK."""

from google.adk.utils.adk_to_mcp_tool_type import adk_to_mcp_tool_type
from google.adk.utils.async_utils import gather_results
from google.adk.utils.date_utils import (
    from_rfc3339_datetime,
    to_rfc3339_datetime,
)
from google.adk.utils.image_utils import get_base64_image_from_uri
from google.adk.utils.langgraph_utils import (
    LangGraphContextManager,
    create_reference_aware_merge,
)
from google.adk.utils.multipart_utils import (
    create_multipart_message,
    extract_boundary_from_content_type,
    parse_multipart_message,
)
from google.adk.utils.structured_output_utils import (
    input_or_function_schema_to_signature,
    to_function_schema,
    typescript_schema_to_pydantic,
)
from google.adk.utils.truncate_utils import truncate_data
from google.adk.utils.uri_utils import uri_to_file_path

__all__ = [
    "adk_to_mcp_tool_type",
    "create_multipart_message",
    "extract_boundary_from_content_type",
    "from_rfc3339_datetime",
    "gather_results",
    "get_base64_image_from_uri",
    "input_or_function_schema_to_signature",
    "LangGraphContextManager",
    "create_reference_aware_merge",
    "parse_multipart_message",
    "to_function_schema",
    "to_rfc3339_datetime",
    "truncate_data",
    "typescript_schema_to_pydantic",
    "uri_to_file_path",
]
