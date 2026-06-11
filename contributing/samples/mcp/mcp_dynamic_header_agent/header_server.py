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

from fastapi import Request
from mcp.server.fastmcp import Context
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Header Check Server', host='localhost', port=3000)

TENANT_DATA = {
    'tenant1': {'name': 'Tenant 1', 'data': 'Data for tenant 1'},
    'tenant2': {'name': 'Tenant 2', 'data': 'Data for tenant 2'},
}


@mcp.tool(
    description='Returns tenant specific data based on X-Tenant-ID header.'
)
def get_tenant_data(context: Context) -> dict:
  """Return tenant specific data."""
  if context.request_context and context.request_context.request:
    headers = context.request_context.request.headers
    tenant_id = headers.get('x-tenant-id')
    if tenant_id in TENANT_DATA:
      return TENANT_DATA[tenant_id]
    else:
      return {'error': f'Tenant {tenant_id} not found'}
  else:
    return {'error': 'Could not get request context'}


if __name__ == '__main__':
  try:
    print('Starting Header Check MCP server on http://localhost:3000')
    mcp.run(transport='streamable-http')
  except KeyboardInterrupt:
    print('\nServer stopped.')
