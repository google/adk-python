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


import asyncio
import os
import pathlib
import ssl
import sys
import tempfile

import google.auth.transport.mtls as google_mtls
from mcp.server.fastmcp import FastMCP
import uvicorn

# Create an MCP server with a name
mcp = FastMCP("Filesystem Server (mTLS)", host="localhost", port=3000)


# Add a tool to read file contents
@mcp.tool(description="Read contents of a file")
def read_file(filepath: str) -> str:
  """Read and return the contents of a file."""
  with open(filepath, "r") as f:
    return f.read()


# Add a tool to list directory contents
@mcp.tool(description="List contents of a directory")
def list_directory(dirpath: str) -> list:
  """List all files and directories in the given directory."""
  return os.listdir(dirpath)


# Add a tool to get current working directory
@mcp.tool(description="Get current working directory")
def get_cwd() -> str:
  """Return the current working directory."""
  return str(pathlib.Path.cwd())


# Add a prompt for accessing file systems
@mcp.prompt()
def file_system_prompt() -> str:
  """Prompt helper for accessing file systems."""
  return (
      "You are a helpful assistant with access to the local filesystem. You can"
      " read files and list directories to help the user with their request."
  )


# Graceful shutdown handler
async def shutdown(signal, loop):
  """Cleanup tasks on shutdown."""
  print(f"\nReceived exit signal {signal.name}...")
  tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  for task in tasks:
    task.cancel()
  print(f"Cancelling {len(tasks)} outstanding tasks")
  await asyncio.gather(*tasks, return_exceptions=True)
  loop.stop()


# Main entry point with mTLS enabled
if __name__ == "__main__":
  cert_dir = os.path.dirname(os.path.abspath(__file__))
  keyfile = os.path.join(cert_dir, "server.key")
  certfile = os.path.join(cert_dir, "server.crt")

  if not (os.path.exists(keyfile) and os.path.exists(certfile)):
    print(f"Error: mTLS cert files not found in {cert_dir}")
    print("Please generate them using the helper script:")
    print(f"  ./generate_mtls_certs.sh")
    sys.exit(1)

  # Configure SSL context for mTLS
  print("Configuring SSL context for mTLS...")

  # Allow explicit CA certs override (useful for testing with custom CA signed certs)
  ca_certs = os.environ.get("SSL_CA_CERTS")
  temp_ca_file = None

  if ca_certs:
    print(f"  Using explicit SSL_CA_CERTS: {ca_certs}")
  else:
    has_cert_source = google_mtls.has_default_client_cert_source()
    print(f"  has_default_client_cert_source: {has_cert_source}")
    print(f"  default cafile: {ssl.get_default_verify_paths().cafile}")

    if has_cert_source:
      try:
        callback = google_mtls.default_client_cert_source()
        client_cert_bytes, _ = callback()
        temp_ca_file = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
        temp_ca_file.write(client_cert_bytes)
        temp_ca_file.close()
        ca_certs = temp_ca_file.name
        print(f"  Loaded client cert to trust: {ca_certs}")
      except Exception as e:
        print(f"  Warning: Failed to load default client cert: {e}")
        ca_certs = ssl.get_default_verify_paths().cafile
    else:
      print("  No default client cert source found. Using system CAs.")
      ca_certs = ssl.get_default_verify_paths().cafile

  print(f"  Using ca_certs for client verification: {ca_certs}")

  app = mcp.sse_app()

  config = uvicorn.Config(
      app,
      host=mcp.settings.host,
      port=mcp.settings.port,
      log_level=mcp.settings.log_level.lower(),
      ssl_keyfile=keyfile,
      ssl_certfile=certfile,
      ssl_cert_reqs=int(ssl.CERT_REQUIRED),
      ssl_ca_certs=ca_certs,
  )
  server = uvicorn.Server(config)

  print(
      "Starting MCP server with mTLS on"
      f" https://{mcp.settings.host}:{mcp.settings.port}"
  )
  try:
    asyncio.run(server.serve())
  except KeyboardInterrupt:
    print("\nServer shutting down gracefully...")
  except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)
  finally:
    if temp_ca_file:
      try:
        os.unlink(temp_ca_file.name)
        print(f"Cleaned up temp CA file: {temp_ca_file.name}")
      except OSError:
        pass
    print("Thank you for using the Filesystem MCP Server!")
