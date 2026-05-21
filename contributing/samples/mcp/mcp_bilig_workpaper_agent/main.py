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
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client


def workpaper_path() -> Path:
  configured = os.getenv("BILIG_WORKPAPER_PATH")
  if configured:
    return Path(configured)
  return Path(tempfile.mkdtemp(prefix="adk-bilig-")) / "quote.workpaper.json"


def server_params(path: Path) -> StdioServerParameters:
  return StdioServerParameters(
      command="npm",
      args=[
          "exec",
          "--yes",
          "--package",
          "@bilig/headless@0.40.41",
          "--",
          "bilig-workpaper-mcp",
          "--workpaper",
          str(path),
          "--init-demo-workpaper",
          "--writable",
      ],
  )


async def call_json(
    session: ClientSession, name: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
  result = await session.call_tool(
      name,
      arguments or {},
      read_timeout_seconds=timedelta(seconds=60),
  )
  if result.isError:
    raise RuntimeError(f"{name} failed: {result.content}")

  text_items = [item.text for item in result.content if item.type == "text"]
  if not text_items:
    raise RuntimeError(f"{name} returned no text content")
  return json.loads(text_items[0])


async def main() -> None:
  path = workpaper_path()
  async with stdio_client(server_params(path)) as streams:
    async with ClientSession(
        *streams, read_timeout_seconds=timedelta(seconds=60)
    ) as session:
      await session.initialize()

      tools = await session.list_tools()
      print("Tools:", [tool.name for tool in tools.tools])

      inputs = await call_json(session, "read_range", {"range": "Inputs!A1:B5"})
      summary = await call_json(
          session, "read_range", {"range": "Summary!A1:B5"}
      )
      print("Inputs:", inputs["serialized"])
      print("Summary formulas:", summary["serialized"])

      before_arr = await call_json(
          session, "read_cell", {"sheetName": "Summary", "address": "B3"}
      )
      edit = await call_json(
          session,
          "set_cell_contents",
          {"sheetName": "Inputs", "address": "B3", "value": 0.4},
      )
      after_customers = await call_json(
          session, "read_cell", {"sheetName": "Summary", "address": "B2"}
      )
      after_arr = await call_json(
          session, "read_cell", {"sheetName": "Summary", "address": "B3"}
      )
      exported = await call_json(
          session, "export_workpaper_document", {"includeConfig": True}
      )

      print("ARR before:", before_arr["value"]["value"])
      print("Customers after:", after_customers["value"]["value"])
      print("ARR after:", after_arr["value"]["value"])
      print("Persisted:", edit["checks"]["persisted"])
      print("Restored matches after:", edit["checks"]["restoredMatchesAfter"])
      print("Exported bytes:", exported["serializedBytes"])
      print("WorkPaper path:", path)

      assert before_arr["value"]["value"] == 60000
      assert after_customers["value"]["value"] == 8
      assert after_arr["value"]["value"] == 96000
      assert edit["checks"]["persisted"] is True
      assert edit["checks"]["restoredMatchesAfter"] is True
      assert path.exists()


if __name__ == "__main__":
  asyncio.run(main())
