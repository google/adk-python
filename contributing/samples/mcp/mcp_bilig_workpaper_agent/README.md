# Bilig WorkPaper MCP Agent

This sample shows how to connect ADK to a Bilig WorkPaper MCP server over
stdio. The server exposes a formula workbook as tools, so an agent can inspect
workbook inputs, edit input cells, recalculate dependent formulas, verify
readback, and persist the WorkPaper JSON document.

This is useful for formula-backed business logic where an agent needs a
reviewable workbook state without automating Excel, Google Sheets, or a browser
UI.

## Setup

Install the usual ADK sample dependencies and ensure Node.js/npm are available.
The MCP server is launched automatically with `npm exec`:

```bash
npm exec --yes --package @bilig/headless@0.40.41 -- \
  bilig-workpaper-mcp --workpaper ./quote.workpaper.json \
  --init-demo-workpaper --writable
```

You can choose the WorkPaper path by setting `BILIG_WORKPAPER_PATH`. If unset,
the ADK agent uses a temporary file path.

## Run the no-key smoke test

The deterministic smoke test exercises the MCP server directly and does not
require an LLM API key:

```bash
python contributing/samples/mcp/mcp_bilig_workpaper_agent/main.py
```

Expected proof:

- `Inputs!B3` changes from `0.25` to `0.4`
- expected customers changes from `5` to `8`
- expected ARR changes from `60000` to `96000`
- the WorkPaper JSON file persists
- restored readback matches the edited value

## Run in ADK Web

```bash
adk web contributing/samples
```

Then select **mcp_bilig_workpaper_agent** and try:

```text
Increase the win rate to 40%, then report expected customers, expected ARR,
and whether the WorkPaper JSON persisted.
```

The agent should use the WorkPaper tools to edit `Inputs!B3`, read dependent
formula cells from `Summary`, and report the persistence checks returned by the
MCP server.
