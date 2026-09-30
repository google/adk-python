# Debugging with `adk web`

`adk web` is a FastAPI server: a browser UI at `http://localhost:8000/dev-ui/`
plus the HTTP API below. Reach for it when you need to click through a
persisted session, or when you need the trace endpoints
(`references/logs-and-traces.md`).

## Starting the server

Check for an existing server before starting your own — a second one will fail
to bind port 8000, and the user may already have one with the sessions you care
about:

```bash
curl -s http://localhost:8000/health     # {"status":"ok"} if one is running
```

If none is running, start it in the background and shut it down when you are
done:

```bash
adk web {agents_dir}                     # http://127.0.0.1:8000
adk web -v --reload_agents {agents_dir}
```

`{agents_dir}` is a directory of agent subdirectories, or a single agent folder
(one containing `agent.py` or `root_agent.yaml`). It defaults to the current
directory.

Flag | Default | Note
--- | --- | ---
`--port` | `8000` | Use a second port to run two servers side by side.
`--host` | `127.0.0.1` | Endpoints are unauthenticated; keep it on loopback.
`--reload_agents` | off | Re-import agent modules when their files change. This is the one you want while editing an agent.
`--reload` | **on** | Uvicorn's own source autoreload. Pass `--no-reload` when a restart mid-run is confusing you.
`-v` / `--log_level` | `INFO` | Logs go to the terminal, not to a file — see `references/logs-and-traces.md`.

`adk api_server` takes the same flags but serves only the production-safe
routes — no UI and no `/dev/...` debug or trace endpoints. Use `adk web` when
debugging.

## Inspecting sessions

```bash
curl -s http://localhost:8000/list-apps | python3 -m json.tool

curl -s http://localhost:8000/apps/{app_name}/users/{user_id}/sessions \
  | python3 -m json.tool

curl -s http://localhost:8000/apps/{app_name}/users/{user_id}/sessions/{session_id} \
  | python3 -m json.tool
```

The session response holds the full event list. Fetch the raw JSON and write a
summarizer against the structure you actually see rather than a remembered
schema. Fields worth pulling per event: `author`, `branch`, `nodeInfo.path`,
`content.parts` (`text`, `functionCall`, `functionResponse`), `output`, and
`actions` (`transferToAgent`, `escalate`, `endOfAgent`). Keys are camelCase.

`DELETE .../sessions/{session_id}` exists; do not use it to tidy up after
yourself, because the user may still want the session in the UI.

## Sending a test message

Create a session, then post one turn. `/run` returns the whole event list as
JSON, which is far easier to assert on than a stream:

```bash
SESSION=$(curl -s -X POST http://localhost:8000/apps/{app_name}/users/test/sessions \
  -H "Content-Type: application/json" -d '{}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

curl -s -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d "{\"app_name\":\"{app_name}\",\"user_id\":\"test\",\"session_id\":\"$SESSION\",
       \"new_message\":{\"role\":\"user\",\"parts\":[{\"text\":\"{query}\"}]}}" \
  | python3 -m json.tool
```

Use `/run_sse` with `"streaming":true` and `curl -N` only when the bug is in
streaming itself — partial events, chunk ordering, or a stream that never
terminates.

A `POST` to a session id that does not exist returns 404 rather than creating
it, so create the session first. Supply your own id by passing
`{"session_id": "..."}` in the create body when you want a stable id across
runs.
