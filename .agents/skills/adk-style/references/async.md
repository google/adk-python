# Async and Concurrency Style Guide

- **I/O belongs in `async def`**: network calls, file system access, database
  queries, and subprocess waits all go in async functions. ADK runs everything
  on one event loop, so a synchronous call inside it stalls every concurrent
  agent, not just the caller.
- **Don't block the event loop**: no synchronous HTTP clients, `time.sleep`,
  or blocking file reads inside async code.
- **Wrap synchronous I/O in `asyncio.to_thread`** when a library offers no
  async API (`open()`, `pathlib`, most cloud SDK clients):

```python
async def save_data(path: Path, data: bytes) -> None:
  # Wrap the blocking write so the event loop stays free.
  await asyncio.to_thread(path.write_bytes, data)
```

No hook checks this — a blocking call passes CI and shows up later as
unexplained latency under concurrency, so it is worth catching in review.
