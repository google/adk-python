# Async and Concurrency Style Guide

-   **All I/O operations must be in async functions**: Any operation that
    performs I/O (network calls, file system access, database queries, etc.)
    must be defined in an `async def` function.
-   **Do not block the event loop**: Avoid calling blocking synchronous
    functions directly from async code.
-   **Wrap synchronous I/O**: If you must use a synchronous library for I/O
    (e.g., standard `open()`, `pathlib` file operations, or synchronous
    clients), wrap the blocking call in `asyncio.to_thread` to run it in a
    separate thread and prevent blocking the main event loop.

Example:

```python
async def save_data(path: Path, data: bytes) -> None:
  # Wrap blocking file write in asyncio.to_thread
  await asyncio.to_thread(path.write_bytes, data)
```
