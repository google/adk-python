"""
Memory service implementation using Mem0 for storing and retrieving memories.
"""

from collections import OrderedDict
import json
import os

from google.genai import types
from typing_extensions import override

from ..events.event import Event
from ..sessions.session import Session
from .base_memory_service import BaseMemoryService
from .base_memory_service import MemoryResult
from .base_memory_service import SearchMemoryResponse


class Mem0MemoryService(BaseMemoryService):
  """
  Memory service implementation using Mem0 for storing and retrieving memories.

  This service can be configured in two ways:

  1. Using a managed Mem0 service with an API key
     - Requires MEM0_API_KEY or passing api_key in config
     - Optionally specify org_id and project_id for organization projects

  2. Using a local Mem0 instance
     - Specify local_config with vector_store, llm, and embedder configurations

  Example local config:
  {
      "local_config": {
          "vector_store": {
              "provider": "qdrant",
              "config": {
                  "collection_name": "user_memory",
                  "host": "localhost",
                  "port": 6333,
                  "embedding_model_dims": "1536",
              },
          },
          "llm": {
              "provider": "openai",
              "config": {
                "api_key": "your_openai_api_key",
                "model": "gpt-4o-mini",
              },
          },
          "embedder": {
              "provider": "openai",
              "config": {
                  "api_key": "your_openai_api_key",
                  "model": "text-embedding-3-small",
              },
          },
      }
  }

  Example managed config:
  {
      "api_key": "your_mem0_api_key",
      "org_id": "your_org_id",
      "project_id": "your_project_id"
  }

  Note: Requires the mem0 package to be installed.
  """

  def __init__(self, mem0_config=None):
    try:
      from mem0 import Memory
      from mem0 import MemoryClient
    except ImportError as exc:
      raise ImportError(
          "Could not import mem0. Please install it with 'pip install mem0' "
          "to use the Mem0MemoryService."
      ) from exc

    mem0_config = mem0_config or {}
    mem0_api_key = mem0_config.get("api_key") or os.getenv("MEM0_API_KEY")
    mem0_org_id = mem0_config.get("org_id")
    mem0_project_id = mem0_config.get("project_id")
    mem0_local_config = mem0_config.get("local_config")

    if mem0_api_key:
      if mem0_org_id and mem0_project_id:
        self.memory = MemoryClient(
            api_key=mem0_api_key,
            org_id=mem0_org_id,
            project_id=mem0_project_id,
        )
      else:
        self.memory = MemoryClient(api_key=mem0_api_key)
    else:
      if mem0_local_config:
        self.memory = Memory.from_config(mem0_local_config)
      else:
        self.memory = Memory()

  @override
  def add_session_to_memory(self, session: Session):
    """
    Add a session to the memory.
    """

    session_events = []
    for event in session.events:
      if not event.content or not event.content.parts:
        continue
      text_parts = [
          part.text.replace("\n", " ")
          for part in event.content.parts
          if part.text
      ]
      if text_parts:
        session_events.append(
            json.dumps({
                "author": event.author,
                "timestamp": event.timestamp,
                "text": ".".join(text_parts),
            })
        )

    output_string = "\n".join(session_events)

    extra_params = {}

    if not hasattr(self.memory, "llm"):
      extra_params["output_format"] = "v1.1"

    self.memory.add(
        messages=output_string,
        user_id=session.user_id,
        run_id=session.id,
        metadata={
            "app_name": session.app_name,
            "display_name": (
                f"{session.app_name}.{session.user_id}.{session.id}"
            ),
        },
        infer=False,
        **extra_params,
    )

  @override
  def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """
    Search the memory for the most relevant sessions.
    """

    extra_params = {"filters": {"app_name": app_name}}

    if not hasattr(self.memory, "llm"):
      extra_params["output_format"] = "v1.1"
      extra_params["version"] = "v2"
      extra_params["filters"] = {"metadata": {"app_name": app_name}}

    response = self.memory.search(
        query=query,
        user_id=user_id,
        limit=10,
        **extra_params,
    )

    memory_results = []
    session_events_map = OrderedDict()

    for result in response["results"]:
      memory = result["memory"]
      metadata = result["metadata"]
      session_id = metadata["display_name"].split(".")[-1]

      events = []
      lines = memory.split("\n")
      for line in lines:
        line = line.strip()
        if not line:
          continue

        try:
          event_data = json.loads(line)

          author = event_data.get("author", "")
          timestamp = float(event_data.get("timestamp", 0))
          text = event_data.get("text", "")

          content = types.Content(parts=[types.Part(text=text)])
          event = Event(author=author, timestamp=timestamp, content=content)
          events.append(event)
        except json.JSONDecodeError:
          continue

      if session_id in session_events_map:
        session_events_map[session_id].append(events)
      else:
        session_events_map[session_id] = [events]

    for session_id, event_lists in session_events_map.items():
      for events in _merge_event_lists(event_lists):
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        memory_results.append(
            MemoryResult(session_id=session_id, events=sorted_events)
        )

    return SearchMemoryResponse(memories=memory_results)


def _merge_event_lists(event_lists: list[list[Event]]) -> list[list[Event]]:
  """Merge event lists that have overlapping timestamps."""
  merged = []
  while event_lists:
    current = event_lists.pop(0)
    current_ts = {event.timestamp for event in current}
    merge_found = True

    while merge_found:
      merge_found = False
      remaining = []
      for other in event_lists:
        other_ts = {event.timestamp for event in other}
        if current_ts & other_ts:
          new_events = [e for e in other if e.timestamp not in current_ts]
          current.extend(new_events)
          current_ts.update(e.timestamp for e in new_events)
          merge_found = True
        else:
          remaining.append(other)
      event_lists = remaining
    merged.append(current)
  return merged
