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


from __future__ import annotations

from collections import OrderedDict
import json
import os
import tempfile
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override
from vertexai.preview import rag

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..events.event import Event
  from ..sessions.session import Session


class VertexAiRagMemoryService(BaseMemoryService):
  """A memory service that uses Vertex AI RAG for storage and retrieval."""

  def __init__(
      self,
      rag_corpus: Optional[str] = None,
      similarity_top_k: Optional[int] = None,
      vector_distance_threshold: float = 10,
  ):
    """Initializes a VertexAiRagMemoryService.

    Args:
        rag_corpus: The name of the Vertex AI RAG corpus to use. Format:
          ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}``
          or ``{rag_corpus_id}``
        similarity_top_k: The number of contexts to retrieve.
        vector_distance_threshold: Only returns contexts with vector distance
          smaller than the threshold..
    """
    self._vertex_rag_store = types.VertexRagStore(
        rag_resources=[
            types.VertexRagStoreRagResource(rag_corpus=rag_corpus),
        ],
        similarity_top_k=similarity_top_k,
        vector_distance_threshold=vector_distance_threshold,
    )

  @override
  async def add_session_to_memory(self, session: Session):
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt"
    ) as temp_file:

      output_lines = []
      for event in session.events:
        if not event.content or not event.content.parts:
          continue
        text_parts = [
            part.text.replace("\n", " ")
            for part in event.content.parts
            if part.text
        ]
        if text_parts:
          output_lines.append(
              json.dumps({
                  "author": event.author,
                  "timestamp": event.timestamp,
                  "text": ".".join(text_parts),
              })
          )
      output_string = "\n".join(output_lines)
      temp_file.write(output_string)
      temp_file_path = temp_file.name

    if not self._vertex_rag_store.rag_resources:
      raise ValueError("Rag resources must be set.")

    for rag_resource in self._vertex_rag_store.rag_resources:
      rag.upload_file(
          corpus_name=rag_resource.rag_corpus,
          path=temp_file_path,
          # this is the temp workaround as upload file does not support
          # adding metadata, thus use display_name to store the session info.
          display_name=f"{session.app_name}.{session.user_id}.{session.id}",
      )

    os.remove(temp_file_path)

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches for sessions that match the query using rag.retrieval_query."""
    from ..events.event import Event

    response = rag.retrieval_query(
        text=query,
        rag_resources=self._vertex_rag_store.rag_resources,
        rag_corpora=self._vertex_rag_store.rag_corpora,
        similarity_top_k=self._vertex_rag_store.similarity_top_k,
        vector_distance_threshold=self._vertex_rag_store.vector_distance_threshold,
    )

    memory_results = []
    session_events_map = OrderedDict()
    for context in response.contexts.contexts:
      # filter out context that is not related
      # TODO: Add server side filtering by app_name and user_id.
      if not context.source_display_name.startswith(f"{app_name}.{user_id}."):
        continue
      session_id = context.source_display_name.split(".")[-1]
      events = []
      if context.text:
        lines = context.text.split("\n")

        for line in lines:
          line = line.strip()
          if not line:
            continue

          try:
            # Try to parse as JSON
            event_data = json.loads(line)

            author = event_data.get("author", "")
            timestamp = float(event_data.get("timestamp", 0))
            text = event_data.get("text", "")

            content = types.Content(parts=[types.Part(text=text)])
            event = Event(author=author, timestamp=timestamp, content=content)
            events.append(event)
          except json.JSONDecodeError:
            # Not valid JSON, skip this line
            continue

      if session_id in session_events_map:
        session_events_map[session_id].append(events)
      else:
        session_events_map[session_id] = [events]

    # Remove overlap and combine events from the same session.
    for session_id, event_lists in session_events_map.items():
      for events in _merge_event_lists(event_lists):
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        memory_results.extend([
            MemoryEntry(
                author=event.author,
                content=event.content,
                timestamp=_utils.format_timestamp(event.timestamp),
            )
            for event in sorted_events
            if event.content
        ])
    return SearchMemoryResponse(memories=memory_results)


def _merge_event_lists(event_lists: list[list[Event]]) -> list[list[Event]]:
  """Merge event lists that have overlapping timestamps."""
  merged = []
  while event_lists:
    current = event_lists.pop(0)
    current_ts = {event.timestamp for event in current}
    merge_found = True

    # Keep merging until no new overlap is found.
    while merge_found:
      merge_found = False
      remaining = []
      for other in event_lists:
        other_ts = {event.timestamp for event in other}
        # Overlap exists, so we merge and use the merged list to check again
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
