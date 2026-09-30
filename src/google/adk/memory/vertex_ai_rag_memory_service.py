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

import asyncio
import base64
import binascii
from collections import OrderedDict
import json
import logging
import os
import tempfile
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  import agentplatform

  from ..events.event import Event
  from ..sessions.session import Session


logger = logging.getLogger("google_adk." + __name__)

_SOURCE_DISPLAY_NAME_PREFIX = "adk-memory-v1."

_RAG_FILE_PAGE_SIZE = 100

# Scoping walks the corpus file list, so it is capped to keep the cost of a
# search independent of how large a shared corpus grows.
_MAX_RAG_FILE_PAGES = 10


def _encode_source_display_name_part(value: str) -> str:
  return (
      base64.urlsafe_b64encode(value.encode("utf-8"))
      .decode("ascii")
      .rstrip("=")
  )


def _decode_source_display_name_part(value: str) -> str:
  padded_value = value + "=" * (-len(value) % 4)
  return base64.b64decode(
      padded_value.encode("ascii"), altchars=b"-_", validate=True
  ).decode("utf-8")


def _build_source_display_name(
    app_name: str, user_id: str, session_id: str
) -> str:
  return _SOURCE_DISPLAY_NAME_PREFIX + ".".join([
      _encode_source_display_name_part(app_name),
      _encode_source_display_name_part(user_id),
      _encode_source_display_name_part(session_id),
  ])


def _parse_source_display_name(
    source_display_name: str,
) -> tuple[str, str, str] | None:
  if source_display_name.startswith(_SOURCE_DISPLAY_NAME_PREFIX):
    parts = source_display_name[len(_SOURCE_DISPLAY_NAME_PREFIX) :].split(".")
    if len(parts) != 3:
      return None
    try:
      return (
          _decode_source_display_name_part(parts[0]),
          _decode_source_display_name_part(parts[1]),
          _decode_source_display_name_part(parts[2]),
      )
    except (binascii.Error, UnicodeDecodeError, UnicodeEncodeError):
      return None

  # Legacy display names were dot-delimited. Only the exact three-part form is
  # unambiguous, so dotted app/user/session IDs are intentionally ignored.
  parts = source_display_name.split(".")
  if len(parts) != 3:
    return None
  return parts[0], parts[1], parts[2]


async def _scoped_rag_resources(
    client: agentplatform.AsyncClient,
    rag_resources: list[types.VertexRagStoreRagResource],
    app_name: str,
    user_id: str,
) -> list[types.VertexRagStoreRagResource] | None:
  """Returns resources naming only the files owned by one app and user.

  Returns None when a corpus cannot be listed within the page budget, in which
  case the caller retrieves without narrowing the resources.
  """
  from agentplatform import types as agentplatform_types

  scoped_resources: list[types.VertexRagStoreRagResource] = []
  for rag_resource in rag_resources:
    rag_corpus = rag_resource.rag_corpus
    if not rag_corpus:
      return None

    rag_file_ids: list[str] = []
    page_token: str | None = None
    for _ in range(_MAX_RAG_FILE_PAGES):
      response = await client.rag.list_files(
          name=rag_corpus,
          config=agentplatform_types.ListRagFilesConfig(
              page_size=_RAG_FILE_PAGE_SIZE, page_token=page_token
          ),
      )
      for rag_file in response.rag_files or []:
        session_info = _parse_source_display_name(rag_file.display_name or "")
        if (
            not session_info
            or session_info[0] != app_name
            or session_info[1] != user_id
            or not rag_file.name
        ):
          continue
        # rag_file_ids takes the bare file id, not the full resource name.
        rag_file_ids.append(rag_file.name.rsplit("/", 1)[-1])
      page_token = response.next_page_token
      if not page_token:
        break

    if page_token:
      # Scoping to the files seen so far would hide the caller's own memories,
      # so an incomplete listing is abandoned instead.
      logger.warning(
          "Listing %s did not finish within %d pages, so retrieval is not"
          " scoped to the requesting app and user.",
          rag_corpus,
          _MAX_RAG_FILE_PAGES,
      )
      return None
    if rag_file_ids:
      scoped_resources.append(
          types.VertexRagStoreRagResource(
              rag_corpus=rag_corpus, rag_file_ids=rag_file_ids
          )
      )

  return scoped_resources


class VertexAiRagMemoryService(BaseMemoryService):
  """A memory service that uses Agent Platform RAG for storage and retrieval."""

  def __init__(
      self,
      rag_corpus: Optional[str] = None,
      similarity_top_k: Optional[int] = None,
      vector_distance_threshold: float = 10,
      project: Optional[str] = None,
      location: Optional[str] = None,
  ):
    """Initializes a VertexAiRagMemoryService.

    Args:
        rag_corpus: The name of the Agent Platform RAG corpus to use. Format:
          ``projects/{project}/locations/{location}/ragCorpora/{rag_corpus_id}``
          or ``{rag_corpus_id}``
        similarity_top_k: The number of contexts to retrieve.
        vector_distance_threshold: Only returns contexts with vector distance
          smaller than the threshold.
        project: The project to use for the Agent Platform RAG corpus. If not
          set, the value of the GOOGLE_CLOUD_PROJECT environment variable is
          used.
        location: The location to use for the Agent Platform RAG corpus. If not
          set, the value of the GOOGLE_CLOUD_LOCATION environment variable is
          used.
    """
    try:
      import agentplatform  # noqa: F401
    except ImportError as e:
      from ..utils._dependency import missing_extra

      raise missing_extra("google-cloud-aiplatform", "gcp") from e

    self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")

    # Fallback: if the fully-qualified corpus name is provided, use it to
    # determine the project and location, if they are not already set.
    if (not self._project or not self._location) and (
        rag_corpus and rag_corpus.startswith("projects/")
    ):
      parts = rag_corpus.split("/")
      if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "locations":
        self._project = self._project or parts[1]
        self._location = self._location or parts[3]

    # Top-k belongs on the retrieval query, not on the store: the
    # retrieveContexts request's VertexRagStore has no such field.
    self._similarity_top_k = similarity_top_k
    self._vertex_rag_store = types.VertexRagStore(
        rag_resources=[
            types.VertexRagStoreRagResource(rag_corpus=rag_corpus),
        ],
        vector_distance_threshold=vector_distance_threshold,
    )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    rag_resources = self._vertex_rag_store.rag_resources or ()
    corpus_names = tuple(
        resource.rag_corpus for resource in rag_resources if resource.rag_corpus
    )
    if not corpus_names or len(corpus_names) != len(rag_resources):
      raise ValueError("rag_corpus must be set on every RAG resource.")

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

    import agentplatform

    temp_file_path: str | None = None
    try:
      with tempfile.NamedTemporaryFile(
          mode="w",
          delete=False,
          encoding="utf-8",
          suffix=".txt",
      ) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(output_string)

      client = agentplatform.Client(
          project=self._project, location=self._location
      ).aio
      rag = client.rag
      try:
        # Fails fast: the RAG API cannot roll back corpora already written.
        for corpus_name in corpus_names:
          await rag.upload_file(
              corpus_name=corpus_name,
              path=temp_file_path,
              display_name=_build_source_display_name(
                  session.app_name, session.user_id, session.id
              ),
          )
      finally:
        # Shielded so a cancellation racing the close cannot leak the
        # underlying HTTP session.
        await asyncio.shield(client.aclose())
    finally:
      if temp_file_path:
        try:
          os.remove(temp_file_path)
        except OSError:
          # Best effort: this runs in a finally, so raising here would
          # displace the exception already propagating, and a cancelled
          # upload can leave the worker thread still holding the file.
          logger.warning(
              "Could not remove the temporary transcript at %s",
              temp_file_path,
              exc_info=True,
          )

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches for sessions that match the query using rag.retrieval_query."""
    import agentplatform
    from agentplatform import types as agentplatform_types

    from ..events.event import Event

    rag_resources = self._vertex_rag_store.rag_resources
    if not rag_resources:
      raise ValueError("Rag resources must be set.")

    client = agentplatform.Client(
        project=self._project, location=self._location
    ).aio
    try:
      try:
        scoped_resources = await _scoped_rag_resources(
            client, rag_resources, app_name, user_id
        )
      except Exception:  # pylint: disable=broad-except
        # Narrowing the resources only improves ranking and transfer; the
        # response filter below keeps an unnarrowed retrieval correct.
        logger.warning(
            "Listing the corpus failed, so retrieval is not scoped to the"
            " requesting app and user.",
            exc_info=True,
        )
        scoped_resources = None

      vertex_rag_store = self._vertex_rag_store
      if scoped_resources is not None:
        if not scoped_resources:
          return SearchMemoryResponse()
        vertex_rag_store = self._vertex_rag_store.model_copy(
            update={"rag_resources": scoped_resources}
        )

      response = await client.rag.retrieve_contexts(
          vertex_rag_store=vertex_rag_store,
          query=agentplatform_types.RagQuery(
              text=query,
              similarity_top_k=self._similarity_top_k,
          ),
      )
    finally:
      # Shielded so a cancellation racing the close cannot leak the
      # underlying HTTP session.
      await asyncio.shield(client.aclose())
    memory_results = []
    session_events_map: OrderedDict[str, list[list[Event]]] = OrderedDict()
    for context in response.contexts.contexts:
      # Still required: retrieval is unscoped whenever listing did not finish.
      source_display_name = getattr(context, "source_display_name", "")
      if not isinstance(source_display_name, str):
        continue
      session_info = _parse_source_display_name(source_display_name)
      if not session_info:
        continue
      source_app_name, source_user_id, session_id = session_info
      if source_app_name != app_name or source_user_id != user_id:
        continue
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
