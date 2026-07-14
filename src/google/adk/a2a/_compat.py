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

"""Single point of divergence between a2a-sdk 0.3.x and 1.x.

Isolates every API that differs between a2a-sdk 0.3.x and 1.x so the rest of ADK
imports version-agnostic helpers from here instead of reaching into ``a2a.*``
directly. ``IS_A2A_V1`` selects the active branch at import time based on the
installed a2a-sdk version.
"""

from __future__ import annotations

import base64
import dataclasses
from datetime import datetime
from datetime import timezone
import json
from typing import Any
from typing import AsyncGenerator
from typing import Callable
from typing import Optional

from a2a.client.client import ClientConfig as A2AClientConfig
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.types import AgentCard
from a2a.types import APIKeySecurityScheme
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Part
from a2a.types import Role
from a2a.types import SecurityScheme
from a2a.types import Task
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import ParseDict


def _make_proto_timestamp(dt: Optional[datetime] = None) -> Any:
  """Build a google.protobuf.Timestamp from a datetime (or now). 1.x only."""
  from google.protobuf import timestamp_pb2

  ts = timestamp_pb2.Timestamp()
  ts.FromDatetime(dt or datetime.now(timezone.utc))
  return ts


def _make_proto_value_from_dict(d: dict[str, Any]) -> Any:
  """Wrap a plain dict as a google.protobuf.Value (struct_value). 1.x only."""
  from google.protobuf.struct_pb2 import Struct
  from google.protobuf.struct_pb2 import Value

  v = Value()
  s = Struct()
  ParseDict(d, s)
  v.struct_value.CopyFrom(s)
  return v


def _proto_to_dict(msg: Any) -> dict[str, Any]:
  """Convert a protobuf message (e.g. Struct/Value) to a plain dict."""
  result: dict[str, Any] = MessageToDict(msg)
  return result


# -----------------------------------------------------------------------------
# Version detection
# -----------------------------------------------------------------------------
try:
  from a2a.types import StreamResponse as _StreamResponse  # noqa: F401

  IS_A2A_V1 = True
except ImportError:
  IS_A2A_V1 = False


# -----------------------------------------------------------------------------
# Enum & constant wrappers
# -----------------------------------------------------------------------------
if IS_A2A_V1:
  # 1.x: protobuf EnumTypeWrapper — access values as integer constants.
  ROLE_USER = Role.Value("ROLE_USER")
  ROLE_AGENT = Role.Value("ROLE_AGENT")
  TS_SUBMITTED = TaskState.Value("TASK_STATE_SUBMITTED")
  TS_WORKING = TaskState.Value("TASK_STATE_WORKING")
  TS_COMPLETED = TaskState.Value("TASK_STATE_COMPLETED")
  TS_FAILED = TaskState.Value("TASK_STATE_FAILED")
  TS_INPUT_REQUIRED = TaskState.Value("TASK_STATE_INPUT_REQUIRED")
  TS_AUTH_REQUIRED = TaskState.Value("TASK_STATE_AUTH_REQUIRED")
  TS_CANCELED = TaskState.Value("TASK_STATE_CANCELED")

  # 1.x: TransportProtocol is in ``a2a.utils.constants`` as a ``str`` Enum.
  from a2a.utils.constants import TransportProtocol as TransportProtocol

  TP_JSONRPC = TransportProtocol.JSONRPC
  TP_HTTP_JSON = TransportProtocol.HTTP_JSON
  TP_GRPC = TransportProtocol.GRPC

else:
  # 0.3.x: pydantic enum
  ROLE_USER, ROLE_AGENT = Role.user, Role.agent
  TS_SUBMITTED = TaskState.submitted
  TS_WORKING = TaskState.working
  TS_COMPLETED = TaskState.completed
  TS_FAILED = TaskState.failed
  TS_INPUT_REQUIRED = TaskState.input_required
  TS_AUTH_REQUIRED = TaskState.auth_required
  TS_CANCELED = TaskState.canceled

  # 0.3.x: TransportProtocol is in ``a2a.types``.
  from a2a.types import TransportProtocol as TransportProtocol  # type: ignore[assignment,no-redef,attr-defined]

  TP_JSONRPC = TransportProtocol.jsonrpc
  TP_HTTP_JSON = TransportProtocol.http_json
  TP_GRPC = TransportProtocol.grpc


# Normalized client-stream item (output of ``make_stream_normalizer``). On 0.3.x
# this is the SDK's ``ClientEvent`` tuple; 1.x removed it, so rebuild the
# equivalent tuple from that version's types.
if IS_A2A_V1:
  A2AClientEvent = tuple[
      Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None
  ]
else:
  from a2a.client import ClientEvent as A2AClientEvent  # type: ignore[assignment,no-redef,attr-defined]  # noqa: F401


# -----------------------------------------------------------------------------
# Part construction & reading
# -----------------------------------------------------------------------------
def make_text_part(text: str) -> Part:
  """Builds a text Part."""
  if IS_A2A_V1:
    # 1.x: Part is a flat proto message; oneof ``content`` selects the variant.
    return Part(text=text)
  else:
    # 0.3.x: Part wraps a discriminated union via ``.root``.
    from a2a.types import TextPart

    return Part(root=TextPart(text=text))


def is_text_part(p: Part) -> bool:
  """Returns True if the Part carries text content."""
  if IS_A2A_V1:
    is_text: bool = p.WhichOneof("content") == "text"
    return is_text
  else:
    from a2a.types import TextPart

    return isinstance(p.root, TextPart)


def is_file_part(p: Part) -> bool:
  """Returns True if the Part carries raw bytes or a URL."""
  if IS_A2A_V1:
    return p.WhichOneof("content") in ("raw", "url")
  else:
    from a2a.types import FilePart

    return isinstance(p.root, FilePart)


def is_data_part(p: Part) -> bool:
  """Returns True if the Part carries structured data."""
  if IS_A2A_V1:
    is_data: bool = p.WhichOneof("content") == "data"
    return is_data
  else:
    from a2a.types import DataPart

    return isinstance(p.root, DataPart)


def part_text(p: Part) -> str:
  """Reads the text of a text Part."""
  if IS_A2A_V1:
    v1_text: str = p.text
    return v1_text
  else:
    text: str = p.root.text
    return text


# -----------------------------------------------------------------------------
# Generic metadata access/mutation helpers
# -----------------------------------------------------------------------------
def part_metadata(p: Part) -> dict[str, Any]:
  """Reads a Part's metadata."""
  if IS_A2A_V1:
    # 1.x: Part.metadata is a Struct field (flat on Part, not on ``root``).
    if p.HasField("metadata"):
      meta: dict[str, Any] = MessageToDict(p.metadata)
      return meta
    return {}
  else:
    # 0.3.x: metadata lives on ``p.root`` (the discriminated-union inner).
    return getattr(p.root, "metadata", None) or {}


def set_part_metadata(p: Part, metadata: dict[str, Any]) -> None:
  """Writes a Part's metadata."""
  if IS_A2A_V1:
    from google.protobuf.struct_pb2 import Struct

    p.metadata.CopyFrom(ParseDict(metadata, Struct()))
  else:
    p.root.metadata = metadata


# -----------------------------------------------------------------------------
# File / Data Part builders & readers
# 1.x: ``Part`` is flat — URI in scalar ``url`` + ``media_type``/``filename``;
#      bytes in scalar ``raw`` + ``media_type``; data in proto ``Value``.
# 0.3.x: ``Part(root=FilePart(file=FileWithUri/FileWithBytes))`` /
#        ``Part(root=DataPart(data=..., metadata=...))``
# -----------------------------------------------------------------------------
def make_file_part_with_uri(
    *, uri: str, mime_type: str = "", name: Optional[str] = None
) -> Part:
  """Builds a file Part referencing a URI."""
  if IS_A2A_V1:
    p = Part()
    p.url = uri or ""
    p.media_type = mime_type or ""
    if name:
      p.filename = name
    return p
  else:
    from a2a.types import FilePart
    from a2a.types import FileWithUri

    return Part(
        root=FilePart(file=FileWithUri(uri=uri, mime_type=mime_type, name=name))
    )


def make_file_part_with_bytes(
    *, data: bytes, mime_type: str = "", name: Optional[str] = None
) -> Part:
  """Builds a file Part carrying raw bytes.

  ``data`` is the raw (already-decoded) bytes; 0.3.x stores it base64-encoded.
  """
  if IS_A2A_V1:
    p = Part()
    p.raw = data or b""
    p.media_type = mime_type or ""
    if name:
      p.filename = name
    return p
  else:
    from a2a.types import FilePart
    from a2a.types import FileWithBytes

    return Part(
        root=FilePart(
            file=FileWithBytes(
                bytes=base64.b64encode(data).decode("utf-8"),
                mime_type=mime_type,
                name=name,
            )
        )
    )


def make_data_part(
    *, data: dict[str, Any], metadata: Optional[dict[str, Any]] = None
) -> Part:
  """Builds a structured-data Part."""
  if IS_A2A_V1:
    p = Part()
    p.data.CopyFrom(_make_proto_value_from_dict(data))
    if metadata:
      set_part_metadata(p, metadata)
    return p
  else:
    from a2a.types import DataPart

    return Part(root=DataPart(data=data, metadata=metadata))


def make_data_part_from_blob(
    raw_json: bytes, *, extra_metadata: Optional[dict[str, Any]] = None
) -> Part:
  """Rebuilds a data Part from a generic inline-blob payload.

  Inverse of ``data_part_blob_bytes``.

  1.x: the blob holds only the structured ``data`` dict, so deserialize it and
  attach ``extra_metadata`` (carried separately) onto the Part.
  0.3.x: the blob is a fully serialized ``DataPart`` (``data`` + embedded
  ``metadata``), so deserialize it directly and merge any ``extra_metadata``.
  """
  if IS_A2A_V1:
    data_dict = json.loads(raw_json)
    return make_data_part(data=data_dict, metadata=extra_metadata)
  else:
    from a2a.types import DataPart

    inner = DataPart.model_validate_json(raw_json)
    if extra_metadata:
      if inner.metadata is None:
        inner.metadata = {}
      inner.metadata.update(extra_metadata)
    return Part(root=inner)


def file_part_uri(p: Part) -> Optional[str]:
  """Returns the URI of a URI-backed file Part, else None."""
  if IS_A2A_V1:
    return p.url if p.WhichOneof("content") == "url" else None
  else:
    from a2a.types import FileWithUri

    inner = p.root
    file = getattr(inner, "file", None)
    return getattr(file, "uri", None) if isinstance(file, FileWithUri) else None


def file_part_bytes(p: Part) -> Optional[bytes]:
  """Returns the raw (decoded) bytes of a bytes-backed file Part, else None."""
  if IS_A2A_V1:
    return p.raw if p.WhichOneof("content") == "raw" else None
  else:
    from a2a.types import FileWithBytes

    inner = p.root
    file = getattr(inner, "file", None)
    if isinstance(file, FileWithBytes):
      return base64.b64decode(file.bytes)
    return None


def file_part_mime_type(p: Part) -> Optional[str]:
  """Returns the media type of a file Part."""
  if IS_A2A_V1:
    return p.media_type or None
  else:
    file = getattr(p.root, "file", None)
    return getattr(file, "mime_type", None) if file is not None else None


def file_part_name(p: Part) -> Optional[str]:
  """Returns the display name / filename of a file Part."""
  if IS_A2A_V1:
    return p.filename or None
  else:
    file = getattr(p.root, "file", None)
    return getattr(file, "name", None) if file is not None else None


def data_part_dict(p: Part) -> dict[str, Any]:
  """Returns the structured data of a data Part as a plain dict.

  1.x: protobuf ``Value``/``Struct`` has no integer type, so all numbers
  round-trip as ``float`` (e.g. an int ``5`` becomes ``5.0``). Callers
  comparing numeric fields across the version boundary must account for this
  (compare as ``float`` or normalize). 0.3.x preserves the original Python
  types (e.g. ints stay ints).
  """
  if IS_A2A_V1:
    data: dict[str, Any] = MessageToDict(p.data)
    return data
  else:
    root_data: dict[str, Any] = p.root.data
    return root_data


def data_part_blob_bytes(p: Part) -> bytes:
  """Serializes a data Part for embedding as a generic inline blob.

  1.x: only the structured ``data`` dict is serialized; the part metadata is
  carried separately on the GenAI part.
  0.3.x: the *entire* ``DataPart`` is serialized (``data`` + ``metadata`` +
  ``kind``).
  """
  if IS_A2A_V1:
    return json.dumps(data_part_dict(p)).encode("utf-8")
  else:
    blob: bytes = p.root.model_dump_json(
        by_alias=True, exclude_none=True
    ).encode("utf-8")
    return blob


# -----------------------------------------------------------------------------
# Serialization helper (model_dump → MessageToDict)
# -----------------------------------------------------------------------------
def a2a_to_dict(obj: Any) -> dict[str, Any]:
  """Serializes an A2A object to a plain dict."""
  if IS_A2A_V1:
    proto_dict: dict[str, Any] = MessageToDict(obj)
    return proto_dict
  else:
    model_dict: dict[str, Any] = obj.model_dump(
        exclude_none=True, by_alias=True
    )
    return model_dict


# -----------------------------------------------------------------------------
# AgentCard construction from JSON dict
# -----------------------------------------------------------------------------
def parse_agent_card(data: dict[str, Any]) -> AgentCard:
  """Builds an AgentCard from a JSON dict."""
  if IS_A2A_V1:
    from a2a.client.card_resolver import parse_agent_card as _parse

    return _parse(data)
  else:
    return AgentCard(**data)


def build_agent_card(
    *,
    name: str,
    description: str,
    version: str,
    url: str,
    protocol_binding: str,
    protocol_version: Optional[str] = None,
    skills: Any = (),
    capabilities: Any = None,
    provider: Any = None,
    security_schemes: Any = None,
    doc_url: Optional[str] = None,
    default_input_modes: Any = ("text/plain",),
    default_output_modes: Any = ("text/plain",),
    supports_authenticated_extended_card: bool = False,
    streaming: bool = False,
) -> AgentCard:
  """Builds an ``AgentCard`` from primitive fields.

  0.3.x: ``AgentCard`` is pydantic — RPC URL is the top-level ``url`` field,
         transport is ``preferredTransport``.
  1.x:   ``AgentCard`` is a proto message — RPC URL lives in
         ``supported_interfaces[i].url`` (with ``protocol_binding``).
  """

  def _as_dict(obj: Any) -> Any:
    if obj is None:
      return None
    if isinstance(obj, dict):
      return obj
    return a2a_to_dict(obj)

  # Version-correct default protocol version when the caller doesn't specify
  # one (1.x interfaces default to "1.0"; 0.3.x cards default to "0.3.0").
  resolved_protocol_version = protocol_version or (
      "1.0" if IS_A2A_V1 else "0.3.0"
  )

  default_capabilities = {"streaming": streaming, "push_notifications": False}

  if IS_A2A_V1:
    iface: dict[str, Any] = {
        "url": url.rstrip("/"),
        "protocol_binding": protocol_binding,
        "protocol_version": resolved_protocol_version,
    }
    card_data: dict[str, Any] = {
        "name": name,
        "description": description,
        "version": version,
        "supported_interfaces": [iface],
        "skills": [_as_dict(skill) for skill in skills],
        "default_input_modes": list(default_input_modes),
        "default_output_modes": list(default_output_modes),
        "capabilities": _as_dict(capabilities) or default_capabilities,
    }
  else:
    card_data = {
        "name": name,
        "description": description,
        "version": version,
        "url": url.rstrip("/"),
        "preferredTransport": protocol_binding,
        "skills": [_as_dict(skill) for skill in skills],
        "defaultInputModes": list(default_input_modes),
        "defaultOutputModes": list(default_output_modes),
        "protocolVersion": resolved_protocol_version,
        "supportsAuthenticatedExtendedCard": (
            supports_authenticated_extended_card
        ),
        "capabilities": _as_dict(capabilities) or default_capabilities,
    }

  # ``provider``/``security_schemes``/``doc_url`` are optional; omitted
  # fields fall back to SDK defaults.
  if provider is not None:
    card_data["provider"] = _as_dict(provider)
  if security_schemes:
    card_data["security_schemes"] = {
        key: _as_dict(scheme) for key, scheme in security_schemes.items()
    }
  if doc_url:
    card_data["documentation_url"] = doc_url
  return parse_agent_card(card_data)


# -----------------------------------------------------------------------------
# Client error & ClientCallContext shims
# -----------------------------------------------------------------------------
if IS_A2A_V1:
  # ``ClientCallContext`` moved from ``a2a.client.middleware`` to ``a2a.client.client``
  # ``A2AClientHTTPError`` is gone; use ``A2AClientError`` (carries status_code attr)
  from a2a.client.client import ClientCallContext as ClientCallContext
  from a2a.client.errors import A2AClientError as _A2AClientError

  A2A_HTTP_ERRORS = (_A2AClientError,)
else:
  from a2a.client.errors import A2AClientHTTPError
  from a2a.client.middleware import ClientCallContext as ClientCallContext  # type: ignore[assignment,no-redef]  # noqa: F401

  A2A_HTTP_ERRORS = (A2AClientHTTPError,)


# -----------------------------------------------------------------------------
# Agent-card URL helper
# -----------------------------------------------------------------------------
def agent_card_url(
    card: AgentCard,
    *,
    protocol_binding: str = TP_JSONRPC,
) -> Optional[str]:
  """Returns the RPC URL for a given protocol binding from an AgentCard.

  1.x: URL lives in ``supported_interfaces[i].url``; pick the first interface
  matching ``protocol_binding``, falling back to the first interface overall.
  ``protocol_binding`` must be a wire string (``'JSONRPC'``/``'HTTP+JSON'``),
  i.e. ``TP_*.value`` — not ``str(TP_*)``.
  0.3.x: URL is the top-level ``url`` field (``protocol_binding`` is unused).
  """
  if IS_A2A_V1:
    interfaces = list(card.supported_interfaces)
    if not interfaces:
      return None
    for iface in interfaces:
      if getattr(iface, "protocol_binding", None) == protocol_binding:
        matched_url: Optional[str] = iface.url
        return matched_url
    first_url: Optional[str] = interfaces[0].url
    return first_url
  else:
    del protocol_binding  # Only used by the v1.x path.
    return getattr(card, "url", None)


# -----------------------------------------------------------------------------
# Stream-item normalization
# -----------------------------------------------------------------------------
def stream_item_kind(item: Any) -> tuple[str, Any]:
  """Returns ``(kind, payload)`` for a stream item.

  1.x: ``send_message`` yields ``StreamResponse`` proto objects whose oneof
  ``payload`` is one of ``task``/``message``/``status_update``/
  ``artifact_update``.
  0.3.x: ``send_message`` yields ``tuple[Task, UpdateEvent | None]`` or a bare
  ``Message``.
  """
  if IS_A2A_V1:
    for kind in ("task", "message", "status_update", "artifact_update"):
      if item.HasField(kind):
        return kind, getattr(item, kind)
    raise ValueError(f"StreamResponse with no known payload field: {item!r}")
  else:
    if isinstance(item, tuple):
      task, update = item
      if update is None:
        return "task", task
      if isinstance(update, TaskStatusUpdateEvent):
        return "status_update", update
      if isinstance(update, TaskArtifactUpdateEvent):
        return "artifact_update", update
      raise ValueError(f"Unknown v0.3 update event: {update!r}")
    return "message", item


def make_stream_normalizer() -> Callable[[Any], Any]:
  """Returns a stateful normalizer that aggregates task state across a stream.

  ``send_message`` may deliver a task incrementally as a sequence of
  ``status_update``/``artifact_update`` items. The 0.3.x client aggregated these
  into a running ``Task`` (via ``ClientTaskManager``) so consumers always saw an
  accumulated task. This factory restores that behavior for 1.x: the returned
  callable holds a running ``Task`` and, for each item, returns the legacy shape
  with the *aggregated* task.
  """
  if not IS_A2A_V1:
    return lambda item: item

  from a2a.server.tasks.task_manager import append_artifact_to_task

  state: dict[str, Any] = {"task": None}

  def _ensure_task(payload: Any) -> Task:
    task: Optional[Task] = state["task"]
    if task is None:
      task = Task(
          id=getattr(payload, "task_id", "") or "",
          context_id=getattr(payload, "context_id", "") or "",
      )
      state["task"] = task
    return task

  def _snapshot(task: Task) -> Task:
    # Return a copy so each yielded tuple reflects the task state *at that
    # point* in the stream; later updates must not mutate already-yielded items.
    copy = Task()
    copy.CopyFrom(task)
    return copy

  def normalize(item: Any) -> Any:
    kind, payload = stream_item_kind(item)
    if kind == "message":
      return payload
    if kind == "task":
      # A full task state is already passed; use it as the aggregate.
      state["task"] = payload
      return (_snapshot(payload), None)
    task = _ensure_task(payload)
    if kind == "artifact_update":
      if payload.HasField("artifact"):
        append_artifact_to_task(task, payload)
    elif kind == "status_update":
      if payload.HasField("status"):
        # Accumulate the status message into history, matching
        # the 0.3.x ClientTaskManager
        if payload.status.HasField("message"):
          task.history.append(payload.status.message)
        task.status.CopyFrom(payload.status)
      if payload.HasField("metadata"):
        task.metadata.MergeFrom(payload.metadata)
    return (_snapshot(task), payload)

  return normalize


# -----------------------------------------------------------------------------
# send_message adapter
# -----------------------------------------------------------------------------
async def send_message(
    client: Any,
    *,
    request: Any,
    request_metadata: Optional[dict[str, Any]] = None,
    context: Any = None,
) -> AsyncGenerator[Any, None]:
  """Version-agnostic send_message invocation; yields raw stream items.

  1.x: ``send_message(request, *, context)`` takes no ``request_metadata``
  kwarg; metadata is embedded in ``SendMessageRequest.metadata`` (a proto
  ``Struct``).
  0.3.x: ``send_message`` accepts ``request_metadata`` directly as a kwarg.
  """
  if IS_A2A_V1:
    from a2a.types import SendMessageRequest
    from google.protobuf.struct_pb2 import Struct

    smr = SendMessageRequest()
    smr.message.CopyFrom(request)
    if request_metadata:
      smr.metadata.CopyFrom(ParseDict(request_metadata, Struct()))
    async for item in client.send_message(smr, context=context):
      yield item
  else:
    async for item in client.send_message(
        request=request, request_metadata=request_metadata, context=context
    ):
      yield item


# -----------------------------------------------------------------------------
# Client config builder
# -----------------------------------------------------------------------------
def make_client_config(*, httpx_client: Any, **kwargs: Any) -> Any:
  """Builds a version-correct A2A ``ClientConfig``.

  1.x: transport preference is set via ``supported_protocol_bindings`` (a list
  of wire strings); the default JSON-RPC + HTTP+JSON preference is applied
  unless the caller overrides it.
  0.3.x: transport preference is set via ``supported_transports`` (a list of
  ``TransportProtocol`` members, renamed to ``supported_protocol_bindings`` on
  1.x). ADK applies its defaults ``streaming=False``/``polling=False``.
  """
  if IS_A2A_V1:
    kwargs.setdefault(
        "supported_protocol_bindings",
        [
            TP_JSONRPC,
            TP_HTTP_JSON,
        ],
    )
    return A2AClientConfig(httpx_client=httpx_client, **kwargs)
  else:
    kwargs.setdefault("streaming", False)
    kwargs.setdefault("polling", False)
    kwargs.setdefault("supported_transports", [TP_JSONRPC, TP_HTTP_JSON])
    return A2AClientConfig(httpx_client=httpx_client, **kwargs)


def rebind_client_factory_httpx(factory: Any, httpx_client: Any) -> Any:
  """Returns a client factory bound to ``httpx_client``.

  0.3.x: the factory is rebuilt preserving its internal state — the existing
         ``ClientConfig`` (with the new httpx client swapped in via
         ``dataclasses.replace``), its ``consumers``, and any custom transports
         re-registered from ``_registry``. This keeps custom transports working.
  1.x:   the ``ClientFactory`` constructor only accepts ``config`` (no
         ``consumers``/registry to carry over), so a fresh factory is created
         with only the standard protocol bindings (custom transports are not
         carried over — intended behavior).
  """
  if IS_A2A_V1:
    return A2AClientFactory(
        config=make_client_config(httpx_client=httpx_client)
    )

  registry = factory._registry  # pylint: disable=protected-access
  new_factory = A2AClientFactory(
      config=dataclasses.replace(
          factory._config,  # pylint: disable=protected-access
          httpx_client=httpx_client,
      ),
      consumers=factory._consumers,  # pylint: disable=protected-access
  )
  for label, generator in registry.items():
    new_factory.register(label, generator)
  return new_factory


# -----------------------------------------------------------------------------
# HTTP hosting helper
# -----------------------------------------------------------------------------
def attach_a2a_routes_to_app(
    app: Any,
    *,
    agent_card: Any,
    agent_executor: Any,
    task_store: Any,
    enable_v0_3_compat: bool = True,
    push_config_store: Any = None,
    prefix: str = "",
) -> None:
  """Wires an A2A agent executor into an existing Starlette app.

  ``prefix`` mounts both the JSON-RPC route and the agent-card well-known route
  under ``{prefix}`` so that multiple agents hosted on one app do not collide
  on the default ``/`` RPC route and ``/.well-known/...`` card route.
  """
  if IS_A2A_V1:
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import create_agent_card_routes
    from a2a.server.routes import create_jsonrpc_routes

    handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
        push_config_store=push_config_store,
        agent_card=agent_card,
    )
    rpc_url = prefix or "/"
    # Mount the agent-card well-known route under the same prefix as the
    # RPC route so multiple agents hosted on one app don't collide on the
    # default ``/.well-known/agent-card.json`` path.
    card_url = (
        f"{prefix.rstrip('/')}/.well-known/agent-card.json"
        if prefix
        else "/.well-known/agent-card.json"
    )
    app.routes.extend([
        *create_agent_card_routes(agent_card, card_url=card_url),
        *create_jsonrpc_routes(
            handler,
            rpc_url,
            enable_v0_3_compat=enable_v0_3_compat,
        ),
    ])
  else:
    del enable_v0_3_compat  # Only consumed by the v1.x route factory.
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler

    try:
      from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
    except ImportError:
      AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"

    handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
        push_config_store=push_config_store,
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=handler
    )
    if prefix:
      a2a_app.add_routes_to_app(
          app,
          rpc_url=prefix,
          agent_card_url=f"{prefix.rstrip('/')}{AGENT_CARD_WELL_KNOWN_PATH}",
      )
    else:
      a2a_app.add_routes_to_app(app)


# -----------------------------------------------------------------------------
# Executor "Task-first" event shim
# -----------------------------------------------------------------------------
async def enqueue_submitted_signal(event_queue: Any, *, context: Any) -> None:
  """Publishes the initial "submitted" signal for a brand-new task.

  1.x:   The first enqueued event for a new task MUST be a ``Task`` (the server
         raises ``InvalidAgentResponseError`` otherwise). Publish a leading
         submitted ``Task`` and emit no redundant ``TaskStatusUpdateEvent``.
  0.3.x: The SDK tolerates a status-update-first stream, so emit a submitted
         ``TaskStatusUpdateEvent`` (historical behavior).

  No-op if the task already exists (``context.current_task`` is set).
  """
  if context.current_task:
    return
  if IS_A2A_V1:
    # 1.x requires a new task's first event to be a Task; otherwise the server
    # raises InvalidAgentResponseError.
    await event_queue.enqueue_event(
        make_task(
            id=context.task_id,
            context_id=context.context_id,
            status=make_task_status(TS_SUBMITTED),
            history=[context.message] if context.message else [],
        )
    )
  else:
    await event_queue.enqueue_event(
        make_task_status_update_event(
            task_id=context.task_id,
            context_id=context.context_id,
            status=make_task_status(TS_SUBMITTED, message=context.message),
            final=False,
        )
    )


# -----------------------------------------------------------------------------
# SecurityScheme builder
# -----------------------------------------------------------------------------
def make_api_key_scheme(*, name: str, location: str = "header") -> Any:
  """Builds an API-key SecurityScheme.

  1.x: SecurityScheme is a proto oneof; the sub-message field is ``location``.
  0.3.x: SecurityScheme wraps via ``root``; APIKeySecurityScheme uses ``in``
  (a Python keyword, passed as ``**{'in': location}``).
  """
  if IS_A2A_V1:
    return SecurityScheme(
        api_key_security_scheme=APIKeySecurityScheme(
            name=name,
            location=location,
        )
    )
  else:
    return SecurityScheme(
        root=APIKeySecurityScheme(name=name, **{"in": location})
    )


# -----------------------------------------------------------------------------
# Extension activation gate
# -----------------------------------------------------------------------------
def add_activated_extension(context: Any, uri: str) -> None:
  """Activates an extension on the request context if supported.

  In v1.x ``RequestContext.add_activated_extension`` no longer exists
  (extensions propagate via message metadata), so this becomes a no-op.
  In v0.3.x it is called directly.
  """
  if not IS_A2A_V1:
    context.add_activated_extension(uri)


# -----------------------------------------------------------------------------
# Version-agnostic object builders.
#
# These centralize every constructor whose kwargs diverge between 0.3.x and 1.x:
#   - Message: ``role`` is a pydantic enum / string on 0.3.x, a proto enum int
#     on 1.x. Callers may pass ``"user"``/``"agent"`` and we normalize.
#   - Task: 0.3.x has ``kind``; 1.x has no ``kind``/``metadata_version`` fields.
#   - Artifact: 0.3.x has ``artifact_type``; 1.x does not.
#   - TaskStatus: ``timestamp`` is an ISO string on 0.3.x, a proto Timestamp on
#     1.x; ``message`` must be assigned via ``CopyFrom`` on 1.x.
#   - TaskStatusUpdateEvent: 0.3.x has ``final``; 1.x infers finality from the
#     stream terminating.
# Production and test code should construct these types via the builders rather
# than calling the re-exported classes directly.
# -----------------------------------------------------------------------------


def _normalize_role(role: Any) -> Any:
  """Coerce a string role ('user'/'agent') to the version-correct value."""
  if isinstance(role, str):
    if role == "user":
      return ROLE_USER
    if role == "agent":
      return ROLE_AGENT
  return role


def make_message(
    *,
    message_id: str,
    role: Any,
    parts: Any = None,
    **kwargs: Any,
) -> Any:
  """Build a Message, normalizing ``role`` for the active SDK version."""
  return Message(
      message_id=message_id,
      role=_normalize_role(role),
      parts=list(parts) if parts is not None else [],
      **kwargs,
  )


def make_task(*, id: str, status: Any, **kwargs: Any) -> Any:
  """Build a Task, dropping 0.3-only kwargs (``kind``, ``metadata_version``) on 1.x."""
  if IS_A2A_V1:
    kwargs.pop("kind", None)
    kwargs.pop("metadata_version", None)
  return Task(id=id, status=status, **kwargs)


def make_artifact(*, artifact_id: str, parts: Any = None, **kwargs: Any) -> Any:
  """Build an Artifact, dropping the 0.3-only ``artifact_type`` kwarg on 1.x."""
  if IS_A2A_V1:
    kwargs.pop("artifact_type", None)
  if parts is not None:
    kwargs["parts"] = list(parts)
  return Artifact(artifact_id=artifact_id, **kwargs)


def make_task_status(
    state: Any, *, message: Any = None, timestamp: Any = None
) -> Any:
  """Build a TaskStatus with the correct timestamp type for each SDK version.

  0.3.x: ``timestamp`` is an ISO-format string.
  1.x:   ``timestamp`` is a ``google.protobuf.Timestamp`` message and
         ``message`` is assigned via ``CopyFrom``.

  ``timestamp`` may be passed as an ISO string or a datetime; if omitted, the
  current time is used.
  """
  if IS_A2A_V1:
    ts = TaskStatus(state=state)
    ts.timestamp.CopyFrom(_coerce_proto_timestamp(timestamp))
    if message is not None:
      ts.message.CopyFrom(message)
    return ts
  else:
    if timestamp is None:
      timestamp = datetime.now(timezone.utc).isoformat()
    elif not isinstance(timestamp, str):
      timestamp = timestamp.isoformat()
    kwargs = dict(state=state, timestamp=timestamp)
    if message is not None:
      kwargs["message"] = message
    return TaskStatus(**kwargs)


def _coerce_proto_timestamp(timestamp: Any) -> Any:
  """Return a proto Timestamp from a str/datetime/Timestamp/None (1.x only)."""
  if timestamp is None:
    return _make_proto_timestamp()
  if isinstance(timestamp, str):
    try:
      dt = datetime.fromisoformat(timestamp)
    except ValueError:
      dt = datetime.now(timezone.utc)
    return _make_proto_timestamp(dt)
  if hasattr(timestamp, "seconds"):  # already a proto Timestamp
    return timestamp
  return _make_proto_timestamp(timestamp)  # assume datetime


def make_task_status_update_event(
    task_id: Any,
    context_id: Any,
    status: Any,
    *,
    final: bool = True,
    metadata: Any = None,
) -> Any:
  """Build a TaskStatusUpdateEvent, omitting ``final`` on 1.x (field gone).

  0.3.x: ``TaskStatusUpdateEvent`` has a ``final`` bool field.
  1.x:   ``final`` field does not exist; finality is inferred from stream end.
  """
  if IS_A2A_V1:
    kwargs = dict(task_id=task_id, context_id=context_id, status=status)
    if metadata is not None:
      kwargs["metadata"] = metadata
    return TaskStatusUpdateEvent(**kwargs)
  else:
    kwargs = dict(
        task_id=task_id,
        context_id=context_id,
        status=status,
        final=final,
    )
    if metadata is not None:
      kwargs["metadata"] = metadata
    return TaskStatusUpdateEvent(**kwargs)


def set_event_metadata(event: Any, metadata: dict[str, Any]) -> None:
  """Set metadata on an A2A event.

  0.3.x: ``event.metadata`` is a plain dict (pydantic model field).
  1.x:   ``event.metadata`` is a proto ``Struct`` field; use ``CopyFrom``.
  """
  if not metadata:
    return
  if IS_A2A_V1:
    from google.protobuf.struct_pb2 import Struct

    event.metadata.CopyFrom(ParseDict(metadata, Struct()))
  else:
    event.metadata = metadata


def metadata_get(metadata: Any, key: str, default: Any = None) -> Any:
  """Reads a key from A2A metadata.

  0.3.x: metadata is a plain dict (has ``.get``).
  1.x:   metadata is a ``google.protobuf.Struct`` (no ``.get``; behaves like a
         mapping but must be accessed via ``in`` / indexing).
  """
  if not metadata:
    return default
  if IS_A2A_V1:
    # proto Struct supports ``in`` and item access, but not ``.get``.
    try:
      return metadata[key] if key in metadata else default
    except (TypeError, ValueError):
      return default
  return metadata.get(key, default)


def meta_to_dict(metadata: Any) -> dict[str, Any]:
  """Normalizes A2A metadata to a plain dict.

  0.3.x: metadata is already a plain dict.
  1.x:   metadata is a ``google.protobuf.Struct`` (has a ``DESCRIPTOR``); it is
         converted via ``MessageToDict``.
  """
  if metadata is None:
    return {}
  if IS_A2A_V1 and hasattr(metadata, "DESCRIPTOR"):
    return _proto_to_dict(metadata)
  if isinstance(metadata, dict):
    return metadata
  return {}


def set_struct_metadata(obj: Any, metadata: dict[str, Any]) -> None:
  """Assigns a metadata dict onto an A2A object's ``.metadata`` field.

  Works for any object carrying a ``metadata`` field (events, messages,
  artifacts).

  0.3.x: ``obj.metadata`` is a plain dict (pydantic field) → assign directly.
  1.x:   ``obj.metadata`` is a proto ``Struct`` field → use ``CopyFrom``.
  """
  if not metadata:
    return
  if IS_A2A_V1:
    from google.protobuf.struct_pb2 import Struct

    obj.metadata.CopyFrom(ParseDict(dict(metadata), Struct()))
  else:
    obj.metadata = dict(metadata)


def role_to_str(role: Any) -> str:
  """Maps an A2A ``Role`` to the GenAI content role string.

  Returns ``"user"`` for the user role and ``"model"`` for everything else
  (including the agent role and ``None``), matching the historical 0.3.x and
  1.x ``_a2a_role_to_content_role`` behavior.
  """
  return "user" if role == ROLE_USER else "model"


def normalize_message(msg: Any) -> Any:
  """Collapses an empty 1.x proto ``Message`` to ``None``.

  On 1.x ``TaskStatus.message`` is always a (possibly empty) proto ``Message``
  rather than ``None``; treat an empty one (no ``message_id`` set) as absent so
  downstream code that checks ``if message`` behaves like 0.3.x. On 0.3.x the
  value is returned unchanged.
  """
  if IS_A2A_V1 and msg is not None:
    # An empty proto Message has no fields set (message_id is empty).
    if hasattr(msg, "message_id") and not msg.message_id:
      return None
  return msg


def part_kind_label(part: Any) -> str:
  """Returns a human-readable kind label for a non-text/data Part (for logs).

  0.3.x: file parts are always wrapped as ``FilePart``.
  1.x:   ``Part`` is a flat proto message, so report its concrete class name.
  """
  return type(part).__name__ if IS_A2A_V1 else "FilePart"
