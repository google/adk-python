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

"""Turn an ``LlmRequest`` into a ``generateContent`` request body.

``LlmRequest.config`` is a ``types.GenerateContentConfig``: one flat namespace
of 35 fields. On the wire those fields go to three different places. Some sit at
the top level of the request, most belong inside ``generationConfig``, and a few
mean something only to the client library and are rejected by the endpoint.

Nothing on the type says which is which, and getting it wrong does not raise.
Bury ``tools`` inside ``generationConfig`` and the call still returns 200 with a
well formed response in it. The tools are simply absent, so the model answers
from memory instead of calling anything, and the failure looks like a model that
chose not to call the tool.

Rather than keep a hand written list of which field goes where, this delegates
to the same converter ``google-genai`` uses for its own built-in models. A hand
written list is a second copy of the mapping and it goes stale the moment
``GenerateContentConfig`` gains a field: every custom ``BaseLlm`` in the wild
then starts silently dropping that field. Delegating means there is one copy and
it moves when the type moves.
"""

from __future__ import annotations

from typing import Any
from typing import Optional

from google.genai import models as _genai_models
from google.genai import types

__all__ = ["to_generate_content_body"]


class _ApiMode:
  """The one thing the genai converters need from an API client.

  ``_GenerateContentParameters_to_vertex`` touches exactly one attribute on the
  client it is handed, ``vertexai``, by way of ``t_model``. Passing a real
  client works and is preferred when the caller has one. This stands in when the
  caller does not, so the helper stays usable from a ``BaseLlm`` that talks REST
  directly and never builds a genai client at all.
  """

  __slots__ = ("vertexai",)

  def __init__(self, vertexai: bool):
    self.vertexai = vertexai


def to_generate_content_body(
    llm_request: Any,
    *,
    vertexai: bool = True,
    api_client: Optional[Any] = None,
) -> dict[str, Any]:
  """Return the request body for ``llm_request``.

  Args:
    llm_request: the ``LlmRequest`` to convert.
    vertexai: whether the body is bound for Vertex AI rather than the Gemini
      Developer API. The two differ in more than the URL: some config fields
      exist on one and not the other, so this is not cosmetic. Ignored when
      ``api_client`` is given.
    api_client: a genai API client, if the caller has one. Preferred over
      ``vertexai`` because it is the real thing rather than a stand-in.

  Returns:
    A dict ready to send as the request body. The model name is not in it; the
    caller already knows the model, and on Vertex it belongs in the URL.

  Raises:
    ValueError: if a config field is not valid for the target API. The message
      names the ``GenerateContentConfig`` field so the error points at the line
      that set it rather than at the converter.
  """
  client = api_client if api_client is not None else _ApiMode(vertexai)

  is_vertex = getattr(client, "vertexai", True)

  params = types._GenerateContentParameters(
      model=llm_request.model,
      contents=llm_request.contents,
      config=llm_request.config,
  )

  convert = (
      _genai_models._GenerateContentParameters_to_vertex
      if is_vertex
      else _genai_models._GenerateContentParameters_to_mldev
  )

  try:
    body = convert(client, params)
  except ValueError as exc:
    raise ValueError(_annotate(str(exc), vertexai=is_vertex)) from exc

  # The converters return the model under a private "_url" key, because the
  # caller is normally the genai client, which uses it to build the path. It is
  # routing information, not body content. Sending it is a 400.
  body.pop("_url", None)
  return body


def _annotate(message: str, *, vertexai: bool) -> str:
  """Point a genai validation error back at the ADK field that caused it.

  The converters raise messages that start with the offending field name, e.g.
  "enable_enhanced_civic_answers parameter is only supported in ...". That is
  accurate and still hard to act on from a ``BaseLlm``, because the reader has
  no reason to connect it to the config they built several layers up.
  """
  first = message.split(" ", 1)[0]
  if first in types.GenerateContentConfig.model_fields:
    mode = "Vertex AI" if vertexai else "the Gemini Developer API"
    return (
        f"{message} (set as LlmRequest.config.{first}; this request targets"
        f" {mode}.)"
    )
  return message
