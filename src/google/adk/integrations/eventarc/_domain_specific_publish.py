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

"""Domain-specific publish tool factory for Eventarc."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import keyword
from typing import Any
from typing import Callable

from google.adk.agents.context import Context
from google.adk.tools.google_tool import GoogleTool
import google.auth.credentials
import pydantic

from ._config import EventarcToolConfig
from ._message_tool import publish_message


class MissingSentinel:
  pass


MISSING = MissingSentinel()


class OmitSentinel:
  pass


OMIT = OmitSentinel()


@dataclass
class AgentProvided:
  """Indicates that a CloudEvent attribute should be provided by the LLM."""

  description: str
  default: Any | OmitSentinel | MissingSentinel = MISSING


AttributeBinding = str | Callable[[Any], str] | AgentProvided
OptionalAttributeBinding = (
    str
    | Callable[[Any], str | OmitSentinel]
    | AgentProvided
    | OmitSentinel
    | MissingSentinel
    | None
)
CustomAttributeBinding = (
    str | Callable[[Any], str | OmitSentinel] | AgentProvided | OmitSentinel
)
SpecVersionBinding = (
    str | Callable[[Any], str] | AgentProvided | MissingSentinel | None
)


@dataclass
class CloudEventAttributesBinding:
  """Configuration for binding CloudEvent attributes to static values, lambdas, or AgentProvided fields."""

  type: AttributeBinding
  source: AttributeBinding
  datacontenttype: OptionalAttributeBinding = MISSING
  subject: OptionalAttributeBinding = MISSING
  time: OptionalAttributeBinding = MISSING
  specversion: SpecVersionBinding = MISSING
  id: OptionalAttributeBinding = MISSING
  custom_attributes: dict[str, CustomAttributeBinding] | None = None


def build_domain_specific_tool(
    toolset: Any,  # Typed as Any to avoid circular import with EventarcToolset
    name: str,
    description: str,
    bus: AttributeBinding,
    ce_attributes_binding: CloudEventAttributesBinding,
    payload_schema: type[pydantic.BaseModel] | None = None,
) -> GoogleTool:
  """Dynamically builds a GoogleTool wrapping publish_message with specific bindings."""

  # 1. Validation
  mandatory_fields = ["type", "source"]
  for field in mandatory_fields:
    val = getattr(ce_attributes_binding, field)
    if val is MISSING:
      raise TypeError(
          f"CloudEventAttributesBinding requires '{field}' to be provided."
      )
    if val is OMIT:
      raise TypeError(
          f"CloudEvent field '{field}' is mandatory and cannot be OMIT."
      )
    if val is None:
      raise TypeError(
          f"CloudEvent field '{field}' is mandatory and cannot be None."
      )

  if bus is OMIT:  # type: ignore[comparison-overlap]
    raise TypeError("The 'bus' parameter is mandatory and cannot be OMIT.")
  if bus is MISSING:  # type: ignore[comparison-overlap]
    raise TypeError("The 'bus' parameter is mandatory and must be provided.")
  if bus is None:
    raise TypeError("The 'bus' parameter is mandatory and cannot be None.")

  reserved_attributes = {
      "type",
      "source",
      "datacontenttype",
      "subject",
      "time",
      "specversion",
      "id",
  }

  if ce_attributes_binding.custom_attributes:
    for k, v in ce_attributes_binding.custom_attributes.items():
      if k in reserved_attributes:
        raise ValueError(
            f"Custom attribute '{k}' shadows a standard CloudEvent attribute."
        )
      if not k.isalnum() or not k.islower():
        raise ValueError(
            f"Custom attribute '{k}' is invalid. CloudEvent attributes MUST "
            "consist of lower-case letters ('a' to 'z') or digits ('0' to '9')."
        )
      if v is MISSING:  # type: ignore[comparison-overlap]
        raise TypeError(f"Custom attribute '{k}' cannot be MISSING.")

  # Collect parameters for the LLM tool signature
  parameters = []
  annotations: dict[str, Any] = {}
  agent_provided_keys = set()

  # Maps the safe parameter name exposed to the LLM to the actual CloudEvent key.
  param_name_to_ce_key = {}

  def add_agent_provided(ce_key: str, ap: AgentProvided) -> None:
    agent_provided_keys.add(ce_key)

    # Convert the CloudEvent key to a safe Python identifier for the tool signature
    param_name = ce_key
    # We will append an underscore to keywords and 'self'/'cls'.
    if keyword.iskeyword(ce_key) or ce_key in ("self", "cls"):
      param_name = ce_key + "_"

    # If the attribute starts with a digit, it is not a valid Python identifier.
    if not param_name.isidentifier():
      param_name = "_" + param_name

    param_name_to_ce_key[param_name] = ce_key

    is_optional = not isinstance(ap.default, MissingSentinel)

    # Determine the type annotation for the parameter
    param_type: Any = str
    if is_optional:
      param_type = str | None

    annotations[param_name] = param_type

    # Build inspect.Parameter
    if is_optional:
      # If default is a Callable or OMIT, from LLM's perspective it is optional, but it passes None.
      # We don't expose the Callable/OMIT directly in the schema as default, we expose None.
      # If default is a static string, we can expose it.
      if callable(ap.default) or isinstance(ap.default, OmitSentinel):
        default_val = None
      else:
        default_val = ap.default

      parameters.append(
          inspect.Parameter(
              name=param_name,
              kind=inspect.Parameter.KEYWORD_ONLY,
              annotation=param_type,
              default=default_val,
          )
      )
    else:
      parameters.append(
          inspect.Parameter(
              name=param_name,
              kind=inspect.Parameter.KEYWORD_ONLY,
              annotation=param_type,
          )
      )

  # Process bindings to find AgentProvided
  if isinstance(bus, AgentProvided):
    add_agent_provided("bus", bus)

  for field in reserved_attributes:
    val = getattr(ce_attributes_binding, field)
    if isinstance(val, AgentProvided):
      add_agent_provided(field, val)

  if ce_attributes_binding.custom_attributes:
    for k, v in ce_attributes_binding.custom_attributes.items():
      if isinstance(v, AgentProvided):
        add_agent_provided(k, v)

  if payload_schema is not None:
    annotations["event_data"] = payload_schema
    parameters.append(
        inspect.Parameter(
            name="event_data",
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=payload_schema,
        )
    )

  # Standard tool parameters
  parameters.append(
      inspect.Parameter(
          name="credentials",
          kind=inspect.Parameter.KEYWORD_ONLY,
          annotation=google.auth.credentials.Credentials | None,
          default=None,
      )
  )
  annotations["credentials"] = google.auth.credentials.Credentials | None

  parameters.append(
      inspect.Parameter(
          name="settings",
          kind=inspect.Parameter.KEYWORD_ONLY,
          annotation=EventarcToolConfig,
          default=None,
      )
  )
  annotations["settings"] = EventarcToolConfig

  parameters.append(
      inspect.Parameter(
          name="tool_context",
          kind=inspect.Parameter.KEYWORD_ONLY,
          annotation=Context,
          default=None,
      )
  )
  annotations["tool_context"] = Context

  # 3. Create the runtime wrapper
  async def _execute(**kwargs: Any) -> dict[str, Any]:
    payload = kwargs.get("event_data", None)
    credentials = kwargs.get("credentials")
    settings: EventarcToolConfig = (
        kwargs.get("settings") or EventarcToolConfig()
    )

    publish_kwargs: dict[str, Any] = {
        "credentials": credentials,
        "settings": settings,
    }
    if payload is not None:
      # We assume payload is a fully instantiated Pydantic model at this point
      # due to FunctionTool's preprocessing. We pass it as a dict/json to publish_message.
      publish_kwargs["data"] = payload.model_dump(exclude_unset=True)

    def resolve_attr(key: str, binding: Any, is_mandatory: bool) -> Any:
      if binding is MISSING:
        return None

      if isinstance(binding, AgentProvided):
        # The LLM provided it, or omitted it
        # We need to find the tool parameter name for this key
        # Reverse lookup for the parameter name
        param_name = next(
            (p for p, k in param_name_to_ce_key.items() if k == key), key
        )
        llm_val = kwargs.get(param_name, None)
        if llm_val is not None:
          val = llm_val
        else:
          if isinstance(binding.default, MissingSentinel):
            # Should have been caught by the tool wrapper missing args validation,
            # but just in case:
            raise ValueError(
                f"Agent did not provide mandatory attribute '{key}'"
            )
          val = binding.default
      else:
        val = binding

      # Evaluate lambdas
      if callable(val):
        val = val(payload)

      if val is OMIT:
        if is_mandatory:
          raise ValueError(
              f"Mandatory CloudEvent attribute '{key}' cannot evaluate to OMIT."
          )
        return OMIT

      return val

    # Resolve bus separately
    bus_val = resolve_attr("bus", bus, is_mandatory=True)
    if bus_val is OMIT or bus_val is None:
      raise ValueError(
          "Mandatory attribute 'bus' cannot evaluate to None or OMIT."
      )
    publish_kwargs["bus"] = bus_val

    # Resolve reserved attributes
    for field in reserved_attributes:
      is_mandatory = field in mandatory_fields
      val = resolve_attr(
          field, getattr(ce_attributes_binding, field), is_mandatory
      )
      if val is not OMIT and val is not None:
        publish_kwargs[field] = val

    # Resolve custom attributes
    custom_attr_dict = {}
    if ce_attributes_binding.custom_attributes:
      for k, v in ce_attributes_binding.custom_attributes.items():
        val = resolve_attr(k, v, is_mandatory=False)
        if val is not OMIT and val is not None:
          custom_attr_dict[k] = val

    if custom_attr_dict:
      publish_kwargs["custom_attributes"] = custom_attr_dict

    return await publish_message(**publish_kwargs)  # type: ignore[arg-type]

  # Attach signature and annotations
  _execute.__signature__ = inspect.Signature(parameters=parameters)  # type: ignore[attr-defined]
  _execute.__annotations__ = annotations  # type: ignore[attr-defined]
  _execute.__name__ = name
  _execute.__doc__ = description

  return GoogleTool(
      func=_execute,
      credentials_config=toolset.credentials_config,
      tool_settings=toolset.tool_config,
  )
