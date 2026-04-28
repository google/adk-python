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

"""DNS-AID tools for Google ADK.

DNS-AID is the decentralized counterpart to ``integrations/agent_registry``:
it lets agents discover and publish themselves via DNS SVCB records rather
than a centralized cloud service. This module exposes the discover, publish
and unpublish operations as ADK ``FunctionTool`` callables so an LLM-driven
agent can invoke them directly.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import Field

try:
  import dns_aid
  from dns_aid.backends import create_backend
  from dns_aid.backends import VALID_BACKEND_NAMES
  from dns_aid.core.models import AgentRecord
  from dns_aid.core.models import Protocol
except ImportError as e:
  raise ImportError(
      'dns_aid is not installed. Please install it with '
      '`pip install "google-adk[dns-aid]"`.'
  ) from e

logger = logging.getLogger('google_adk.' + __name__)

# RFC 1035 DNS-label form: 1-63 chars, alphanumeric or hyphen, no leading/
# trailing hyphen. Used by FunctionTool to constrain LLM-generated names.
_DNS_LABEL_PATTERN = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'

# Pinning the Literal in the tool schema lets the LLM choose from an enum
# rather than emit free-form strings.
ProtocolLiteral = Literal['a2a', 'mcp', 'https']

# create_backend lowercases its input but does NOT translate underscore to
# hyphen, so the literal here must match dns_aid's exact spelling
# (e.g. 'cloud-dns' has a hyphen). Build dynamically from the package's
# canonical set so adding a backend in dns_aid auto-propagates here.
BackendNameLiteral = Literal[  # type: ignore[valid-type]
    tuple(sorted(VALID_BACKEND_NAMES))
]

__all__ = [
    'discover_agents',
    'get_dns_aid_tools',
    'publish_agent',
    'unpublish_agent',
]


def _resolve_backend(backend_name: str | None) -> Any:
  """Returns a dns_aid backend instance, or None to use the default.

  Args:
    backend_name: Backend identifier (e.g. ``route53``, ``cloud-dns``,
      ``infoblox``). If empty/None, dns_aid's default backend is used.

  Returns:
    A ``DNSBackend`` instance from ``dns_aid.backends.create_backend`` when
    ``backend_name`` is non-empty, otherwise ``None``.
  """
  if not backend_name:
    return None
  return create_backend(backend_name)


async def discover_agents(
    domain: str,
    protocol: ProtocolLiteral | None = None,
    name: str | None = None,
    require_dnssec: bool = False,
) -> dict[str, Any]:
  """Discovers DNS-AID agents published under a domain.

  Issues a DNS SVCB query of the form
  ``_{name}._{protocol}._agents.{domain}`` (with wildcards when ``name`` or
  ``protocol`` are omitted) and returns the structured discovery result.

  Args:
    domain: DNS zone to query (e.g. ``agents.example.com``).
    protocol: Optional protocol filter; one of ``a2a``, ``mcp``, ``https``.
    name: Optional specific agent label to look up.
    require_dnssec: If True, the discovery result is rejected unless DNSSEC
      validation succeeds end-to-end.

  Returns:
    A ``dict`` whose keys mirror the dns_aid ``DiscoveryResult`` model
    (``agents``, ``query_name``, ``dnssec_valid``, ``errors``, etc.).
    The ADK ``FunctionTool`` serializer surfaces this directly to the model.
  """
  logger.info('discover_agents domain=%s protocol=%s', domain, protocol)
  result = await dns_aid.discover(
      domain=domain,
      protocol=protocol,
      name=name,
      require_dnssec=require_dnssec,
  )
  return result.model_dump()


async def publish_agent(
    agent_name: Annotated[
        str,
        Field(
            pattern=_DNS_LABEL_PATTERN,
            description=(
                'Agent identifier in RFC 1035 DNS-label form (1-63 chars, '
                'alphanumeric or hyphen, no leading/trailing hyphen).'
            ),
        ),
    ],
    domain: str,
    endpoint: Annotated[
        str,
        Field(
            min_length=1,
            description='Hostname or URL where the agent is reachable.',
        ),
    ] = '',
    protocol: ProtocolLiteral = 'mcp',
    port: Annotated[int, Field(ge=1, le=65535)] = 443,
    capabilities: list[str] | None = None,
    version: str = '1.0.0',
    description: str | None = None,
    ttl: Annotated[
        int,
        Field(
            ge=60,
            description='DNS TTL in seconds. Most providers reject TTL < 60.',
        ),
    ] = 3600,
    backend_name: BackendNameLiteral | None = None,
) -> dict[str, Any]:
  """Publishes an agent record to DNS via dns_aid.

  Writes an SVCB record at ``_{agent_name}._{protocol}._agents.{domain}``
  pointing at ``endpoint:port``. The configured backend
  (``route53``, ``cloud-dns``, ``infoblox``, ...) handles the underlying
  zone update.

  Args:
    agent_name: DNS-label form identifier for the agent.
    domain: Zone to publish under (e.g. ``agents.example.com``).
    endpoint: Reachable hostname or URL for the agent. Required.
    protocol: Protocol the endpoint speaks; one of ``a2a``, ``mcp``,
      ``https``.
    port: TCP port the endpoint listens on.
    capabilities: Optional list of capability strings advertised in the
      record.
    version: Agent version string.
    description: Free-form human description.
    ttl: DNS TTL in seconds; values below 60 are commonly rejected by
      providers.
    backend_name: Backend identifier; falls back to dns_aid's default when
      None.

  Returns:
    A ``dict`` mirroring the dns_aid ``PublishResult`` model.

  Raises:
    ValueError: If ``endpoint`` is empty or ``backend_name`` is not in the
      set of supported backends.
  """
  # endpoint is logically required, but pydantic Annotated semantics make
  # mixing required-positional with later kw-only defaults awkward; enforce
  # it at the body so the FunctionTool schema still flags it via min_length.
  if not endpoint:
    raise ValueError('endpoint must not be empty')

  # The Literal type alone isn't enforceable at runtime — the LLM may emit
  # an unknown string. Validate explicitly.
  if backend_name and backend_name not in VALID_BACKEND_NAMES:
    raise ValueError(
        f'unknown backend_name {backend_name!r}; expected one of '
        f'{sorted(VALID_BACKEND_NAMES)}'
    )

  logger.info(
      'publish_agent agent_name=%s domain=%s protocol=%s endpoint=%s',
      agent_name,
      domain,
      protocol,
      endpoint,
  )
  backend = _resolve_backend(backend_name)
  try:
    result = await dns_aid.publish(
        name=agent_name,
        domain=domain,
        protocol=protocol,
        endpoint=endpoint,
        port=port,
        capabilities=capabilities,
        version=version,
        description=description,
        ttl=ttl,
        backend=backend,
    )
  except Exception as e:
    logger.warning(
        'publish_agent failed agent_name=%s domain=%s error=%s',
        agent_name,
        domain,
        e,
    )
    raise
  logger.info(
      'publish_agent succeeded agent_name=%s domain=%s', agent_name, domain
  )
  return result.model_dump()


def _classify_unpublish_error(exc: BaseException) -> str:
  """Maps a dns_aid exception to a coarse status string.

  dns_aid does not yet expose a stable typed-error taxonomy, so we
  conservatively match on the exception class name. This is a known
  limitation pending dns-aid-core's error-type stabilization.

  Args:
    exc: The exception raised by ``dns_aid.unpublish``.

  Returns:
    One of ``permission_denied``, ``not_found``, ``throttled``,
    ``backend_unavailable``, or ``error``.
  """
  cls = exc.__class__.__name__
  if 'Permission' in cls or 'Auth' in cls:
    return 'permission_denied'
  if 'NotFound' in cls:
    return 'not_found'
  if 'Throttle' in cls or 'RateLimit' in cls:
    return 'throttled'
  if 'Connection' in cls or 'Network' in cls or 'Backend' in cls:
    return 'backend_unavailable'
  return 'error'


async def unpublish_agent(
    agent_name: Annotated[str, Field(pattern=_DNS_LABEL_PATTERN)],
    domain: str,
    protocol: ProtocolLiteral = 'mcp',
    backend_name: BackendNameLiteral | None = None,
) -> dict[str, Any]:
  """Removes an agent's SVCB record via dns_aid.

  Distinguishes failure modes (not-found, permission, throttling, backend
  unavailability) so the calling agent can react appropriately rather than
  collapsing every failure to a generic ``not found``.

  Args:
    agent_name: DNS-label identifier of the agent to remove.
    domain: Zone the record lives in.
    protocol: Protocol of the published record; defaults to ``mcp``.
    backend_name: Backend identifier; falls back to dns_aid's default when
      None.

  Returns:
    On success, ``{'success': True, ...}``; on failure, a dict with a
    ``status`` field that is one of ``not_found``, ``permission_denied``,
    ``backend_unavailable``, ``throttled``, or ``error``, plus a
    human-readable ``message`` and the ``agent_name`` / ``domain`` echoed
    back. Note: this status taxonomy is best-effort because dns_aid does
    not yet expose typed exceptions; class-name substring matching is used.
  """
  if backend_name and backend_name not in VALID_BACKEND_NAMES:
    raise ValueError(
        f'unknown backend_name {backend_name!r}; expected one of '
        f'{sorted(VALID_BACKEND_NAMES)}'
    )

  logger.info(
      'unpublish_agent agent_name=%s domain=%s protocol=%s',
      agent_name,
      domain,
      protocol,
  )
  backend = _resolve_backend(backend_name)
  try:
    result = await dns_aid.unpublish(
        name=agent_name,
        domain=domain,
        protocol=protocol,
        backend=backend,
    )
  except Exception as e:
    status = _classify_unpublish_error(e)
    logger.warning(
        'unpublish_agent failed agent_name=%s domain=%s status=%s error=%s',
        agent_name,
        domain,
        status,
        e,
    )
    return {
        'success': False,
        'status': status,
        'message': str(e),
        'agent_name': agent_name,
        'domain': domain,
    }

  if not result:
    logger.warning(
        'unpublish_agent not_found agent_name=%s domain=%s',
        agent_name,
        domain,
    )
    return {
        'success': False,
        'status': 'not_found',
        'message': f'no SVCB record found for {agent_name}.{protocol}.{domain}',
        'agent_name': agent_name,
        'domain': domain,
    }

  logger.info(
      'unpublish_agent succeeded agent_name=%s domain=%s',
      agent_name,
      domain,
  )
  return {
      'success': True,
      'status': 'ok',
      'agent_name': agent_name,
      'domain': domain,
  }


def _bind_backend(fn: Any, backend_name: str | None) -> Any:
  """Binds ``backend_name`` to ``fn`` while preserving its signature.

  Fixes review issue #14 (closure signature drift): rather than manually
  redeclaring each parameter in an inner def — which silently desyncs
  whenever dns_aid grows a new keyword (e.g. Phase 6 ``policy_uri``) — we
  splice ``backend_name`` out of the signature with ``inspect`` and let
  FunctionTool introspect the wrapped callable directly via
  ``__signature__``.
  """
  sig = inspect.signature(fn)
  params = [p for n, p in sig.parameters.items() if n != 'backend_name']
  new_sig = sig.replace(parameters=params)

  @functools.wraps(fn)
  async def wrapped(*args: Any, **kwargs: Any) -> Any:
    return await fn(*args, backend_name=backend_name, **kwargs)

  wrapped.__signature__ = new_sig  # type: ignore[attr-defined]
  return wrapped


def get_dns_aid_tools(
    backend_name: BackendNameLiteral | None = None,
) -> list[Any]:
  """Returns DNS-AID operations as a list of ADK ``FunctionTool`` instances.

  Each tool is a thin wrapper around ``discover_agents``, ``publish_agent``,
  and ``unpublish_agent`` with ``backend_name`` pre-bound, so the LLM-facing
  schema only contains the operation parameters.

  Args:
    backend_name: Optional backend identifier to bind into every returned
      tool. Must be one of the names in
      ``dns_aid.backends.VALID_BACKEND_NAMES`` (e.g. ``route53``,
      ``cloud-dns``, ``infoblox``).

  Returns:
    A ``list[FunctionTool]`` ready to pass to an ADK agent's ``tools=``.
  """
  if backend_name and backend_name not in VALID_BACKEND_NAMES:
    raise ValueError(
        f'unknown backend_name {backend_name!r}; expected one of '
        f'{sorted(VALID_BACKEND_NAMES)}'
    )

  # Lazy import to keep module import-time cheap and avoid pulling
  # FunctionTool into environments that only need the bare functions.
  from google.adk.tools import FunctionTool

  return [
      FunctionTool(_bind_backend(discover_agents, backend_name)),
      FunctionTool(_bind_backend(publish_agent, backend_name)),
      FunctionTool(_bind_backend(unpublish_agent, backend_name)),
  ]
