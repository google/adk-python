# DNS-AID

Decentralized agent discovery for Google ADK via DNS SVCB records — the
counterpart to [`agent_registry`](../agent_registry/), which uses Google
Cloud's centralized agent registry.

## What is DNS-AID?

DNS-AID encodes agent metadata into DNS SVCB records (RFC 9460) under the
naming convention `_<name>._<protocol>._agents.<domain>` (for example
`_chat._mcp._agents.example.com`). The
[`dns-aid`](https://pypi.org/project/dns-aid/) Python SDK (`dns-aid-core`)
handles SVCB encoding/decoding, backend zone updates, and discovery
queries; this integration is a thin ADK-shaped wrapper over it.

Discovery is decentralized by design: each operator publishes to whatever
DNS provider they already own (Route 53, Cloud DNS, Cloudflare, NS1,
Infoblox NIOS, on-prem BIND/Knot via RFC 2136). There is no single trust
root and no shared registry — which makes DNS-AID a natural fit for
multi-cloud, federated, and air-gapped deployments where a centralized
service is undesirable or unavailable.

## Install

```bash
pip install "google-adk[dns-aid]"
```

This pulls in `dns_aid>=0.18,<1` and the `[a2a]` extra (needed for the
A2A bridge).

## Quickstart — discover agents

```python
import asyncio
from google.adk.integrations.dns_aid import discover_agents


async def main():
    result = await discover_agents(domain='agents.example.com')
    for record in result['agents']:
        print(record['name'], record['protocol'], record['endpoint'])


asyncio.run(main())
```

Filter by protocol or by name with `protocol='a2a'` / `name='chat'`. Pass
`require_dnssec=True` to reject any result where end-to-end DNSSEC
validation fails.

## Quickstart — publish an agent

```python
from google.adk.integrations.dns_aid import publish_agent

await publish_agent(
    agent_name='chat',
    domain='agents.example.com',
    protocol='mcp',
    endpoint='chat.internal.example.com',
    port=443,
    backend_name='route53',
)
```

Omit `backend_name` to use whatever default the host resolver / `dns_aid`
configuration provides. See [Backend credentials](#backend-credentials)
for the supported values.

## Quickstart — bridge to RemoteA2aAgent

```python
from google.adk.integrations.dns_aid import discover_agents
from google.adk.integrations.dns_aid import remote_a2a_agent_from_record

result = await discover_agents(domain='agents.example.com', protocol='a2a')
agents = [
    remote_a2a_agent_from_record(record)
    for record in result['agents']
]
```

`remote_a2a_agent_from_record` is synchronous — it does no I/O.
`RemoteA2aAgent` fetches its agent card lazily on first invocation.

## Quickstart — bridge to McpToolset

```python
from google.adk.integrations.dns_aid import discover_agents
from google.adk.integrations.dns_aid import mcp_toolset_from_record

result = await discover_agents(domain='agents.example.com', protocol='mcp')
toolsets = [
    mcp_toolset_from_record(record)
    for record in result['agents']
]
```

Like its A2A counterpart, `mcp_toolset_from_record` is synchronous; the
underlying `McpToolset` connects on first use.

## Use as ADK FunctionTools

```python
from google.adk.agents import LlmAgent
from google.adk.integrations.dns_aid import get_dns_aid_tools

agent = LlmAgent(
    model='gemini-2.5-flash',
    tools=get_dns_aid_tools(backend_name='route53'),
)
```

`get_dns_aid_tools(backend_name=None)` returns FunctionTool wrappers for
`discover_agents`, `publish_agent`, and `unpublish_agent` with
`backend_name` pre-bound (so the LLM-facing schema does not include it).
Omit `backend_name` to fall back to the default `dns_aid` configuration.

## Backend credentials

`backend_name` must match the exact spelling that `dns_aid` expects
(hyphens, not underscores). Credentials live wherever the underlying
provider SDK looks for them:

| `backend_name` | Auth source | Required env vars / config |
|---|---|---|
| `route53` | AWS SDK default chain | `AWS_PROFILE` or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (or instance profile) |
| `cloudflare` | API token | `CLOUDFLARE_API_TOKEN` |
| `cloud-dns` | Google ADC | `GOOGLE_APPLICATION_CREDENTIALS` (or `gcloud auth application-default login`) |
| `ns1` | API key | `NS1_API_KEY` |
| `infoblox` / `nios` | NIOS API user | `INFOBLOX_HOST`, `INFOBLOX_USERNAME`, `INFOBLOX_PASSWORD` |
| `ddns` | RFC 2136 TSIG | `DDNS_TSIG_KEYFILE`, `DDNS_SERVER` |
| `mock` | n/a — in-memory only | (no creds; for tests) |

> Credentials are resolved by `dns_aid` itself, not by ADK. ADK does not
> read these env vars directly — it just hands the `backend_name` string
> to `dns_aid.backends.create_backend(...)`. If a publish or unpublish
> call fails with a backend-specific authentication error, check the
> `dns-aid-core` docs for that provider.

## Programmatic API

| Symbol | Description |
|---|---|
| `discover_agents` | Async; query DNS SVCB records under a domain. |
| `publish_agent` | Async; create SVCB + TXT records via the named backend. |
| `unpublish_agent` | Async; remove the records. Returns a structured status dict that distinguishes `not_found`, `permission_denied`, `backend_unavailable`, and `throttled`. |
| `get_dns_aid_tools` | Build the FunctionTool list for an `LlmAgent`. |
| `remote_a2a_agent_from_record` | Bridge to `RemoteA2aAgent` (protocol `a2a`). |
| `mcp_toolset_from_record` | Bridge to `McpToolset` (protocol `mcp`). |

## Future

The current integration ships discovery, publish, unpublish, and the two
bridges. Future revisions can layer:

- **Cap-doc fetch and verify** — `dns-aid-core` exposes `cap_fetcher` for
  pulling the SVCB-referenced capability document.
- **Trust verification** — DNSSEC, DANE, and `cap-sha256` checks via
  `dns_aid.core.validator`.
- **Policy enforcement** — Phase 6 of `dns-aid-core` provides
  `PolicyEvaluator` for evaluating policies referenced by `policy_uri`
  in the SVCB record.

## References

- DNS-AID spec: `draft-mozleywilliams-dnsop-dnsaid` (IETF Internet-Draft)
- `dns-aid-core`: the [`dns-aid`](https://pypi.org/project/dns-aid/)
  PyPI package
- A2A spec: [Agent-to-Agent protocol](https://google.github.io/A2A/)
- ADK FunctionTool: [function tools documentation](https://google.github.io/adk-docs/tools/function-tools/)
