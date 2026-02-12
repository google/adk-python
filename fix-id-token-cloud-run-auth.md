# Fix: ID Token support for Cloud Run MCP auth

## Why

Cloud Run (and Cloud Functions, IAP, etc.) with "Require authentication" expects
an **OIDC ID Token**, not an OAuth2 access token. The
`ServiceAccountCredentialExchanger` always fetched an access token via
`credentials.token`, so calls to authenticated Cloud Run services failed with
`401 Unauthorized`.

There was no way for users to ask for an ID token — the only workaround was
monkey-patching the exchanger at runtime.

## What changed

### `ServiceAccount` model (`src/google/adk/auth/auth_credential.py`)

Two new optional fields:

- **`use_id_token`** (`bool`, default `False`) — when `True`, fetch an ID token
  instead of an access token.
- **`audience`** (`str`) — the target audience for the ID token, typically the
  service URL. Required when `use_id_token` is `True`.

`scopes` also got a default (`[]`) so callers using ID tokens don't have to pass
an empty list.

### `ServiceAccountCredentialExchanger` (`src/google/adk/tools/openapi_tool/auth/credential_exchangers/service_account_exchanger.py`)

Added a `_fetch_id_token` helper and a branch at the top of `exchange_credential`
that calls it when `use_id_token` is set. The existing access-token path is
untouched.

For default credentials it uses `google.oauth2.id_token.fetch_id_token()`. For
explicit service-account JSON keys it uses
`service_account.IDTokenCredentials.from_service_account_info()`.

### Tests

Four new tests covering: default-credential ID token, explicit-SA ID token,
missing audience validation, and fetch failure handling. All existing tests still
pass.

## Usage

```python
auth_credential = AuthCredential(
    auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
    service_account=ServiceAccount(
        use_default_credential=True,
        use_id_token=True,
        audience="https://my-service-xyz.us-central1.run.app/mcp",
    ),
)

mcp_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(url=MCP_URL),
    auth_scheme=HTTPBearer(bearerFormat="JWT"),
    auth_credential=auth_credential,
)
```

## PR Description (ready to paste)

### Summary

Cloud Run-protected MCP endpoints require an OIDC ID token, but service-account
exchange always returned an OAuth access token. This change adds an opt-in ID
token path for service account auth and keeps existing access-token behavior as
the default.

### Testing Plan

- Run formatter (`autoformat.sh` equivalent on Windows shell):
  - `.\.venv\Scripts\python.exe -m isort src\google\adk\auth\auth_credential.py src\google\adk\tools\openapi_tool\auth\credential_exchangers\service_account_exchanger.py tests\unittests\tools\openapi_tool\auth\credential_exchangers\test_service_account_exchanger.py`
  - `.\.venv\Scripts\python.exe -m pyink --config pyproject.toml src\google\adk\auth\auth_credential.py src\google\adk\tools\openapi_tool\auth\credential_exchangers\service_account_exchanger.py tests\unittests\tools\openapi_tool\auth\credential_exchangers\test_service_account_exchanger.py`
- Run focused unit tests:
  - `.\.venv\Scripts\python.exe -m pytest tests\unittests\tools\openapi_tool\auth\credential_exchangers\test_service_account_exchanger.py -q`
- Run related broader tests:
  - `.\.venv\Scripts\python.exe -m pytest tests\unittests\tools\openapi_tool\auth\ tests\unittests\auth\test_credential_manager.py tests\unittests\tools\mcp_tool\test_mcp_tool.py -q`

### Unit Test Evidence

- Focused exchanger tests: **11 passed in 1.59s**
- Broader auth + MCP tests: **126 passed, 336 warnings in 2.64s**

### Manual E2E Evidence (MCP + Cloud Run auth)

> Note: This local workspace currently has no Cloud Run endpoint configured
> (`MCP_URL` and `GOOGLE_CLOUD_PROJECT` are empty), so this section is prepared
> for final evidence capture in your Cloud environment.

Please attach in the PR:

1. A screenshot of `adk web` prompt/response where the MCP tool call succeeds.
2. Console logs proving successful authenticated call (no 401).
3. The exact agent config used (`use_id_token=True`, `audience=<cloud-run-url>`).

Suggested log snippet to include:

```text
HTTP Request: POST https://<service>.run.app/mcp "HTTP/1.1 200 OK"
... tool call result ...
```

### Docs Impact

- This introduces user-facing auth fields (`use_id_token`, `audience`) for
  service-account auth flow.
- Recommended follow-up: open/update a docs PR in `google/adk-docs` to document
  Cloud Run authenticated MCP setup with ID token usage.

### Review Request

- Request review from ADK auth/tooling maintainers.
- Suggested focus areas:
  - Backward compatibility of service-account access-token flow.
  - Correctness of ID-token exchange for ADC and explicit service-account keys.
  - Error messaging when `audience` is missing.
