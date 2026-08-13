# OpenSandbox environment verification

Verification recorded on 2026-08-13 against upstream commit
`2e878ed4120b1009080ec4bd189cf8b436d03ccf`.

## Automated checks

```bash
uv run pytest \
  tests/unittests/integrations/opensandbox/test_opensandbox_environment.py -q
./scripts/update_constraints.sh --check
```

Result: 52 tests passed, and all Python 3.10-3.14 constraint files were
up to date.

## Live boundary

The environment was exercised end to end against a local OpenSandbox server
using ADK's default `python:3.11` image. Agent-driven command execution and
workspace file operations passed.

The same agent flow was exercised against a private HTTPS OpenSandbox
deployment using server-proxy mode. Its endpoint, credential, and private image
are intentionally not recorded here.

These live checks require an OpenSandbox server and model credentials, so they
are not part of the default unit-test suite.
