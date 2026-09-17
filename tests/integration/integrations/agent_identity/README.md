# Integration tests for GCP Agent Identity Credentials service

Verifies OAuth flows using GCP Agent Identity Credentials service.

## Setup

To set up your environment for the first time, create a virtual environment
and install dependencies:
```bash
uv venv --python "python3.11" ".venv"
source .venv/bin/activate
uv sync --all-extras
```

Then, install test specific packages
```bash
pip install google-cloud-iamconnectorcredentials
```

## Run Tests
```bash
pytest -s tests/integration/integrations/agent_identity
```
