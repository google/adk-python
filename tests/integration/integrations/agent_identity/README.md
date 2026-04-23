# Integration tests for GCP Agent Identity Credentials service

Verifies OAuth flows using GCP Agent Identity Credentials service.

## Setup

To set up your environment for the first time, run the `uv` setup script:
```bash
cd open_source_workspace
./uv_setup.sh
```

Then, activate the virtual environment:
```bash
source .venv/bin/activate
```

Then, install test specific packages
```bash
pip install google-cloud-iamconnectorcredentials
```

## Run Tests
```bash
pytest -s tests/integration/integrations/agent_identity
```
