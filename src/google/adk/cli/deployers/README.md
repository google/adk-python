# CLI Deployment Tools for ADK Python

This directory contains CLI utilities for deploying ADK Python projects across multiple cloud providers and Docker environments.

## Overview

This generalizes the `adk deploy` command to support flexible deployment across various platforms such as:

- Google Cloud Platform (GCP)
- Local Docker environments
- Google Kubernetes Engine (GKE) - TODO

It provides a modular and extensible deployment interface using a factory pattern to handle cloud-specific deployment logic.

## Features

- **Generalized Dockerfile**: No hard-coded provider-specific variables.
- **Modular Deployers**: Cloud-specific deployers (e.g., `GCPDeployer`, `DockerDeployer`) simplify deployment logic.
- **Environment Variables Injection**:
  - Via `.env` files for local Docker development.
  - Directly via CLI flags for production environments.
- **Provider-Specific Arguments**: Use `--provider-args` to pass provider specific parameters.

## Usage

# Deploy on cloud provider (GCP)
```bash
adk deploy cloud_run --with_ui <agent-folder>
```

# Deploy on cloud provider (GCP) with environment variables
```bash
adk deploy cloud_run --with_ui --env GOOGLE_GENAI_USE_VERTEXAI=1 <agent-folder>
```

# Deploy locally using Docker

```bash
adk deploy docker --with_ui <agent-folder>
```

# Deploy locally using Docker with environment variables

```bash
adk deploy docker --with_ui --env GOOGLE_GENAI_USE_VERTEXAI=1 <agent-folder>
```

## Contributing

To add new CLI commands or support for additional cloud providers:

- Register the deployer in the deployer factory
- Implement a new deployer class
- Implement CLI entry points for the deployer
- Add corresponding test cases
- Update usage instructions and documentation