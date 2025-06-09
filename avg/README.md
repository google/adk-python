# Purpose: Main README for the AVG Financial Expert Demo project.
# Based on avg.json content_prompt for main_readme_md.

# Agent Versatility Gear (AVG) - Financial Expert Demo

## Overview
This project demonstrates the Agent Versatility Gear (AVG) toolkit by implementing a Financial Expert agent.
AVG is designed to be a modular, ADK-agnostic wrapper toolkit that simplifies the creation and orchestration of sophisticated AI agents.
The Financial Expert agent showcases capabilities like leveraging diverse knowledge sources, dynamic information retrieval, complex reasoning, and generating financial insights. This specific implementation uses the Google Agent Development Kit (ADK).

## Features
- Modular design with separated components for knowledge, pipelines, and agent wrapping.
- Configuration-driven agent abilities and knowledge sources.
- Demonstration of complex reasoning for financial tasks.
- (Placeholder for more specific features as developed)

## Architecture
The project utilizes AVG components:
- **`adk_avg_toolkit`**: Core AVG classes like `KnowledgeConfigLoader`, `ExpertisePipelineManager`, and `BaseExpertWrapper`.
- **`financial_expert_agent`**: The Financial Expert agent implementation, including its main logic, specialized expert wrappers, custom ADK tools, and configurations.
- **`demo`**: Scripts and data for running a command-line demonstration.

The Financial Expert agent uses these AVG components along with Google ADK to perform its tasks.

## Setup Instructions
1.  **Clone the repository.**
2.  **Python Version**: Ensure Python 3.9+ is installed.
3.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Environment Variables**:
    Copy `.env.example` to `.env` and fill in the necessary details (e.g., GCP Project ID, region).
    ```bash
    cp .env.example .env
    # Edit .env with your specific configurations
    ```
6.  **Google Cloud Authentication**:
    If using Google Cloud services (like Vertex AI), authenticate with gcloud:
    ```bash
    gcloud auth application-default login
    ```

## How to Run the Demo
Execute the demo script:
```bash
python demo/run_demo.py
```
(Further instructions might be added as `run_demo.py` is developed).

## How to Use ADK Web UI (Conceptual)
If specific agents within this project are exposed via an ADK web server (e.g., using `adk web run main_agent:FinancialAdvisorMasterAgent`), they could be interacted with through the ADK Web UI. This typically involves running the ADK web server and navigating to the provided local URL.

## Project Structure Overview
- `avg/adk_avg_toolkit/`: Core AVG toolkit components.
- `avg/financial_expert_agent/`: Financial expert agent specific implementations.
  - `configs/`: Configuration files like `financial_knowledge_config.yaml`.
  - `data_schemas/`: JSON schemas for data structures.
  - `models/`: Placeholder for ML models.
- `avg/demo/`: Demo application files.
- `avg/requirements.txt`: Project dependencies.
- `avg/README.md`: This README file.
- `avg/.env.example`: Example environment variable setup.

## Configuration
The primary configuration for the financial expert is `avg/financial_expert_agent/configs/financial_knowledge_config.yaml`. This file defines knowledge sources, ingestion pipelines, vector stores, and other aspects crucial for the agent's operation.

## Disclaimer
This project is a demonstration and should not be considered financial advice. Use it for educational and illustrative purposes only.
