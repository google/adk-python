# MCP Service Account Agent Sample

This agent demonstrates how to connect to a remote MCP server using a gcloud service account for authentication. It uses Streamable HTTP for communication.

## Setup

Before running the agent, you need to configure the MCP server URL and your service account credentials in `agent.py`.

1.  **Configure MCP Server URL:**
    Update the `MCP_SERVER_URL` variable with the URL of your MCP server instance.

    ```python
    # agent.py
    # TODO: Update this to the production MCP server url and scopes.
    MCP_SERVER_URL = "https://test.sandbox.googleapis.com/mcp"
    ```

2.  **Set up Service Account Credentials:**
    -   Obtain the JSON key file for your gcloud service account.
    -   In `agent.py`, find the `ServiceAccountCredential` object and populate its parameters (e.g., `project_id`, `private_key`, `client_email`, etc.) with the corresponding values from your JSON key file.

    ```python
    # agent.py
    # TODO: Update this to the user's service account credentials.
    auth_credential=AuthCredential(
        auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
        service_account=ServiceAccount(
            service_account_credential=ServiceAccountCredential(
                type_="service_account",
                project_id="example",
                private_key_id="123",
                private_key="123",
                client_email="test@example.iam.gserviceaccount.com",
                client_id="123",
                auth_uri="https://accounts.google.com/o/oauth2/auth",
                token_uri="https://oauth2.googleapis.com/token",
                auth_provider_x509_cert_url=(
                    "https://www.googleapis.com/oauth2/v1/certs"
                ),
                client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/example.iam.gserviceaccount.com",
                universe_domain="googleapis.com",
            ),
            scopes=SCOPES.keys(),
        ),
    ),
    ```

## Running the Agent

Once configured, you can run the agent.

For example:
```bash
adk web
```