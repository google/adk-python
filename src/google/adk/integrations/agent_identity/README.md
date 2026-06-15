# GCP IAM Connector Auth

Manages the complete lifecycle of an access token using the Google Cloud
Platform Agent Identity Credentials service.

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install "google-adk[agent-identity]"
    ```

2. **Register the provider:**
    Register the `GcpAuthProvider` with the `CredentialManager`. This is to be
    done one time.

    ``` py
    # user_agent_app.py
    from google.adk.auth.credential_manager import CredentialManager
    from google.adk.integrations.agent_identity import GcpAuthProvider

    CredentialManager.register_auth_provider(GcpAuthProvider())
    ```

3.  **Configure the Auth provider:**
    Specify the Agent Identity provider configuration using the
    `GcpAuthProviderScheme`.
    ``` py
    # user_agent_app.py
    from google.adk.integrations.agent_identity import GcpAuthProviderScheme

    # Configures Toolset
    auth_scheme = GcpAuthProviderScheme(name="my-jira-auth_provider")
    mcp_toolset_jira = McpToolset(..., auth_scheme=auth_scheme)
    ```
