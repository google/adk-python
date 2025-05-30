# Application Integration Agent Sample with End-User Credentials

## Introduction

This sample demonstrates how to use the `ApplicationIntegrationToolset` within
an ADK agent to interact with external applications using **end-user OAuth 2.0
credentials**. Specifically, this agent (`agent.py`) is configured to interact
with Google Calendar using a pre-configured Application Integration connection
and authenticating as the end user.

## Prerequisites

1.  **Set up Integration Connection:**
    *   You need an existing
        [Integration connection](https://cloud.google.com/integration-connectors/docs/overview)
        configured to interact with Google Calendar APIs. Follow the
        [documentation](https://google.github.io/adk-docs/tools/google-cloud-tools/#use-integration-connectors)
        to provision the Integration Connector in Google Cloud. You will need
        the `Connection Name`, `Project ID`, and `Location` of your connection.
    *   Ensure the connection is configured to use Google Calendar (e.g., by
        enabling the `google-calendar-connector` or a similar connector).

2.  **Configure OAuth 2.0 Client:**
    *   You need an OAuth 2.0 Client ID and Client Secret that is authorized to
        access the required Google Calendar scopes (e.g.,
        `https://www.googleapis.com/auth/calendar.readonly`). You can create
        OAuth credentials in the Google Cloud Console under "APIs & Services"
        -> "Credentials".

3.  **Configure Environment Variables:**
    *   Create a `.env` file in the same directory as `agent.py` (or add to
        your existing one).
    *   Add the following variables to the `.env` file, replacing the
        placeholder values with your actual connection details:

      ```dotenv
      CONNECTION_NAME=<YOUR_CALENDAR_CONNECTION_NAME>
      CONNECTION_PROJECT=<YOUR_GOOGLE_CLOUD_PROJECT_ID>
      CONNECTION_LOCATION=<YOUR_CONNECTION_LOCATION>
      CLIENT_ID=<YOUR_OAUTH_CLIENT_ID>
      CLIENT_SECRET=<YOUR_OAUTH_CLIENT_SECRET>
      ```

## End-User Authentication (OAuth 2.0)

This agent utilizes the `AuthCredential` and `OAuth2Auth` classes from the ADK
to handle authentication.
*   It defines an OAuth 2.0 scheme (`oauth2_scheme`) based on Google Cloud's
    OAuth endpoints and required scopes.
*   It uses the `CLIENT_ID` and `CLIENT_SECRET` from the environment variables
    (or hardcoded values in the sample) to configure `OAuth2Auth`.
*   This `AuthCredential` is passed to the `ApplicationIntegrationToolset`,
    enabling the tool to make authenticated API calls to Google Calendar on
    behalf of the user running the agent. The ADK framework will typically
    handle the OAuth flow (e.g., prompting the user for consent) when the tool
    is first invoked.

## How to Use

1.  **Install Dependencies:** Ensure you have the necessary libraries installed
    (e.g., `google-adk`, `python-dotenv`).
2.  **Run the Agent:** Execute the agent script from your terminal:
    ```bash
    python agent.py
    ```
3.  **Interact:** Once the agent starts, you can interact with it. If it's the
    first time using the tool requiring OAuth, you might be prompted to go
    through the OAuth consent flow in your browser. After successful
    authentication, you can ask the agent to perform tasks.

## Sample Prompts

Here are some examples of how you can interact with the agent:

*   `Can you list events from my primary calendar?`