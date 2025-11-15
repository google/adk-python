# Hello World with Apigee LLM

This sample demonstrates how to use the Agent Development Kit (ADK) with an LLM fronted by an Apigee proxy. It showcases the flexibility of the `ApigeeLlm` class in configuring the target LLM provider (Gemini or Vertex AI) and API version through the model string.

## Setup

Before running the sample, you need to configure your environment with the necessary credentials.

1.  **Create a `.env` file:**
    Copy the sample environment file to a new file named `.env` in the same directory.
    ```bash
    cp .env-sample .env
    ```

2.  **Set Environment Variables:**
    Open the `.env` file and provide values for the following variables:

    -   `GOOGLE_API_KEY`: Your API key for the Google AI services (Gemini).
    -   `APIGEE_PROXY_URL`: The full URL of your Apigee proxy endpoint.

    Example `.env` file:
    ```
    GOOGLE_API_KEY="your-google-api-key"
    APIGEE_PROXY_URL="https://your-apigee-proxy.net/basepath"
    ```

    The `main.py` script will automatically load these variables when it runs.

## Run the Sample

Once your `.env` file is configured, you can run the sample with the following command:

```bash
python main.py
```

## Configuring the Apigee LLM

The `ApigeeLlm` class is configured using a special model string format in `agent.py`. This string determines which backend provider (Vertex AI or Gemini) and which API version to use.

### Model String Format

The supported format is:

`apigee/[<provider>/][<version>/]<model_id>`

-   **`provider`** (optional): Can be `vertex_ai` or `gemini`.
    -   If specified, it forces the use of that provider.
    -   If omitted, the provider is determined by the `GOOGLE_GENAI_USE_VERTEXAI` environment variable. If this variable is set to `true` or `1`, Vertex AI is used; otherwise, `gemini` is used by default.

-   **`version`** (optional): The API version to use (e.g., `v1`, `v1beta`).
    -   If omitted, the default version for the selected provider is used.

-   **`model_id`** (required): The identifier for the model you want to use (e.g., `gemini-2.5-flash`).

### Configuration Examples

Here are some examples of how to configure the model string in `agent.py` to achieve different behaviors:

1.  **Implicit Provider (determined by environment variable):**

    -   `model="apigee/gemini-2.5-flash"`
        -   Uses the default API version.
        -   Provider is Vertex AI if `GOOGLE_GENAI_USE_VERTEXAI` is true; otherwise, Gemini.

    -   `model="apigee/v1/gemini-2.5-flash"`
        -   Uses API version `v1`.
        -   Provider is determined by the environment variable.

2.  **Explicit Provider (ignores environment variable):**

    -   `model="apigee/vertex_ai/gemini-2.5-flash"`
        -   Uses Vertex AI with the default API version.

    -   `model="apigee/gemini/gemini-2.5-flash"`
        -   Uses Gemini with the default API version.

    -   `model="apigee/gemini/v1/gemini-2.5-flash"`
        -   Uses Gemini with API version `v1`.

    -   `model="apigee/vertex_ai/v1beta/gemini-2.5-flash"`
        -   Uses Vertex AI with API version `v1beta`.

By modifying the `model` string in `agent.py`, you can test various configurations without changing the core logic of the agent.
