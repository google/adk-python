# GCSToolset

GCSToolset and GCSAdminToolset provide an agent with the ability to interact
with Google Cloud Storage (GCS). These toolsets allow an agent to manage
buckets and the objects stored within them.

## Introduction

Google Cloud Storage toolsets bridge the gap between large language models and
cloud-based file systems. The `GCSToolset` class focuses on object-level
operations, such as uploading files, reading content, and listing directory
structures. The `GCSAdminToolset` class handles administrative tasks, including
creating buckets and modifying bucket metadata.

Developers use these toolsets when an agent needs to retrieve context from
stored documents, save generated reports to the cloud, or manage the
infrastructure where data resides. The toolsets rely on `GCSCredentialsConfig`
to handle authentication and `GCSToolSettings` to enforce permission boundaries.

## Get started

This example configures an agent with full read and write access to GCS objects.
The agent uses application default credentials for authentication.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.gcs import GCSToolset
from google.adk.integrations.gcs.settings import Capabilities, GCSToolSettings

# Configure the toolset to allow both reading and writing.
# By default, toolsets are read-only.
tool_settings = GCSToolSettings(capabilities=[Capabilities.READ_WRITE])

# Initialize the toolset. 
# Providing no credentials_config defaults to Application Default Credentials.
gcs_toolset = GCSToolset(gcs_tool_settings=tool_settings)

agent = LlmAgent(
    name="storage_manager",
    instruction="You help users upload and retrieve files from GCS buckets.",
    tools=[gcs_toolset],
)
```

## How it works

The toolsets act as containers that dynamically generate a list of tools based
on the provided configuration. When a `Runner` or an agent calls `get_tools`,
the toolset inspects its `GCSToolSettings` to determine which operations are
permitted.

If the settings include `Capabilities.READ_ONLY` or `Capabilities.READ_WRITE`,
the toolset provides tools for listing objects and retrieving metadata or data.
If the settings specifically include `Capabilities.READ_WRITE`, the toolset
adds tools for creating and deleting objects. This structure ensures that the
agent only sees tools that match its intended permission level, which reduces
the risk of unauthorized operations.

Each tool is wrapped in a `GoogleTool` instance. This wrapper handles the
injection of credentials and project information into the underlying storage
functions at execution time. The tool names are prefixed with `gcs` by default,
helping the model distinguish storage operations from other tools in the same
agent.

## Configuration options

The following options configure how the toolset behaves and which tools it
exposes to the agent.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool_filter` | `ToolPredicate \| list[str] \| None` | `None` | Filters which tools the toolset returns. |
| `credentials_config` | `GCSCredentialsConfig \| None` | `None` | The authentication configuration for Google Cloud. |
| `gcs_tool_settings` | `GCSToolSettings \| None` | `GCSToolSettings()` | Controls the read/write capabilities of the tools. |

### tool_filter
Developers use this option to limit the tools available to an agent. Providing
a list of strings, such as `["list_objects", "get_object_data"]`, ensures the
agent cannot attempt to delete or create files even if the capabilities allow
it. This reduces the token count in the system prompt and prevents the agent
from becoming confused by irrelevant capabilities.

### credentials_config
This object manages how the toolset authenticates with Google Cloud. If it is
omitted, the toolset attempts to use Application Default Credentials (ADC).
Developers provide a `GCSCredentialsConfig` when they need to use a specific
service account key or perform an interactive OAuth2 flow.

### gcs_tool_settings
This defines the operational boundaries for the tools. The `capabilities`
attribute within these settings defaults to `[Capabilities.READ_ONLY]`. This
safe default prevents an agent from modifying data unless a developer
explicitly grants `Capabilities.READ_WRITE` access.

## Advanced applications

### Managing buckets with GCSAdminToolset
While `GCSToolset` manages files, `GCSAdminToolset` manages the buckets
themselves. This is useful for agents that need to set up new environments or
audit storage configurations.

```python
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.gcs import GCSAdminToolset
from google.adk.integrations.gcs.settings import Capabilities, GCSToolSettings

# Admin tools also respect capabilities. 
# READ_WRITE is required to create or delete buckets.
admin_settings = GCSToolSettings(capabilities=[Capabilities.READ_WRITE])
admin_toolset = GCSAdminToolset(gcs_tool_settings=admin_settings)

agent = LlmAgent(
    name="cloud_admin",
    instruction="Create and configure GCS buckets for new projects.",
    tools=[admin_toolset],
)
```

### Customizing authentication scopes
By default, `GCSCredentialsConfig` uses the `devstorage.full_control` scope. If
an application requires more restrictive scopes at the OAuth level, they can
be specified during configuration.

```python
from google.adk.integrations.gcs.gcs_credentials import GCSCredentialsConfig

credentials_config = GCSCredentialsConfig(
    client_id="your-client-id",
    client_secret="your-client-secret",
    scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
)
```

## Limitations

Google Cloud Storage requires a bucket to be empty before it can be deleted.
The `delete_bucket` tool in `GCSAdminToolset` will fail if the bucket contains
any objects. To delete a non-empty bucket, the agent must first use the
`delete_objects` tool from `GCSToolset` to remove all files.

The `get_object_data` tool attempts to decode object content as UTF-8 text. If
the object contains binary data that is not valid UTF-8, the tool returns the
content as a base64-encoded string and sets the `encoding` field in the
response to `base64`. The agent must be instructed how to handle base64
content if it is expected to process binary files.

## Related samples

- [gcs](../../../../../contributing/samples/integrations/gcs/agent.py)
