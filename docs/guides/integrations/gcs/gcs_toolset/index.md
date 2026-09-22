# GCSToolset and GCSAdminToolset

Two toolsets give an agent access to Cloud Storage. `GCSToolset` works with the
objects inside buckets, which it can list, read, create and delete.
`GCSAdminToolset` works with the buckets themselves. Both are read-only until
you say otherwise, and the split between them is the main thing to get right.

To try either one you need a Google Cloud project with Cloud Storage enabled,
credentials that can reach it, and the `google-cloud-storage` client library.
Storage and the requests these tools make are both billed, so an agent that
reads large objects on every turn adds measurably to the cost.

## Introduction

An agent that handles files needs somewhere to put them, and Cloud Storage is
usually already in the picture: the artifact service can be backed by it, and
data the agent is asked to analyze often lives there. Writing the tools by hand
means a storage client, credential plumbing, and encoding decisions for binary
objects.

Two toolsets rather than one is a deliberate separation, along the same line
Google Cloud's own IAM roles draw. Reading an object and deleting a bucket are
not the same kind of act, and an agent that summarizes documents has no reason
to be able to do the second. Give it `GCSToolset` and the bucket operations do
not exist as far as the model is concerned.

Within each toolset, `GCSToolSettings.capabilities` is a second, coarser
control: with the default `READ_ONLY` the mutating tools are never returned at
all.

## Get started

Three steps get a read-only document agent running.

1.  Enable Cloud Storage in a Google Cloud project, and make credentials that
    can reach it available to the process.

2.  Install the client library, either directly or through the `gcp` extra:

    ```bash
    pip install google-cloud-storage
    ```

    It is not optional. If it is missing, every import from
    `google.adk.integrations.gcs` fails, including imports of the settings and
    credentials classes that do not use the library at all.

3.  Construct the toolset and put it in an agent. If you leave `GCSToolSettings`
    out, as the example below does, the toolset returns only the three read
    tools:

    ```python
    from google.adk.agents import LlmAgent
    from google.adk.integrations.gcs import GCSCredentialsConfig
    from google.adk.integrations.gcs import GCSToolset
    import google.auth

    # Application Default Credentials. See
    # https://cloud.google.com/docs/authentication/provide-credentials-adc
    application_default_credentials, _ = google.auth.default()

    gcs_toolset = GCSToolset(
        credentials_config=GCSCredentialsConfig(
            credentials=application_default_credentials
        )
    )

    root_agent = LlmAgent(
        name="gcs_agent",
        description="Answers questions about documents in Cloud Storage.",
        instruction=(
            "You can list and read objects in Cloud Storage. List a bucket's"
            " contents before guessing at an object name, and say which object an"
            " answer came from."
        ),
        tools=[gcs_toolset],
    )
    ```

Allow writes by asking for them explicitly:

```python
from google.adk.integrations.gcs.settings import Capabilities
from google.adk.integrations.gcs.settings import GCSToolSettings

gcs_toolset = GCSToolset(
    credentials_config=credentials_config,
    gcs_tool_settings=GCSToolSettings(capabilities=[Capabilities.READ_WRITE]),
)
```

If you construct `GCSCredentialsConfig()` with no arguments, it raises. It
requires one of `credentials`, `external_access_token_key`, or a `client_id`
and `client_secret` pair.

## The tools

Every tool name carries a `gcs_` prefix once the toolset hands it to the model,
because both toolsets set `tool_name_prefix="gcs"`. The names below are what
the model sees.

**`GCSToolset`, for objects.**

| Tool | Capability | What it does |
| :--- | :--- | :--- |
| `gcs_list_objects` | read | Lists object names in a bucket, with optional `prefix` and pagination. |
| `gcs_get_object_metadata` | read | Size, content type, generation and the rest of an object's properties. |
| `gcs_get_object_data` | read | The object's contents, as text or base64, or downloaded to a local path. |
| `gcs_create_object` | write | Creates an object from inline `data` or from a local file. |
| `gcs_delete_objects` | write | Deletes a list of objects in one call. |

**`GCSAdminToolset`, for buckets.**

| Tool | Capability | What it does |
| :--- | :--- | :--- |
| `gcs_list_buckets` | read | Lists buckets in a project. |
| `gcs_get_bucket` | read | Reads a bucket's metadata. |
| `gcs_create_bucket` | write | Creates a bucket. |
| `gcs_update_bucket` | write | Changes a bucket's configuration. |
| `gcs_delete_bucket` | write | Deletes a bucket. |

The two sets do not overlap, so an agent given both gets ten distinct tools
under one prefix.

### How the tools report failure

Every tool catches its own exceptions and returns a dictionary:
`{"status": "SUCCESS", "results": ...}` or
`{"status": "ERROR", "error_details": "<the exception text>"}`. Nothing
propagates to your application. A permission denial, a missing bucket and a
network failure all arrive at the model as an error string, and the model
decides what to do about it. If you need to know that a tool failed, add an
`after_tool_callback` that inspects the result, or a plugin.

### Read an object

`gcs_get_object_data` has two modes. Given a `destination_file_path` it
downloads to that local path and returns a confirmation string. Without one it
returns the content inline, and how depends on the bytes: valid UTF-8 comes
back as text with `"encoding": "text"`, and anything else is base64-encoded
with `"encoding": "base64"`.

The inline path has no size limit. Ask for a 200 MB object and the whole thing
is read into memory, base64-encoded if binary, and put in the model's context.
Prefer `destination_file_path` for anything that is not a small text file, and
pair it with a `prefix` on `gcs_list_objects` so the model is choosing from a
short list.

## Configuration options

Both toolsets take the same three keyword-only arguments.

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `tool_filter` | `ToolPredicate \| list[str] \| None` | `None` | Which tools to expose. |
| `credentials_config` | `GCSCredentialsConfig \| None` | `None` | How the tools authenticate. |
| `gcs_tool_settings` | `GCSToolSettings \| None` | `None` | Capabilities. A default read-only instance is built when omitted. |

**`tool_filter` is matched against the unprefixed name.** If you write
`tool_filter=["gcs_list_objects"]`, you get no tools at all and no warning,
because filtering happens inside `get_tools`, before the toolset applies its
`gcs_` prefix. Write `tool_filter=["list_objects"]` instead. A predicate sees
the same unprefixed names.

**`credentials_config`** is optional in the signature. If you omit it, each
tool is invoked with `credentials=None`, which the storage client is very
unlikely to accept, so in practice it is required.

### GCSToolSettings

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `capabilities` | `list[Capabilities]` | `[Capabilities.READ_ONLY]` | Which classes of tool the toolset returns. |

`Capabilities` has two members, `READ_ONLY` and `READ_WRITE`, and the list is
interpreted as a set rather than as a sequence: read tools are returned if
either value is present, and write tools only if `READ_WRITE` is. So
`[READ_WRITE]` alone gives you everything, and listing both is the same thing
written longer. An empty list returns no tools at all, without a warning.

The safety this buys is real but shallow. It decides which tools the model is
told about; it does not constrain the credentials. If you configure `READ_ONLY`
on a service account that holds `devstorage.full_control`, the agent cannot
delete an object *through these tools*, and the service account can still
delete it through anything else. The durable control is IAM.

### GCSCredentialsConfig

It extends ADK's shared Google credentials configuration, with three mutually
exclusive ways to authenticate:

*   `credentials=` takes an existing `google.auth` credentials object, such as
    Application Default Credentials or a loaded service-account key. Every end
    user's requests run as that one identity.
*   `client_id=` and `client_secret=` together start an interactive OAuth flow,
    so each end user consents and the tools act as them. That flow produces the
    pause-for-consent handshake described in
    [the tool auth guide](../../../auth/tool_auth/index.md).
*   `external_access_token_key=` names the key under which the surrounding
    platform has already placed an access token in tool context state.

When `scopes` is not set, it defaults to
`https://www.googleapis.com/auth/devstorage.full_control`. That is full
control, not read-only, whatever `capabilities` you configured.

**`scopes` can only be narrowed on the OAuth path.** The three authentication
styles are validated as mutually exclusive, and `scopes` counts as part of the
OAuth one: passing it alongside `credentials` raises
`ValueError: If credentials are provided, external_access_token_key, client_id,
client_secret, and scopes must not be provided.` So this works,

```python
GCSCredentialsConfig(
    client_id=os.environ["OAUTH_CLIENT_ID"],
    client_secret=os.environ["OAUTH_CLIENT_SECRET"],
    scopes=["https://www.googleapis.com/auth/devstorage.read_only"],
)
```

and with a credentials object there is no scope argument at all. Narrow it on
the credentials themselves before you hand them over, or on the service
account's IAM roles.

## Advanced applications

Two arrangements come up once a read-only agent works: combining the two
toolsets in one agent, and allowing some writes while withholding others.

### Data tools and admin tools in one agent

Give the agent both toolsets when it genuinely provisions storage, and keep the
capability settings separate so that "can create buckets" and "can delete
objects" are two decisions rather than one:

```python
root_agent = LlmAgent(
    name="storage_admin",
    description="Provisions buckets and manages the objects in them.",
    instruction="Confirm with the user before creating or deleting anything.",
    tools=[
        GCSToolset(
            credentials_config=credentials_config,
            gcs_tool_settings=GCSToolSettings(
                capabilities=[Capabilities.READ_WRITE]
            ),
        ),
        GCSAdminToolset(
            credentials_config=credentials_config,
            gcs_tool_settings=GCSToolSettings(),  # read-only buckets
        ),
    ],
)
```

### Restrict to specific tools

`tool_filter` is finer-grained than `capabilities`: it can allow writes while
withholding deletion, which the capability enum cannot express.

```python
GCSToolset(
    credentials_config=credentials_config,
    gcs_tool_settings=GCSToolSettings(capabilities=[Capabilities.READ_WRITE]),
    tool_filter=["list_objects", "get_object_data", "create_object"],
)
```

A name in the list that matches nothing is ignored, so a typo removes a tool
instead of raising.

## Limitations

*   **The missing-dependency error does not name the dependency.** Without
    `google-cloud-storage` the import fails with `ImportError: cannot import
    name 'storage' from 'google.cloud' (unknown location)`. There is no guard
    and no install hint, and the namespace-package wording sends people looking
    for the wrong problem.
*   **Errors are returned, not raised.** Every tool catches everything and
    hands the model an error dictionary. Your application never sees it.
*   **`tool_filter` uses unprefixed names while the model sees prefixed ones.**
    Filtering on `gcs_list_objects` silently yields nothing.
*   **`capabilities` is not a permission boundary.** It picks which tools exist
    in the declaration. IAM is what actually stops a write.
*   **The default scope is `devstorage.full_control`, and it can only be
    narrowed on the OAuth path.** `scopes` is rejected alongside `credentials`
    or `external_access_token_key`, so an agent authenticating with a
    credentials object always requests full control.
*   **`gcs_get_object_data` has no size cap.** Inline reads put the entire
    object in the model's context, base64-encoded when it is binary.
*   **`gcs_create_object` and `gcs_get_object_data` touch the local
    filesystem.** `source_file_path` and `destination_file_path` are local
    paths chosen by the model, with no sandboxing and no path validation. On a
    server, either withhold those tools or make sure the process cannot reach
    anything it should not.
*   **Both toolsets are flag-gated and experimental.** They are registered under
    the `GCS_TOOLSET` and `GCS_ADMIN_TOOLSET` feature flags, both on by default,
    and constructing one warns once per process. Setting
    `ADK_DISABLE_GCS_TOOLSET=1` makes the constructor raise `RuntimeError`.
*   **Bucket deletion still requires an empty bucket.** `gcs_delete_bucket`
    does not empty it for you; the agent has to call `gcs_delete_objects` first,
    which means `GCSToolset` with `READ_WRITE` as well.

## Related samples

*   [GCS tools](../../../../../contributing/samples/integrations/gcs/agent.py)
    runs the object toolset across the whole credential matrix.
*   [GCS admin tools](../../../../../contributing/samples/integrations/gcs_admin/agent.py)
    does the same for the bucket toolset.

## Related guides

*   [AuthConfig and authenticated tools](../../../auth/tool_auth/index.md)
    explains what happens when `client_id` and `client_secret` send a user
    through OAuth.
*   [BaseArtifactService](../../../artifacts/artifact_service/index.md) is the
    other way ADK uses Cloud Storage, for artifacts instead of as a tool.
