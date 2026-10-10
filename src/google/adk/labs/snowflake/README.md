# Snowflake Cortex Agents Integration (Experimental)

This folder contains an experimental integration that runs an existing
Snowflake Cortex Agent as an ADK root agent. Like everything under
`google.adk.labs`, it may change or be removed without notice.

`SnowflakeCortexAgent` calls the Cortex Agents Run REST API directly with the
`httpx` client that ADK already depends on, so there is no extra package to
install.

```python
from google.adk.labs.snowflake import SnowflakeCortexAgent
```

See the
[SnowflakeCortexAgent guide](../../../../../docs/guides/labs/snowflake/snowflake_cortex_agent/index.md)
for setup, configuration, limitations, and API details.
