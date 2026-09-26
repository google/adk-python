# OpenAI Integration (Moved)

The OpenAI models moved to
[`google.adk.integrations.openai`](../../integrations/openai/README.md).

This package re-exports them so existing imports keep working:

```python
from google.adk.labs.openai import OpenAILlm  # still works
from google.adk.integrations.openai import OpenAILlm  # preferred
```

New code should import from `google.adk.integrations.openai`.
