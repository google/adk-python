# Logging Style Guide

## Module Logger

Every module that logs declares its logger with the `google_adk.` prefix:

```python
logger = logging.getLogger('google_adk.' + __name__)
```

`scripts/compliance_checks.py` fails the commit on a bare
`logging.getLogger(__name__)`. The prefix puts every ADK record under one
logger tree, so an application embedding ADK can raise or silence the
framework's logging with a single `logging.getLogger('google_adk')` call
without touching its own.

## General Rules

- **Lazy formatting**: pass values as arguments so the string is only built
  when the level is enabled.
  - Good: `logger.info('Processing item %s', item_id)`
  - Bad: `logger.info(f'Processing item {item_id}')`
- **Never log secrets**: API keys, credentials, tokens, or PII.
- **Contextual logging**: include trace IDs when available so records
  correlate across an invocation.

## Log Levels

- **DEBUG**: diagnostic detail. Use freely in internal implementation.
- **INFO**: expected milestones (workflow started, node completed).
- **WARNING**: something unexpected that did not stop the operation (a retry,
  a deprecated field).
- **ERROR**: a failure that prevented the operation from completing.
