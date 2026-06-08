# Logging Style Guide

## General Rules

- **Lazy Evaluation**: Use lazy-evaluated `%`-based templates for logging to avoid overhead when the log level is not enabled.
  - **Good**: `logging.info("Processing item %s", item_id)`
  - **Bad**: `logging.info(f"Processing item {item_id}")`
- **Contextual Logging**: Leverage structured logging and trace IDs when available to correlate logs across operations.
- **No Secrets**: Never log sensitive information (API keys, user credentials, or PII).

## Log Levels

- **DEBUG**: Detailed information for diagnosing problems. Use generously in internal implementation but avoid cluttering production logs.
- **INFO**: Confirmation that things are working as expected (e.g., workflow started, node completed).
- **WARNING**: Indication that something unexpected happened or a problem might occur soon (e.g., retry triggered).
- **ERROR**: A serious problem that prevented a function or operation from completing.
