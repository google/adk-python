# ADK Integrations

This directory houses modules that integrate ADK with external tools and
services. The goal is to provide an organized and scalable way to extend ADK's
capabilities.

Integrations with external systems, such as the Agent Registry, BigQuery,
ApiHub, etc., should be developed within sub-packages in this folder. This
centralization makes it easier for developers to find, use, and contribute to
various integrations.

## What Belongs Here?

*   Code that connects ADK to other services, APIs, or tools.
*   Modules that depend on third-party libraries not included in the core ADK
    dependencies.

## Guidelines for Contributions

1.  **Self-Contained Packages:** Each integration should reside in its own
    sub-directory (e.g., `integrations/my_service/`).
2.  **Internal Structure:** Integration sub-packages are free to manage their
    own internal code structure and design patterns. They do not need to
    strictly follow the core ADK framework's structure.
3.  **Dependencies:** To keep the core ADK lightweight, dependencies required
    for a specific integration must be optional. These should be defined as
    "extras" in the `pyproject.toml`. Users will install them using commands
    like `pip install "google-adk[my_service]"`. The extra name should match the
    integration directory name.
4.  **Lazy Importing:** Implement lazy importing within the integration code. If
    a user tries to use an integration without installing the necessary extras,
    catch the `ModuleNotFoundError` and raise a descriptive error message
    guiding the user to the correct installation command.
5.  **Documentation:** Ensure clear documentation is provided for each
    integration, including setup, configuration, and usage examples.
