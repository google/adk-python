# Changelog

All notable changes to the GCP Security Intelligence Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-10-07

### Fixed
- **ADK Automatic Function Calling Compatibility**
  - Changed all security tool return types from `StructuredToolResponse` to `str`
  - ADK automatic function calling requires simple types (str, dict, int) - custom dataclasses not supported
  - Fixed: `get_security_insights_summary()`, `query_security_insights()`, `get_security_statistics()`

- **BigQuery Schema Corrections**
  - Fixed column reference: `resource_type` → `resource_name` (actual column in security_findings table)
  - Corrected SQL queries to use proper column names
  - Table schema: id, name, category, severity, resource_name, description, recommendation, state, created_at, project_id

- **Chainlit Configuration**
  - Fixed directory structure: `.chainlit` file → `.chainlit/config.toml` directory
  - Resolved `FileExistsError` on Chainlit startup
  - Configured `user_env = []` for local development with .env file

- **Session Management**
  - Prevented duplicate ADK session creation on Chainlit UI refresh
  - Added session reuse logic in `on_chat_start()` method
  - Now maintains single session per user instead of creating duplicates

### Added
- Detailed schema documentation in tool docstrings
  - Added complete column list to `query_security_insights()` with examples
  - Added valid `group_by` values to `get_security_statistics()`
  - Helps AI model generate accurate SQL queries with correct column names

### Changed
- Updated dependency validation in startup script
  - Fast import-based checks for critical packages (flask, google-cloud-aiplatform, requests, python-dotenv)
  - Reduced validation time from ~30s to ~0.5s
  - Maintains reference to full test suite for comprehensive validation

## [1.0.0] - 2025-10-07

### Added
- Modular Chainlit integration (plug-and-play for existing apps)
- Unified service management with `start_all.sh` and `stop_all.sh`
- Comprehensive testing suite with 95.3% dependency validation success
- Complete documentation suite (setup, integration, troubleshooting)
- ADK Evals suite with 13 test cases
- Clean project structure with archived legacy code

### Changed
- Migrated from custom tools to ADK-native implementation
- Consolidated 32 tools into 5 categories (BigQuery, Service Evaluation, Service Discovery, Confluence, Security Feeds)
- Improved error handling and logging across all tools

### Core Components
- **ADK Backend** - Agent orchestration & API (port 8000)
- **Flask UI** - Web interface (port 5001)
- **Chainlit UI** - Chat interface (port 8001)
- **MCP Server** - Claude Desktop integration (stdio)

---

## Version History

- **1.0.1** - Bug fixes for ADK compatibility and BigQuery schema
- **1.0.0** - Initial production release with modular architecture
