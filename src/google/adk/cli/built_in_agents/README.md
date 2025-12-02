# Agent Builder Assistant

An intelligent assistant for building ADK multi-agent systems using YAML configurations.

## Quick Start

### Using ADK Web Interface
```bash
# From the ADK project root
adk web src/google/adk/agent_builder_assistant
```

### Programmatic Usage
```python
# Create with defaults
agent = AgentBuilderAssistant.create_agent()

# Create with custom settings
agent = AgentBuilderAssistant.create_agent(
    model="gemini-2.5-pro",
    schema_mode="query",
    working_directory="/path/to/project"
)
```

## Core Features

### 🎯 **Intelligent Agent Design**
- Analyzes requirements and suggests appropriate agent types
- Designs multi-agent architectures (Sequential, Parallel, Loop patterns)
- Provides high-level design confirmation before implementation

### 📝 **Advanced YAML Configuration**
- Generates AgentConfig schema-compliant YAML files
- Supports all agent types: LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
- Built-in validation with detailed error reporting

### 🛠️ **Multi-File Management**
- **Read/Write Operations**: Batch processing of multiple files
- **File Type Separation**: YAML files use validation tools, Python files use generic tools
- **Backup & Recovery**: Automatic backups before overwriting existing files

### 🗂️ **Project Structure Analysis**
- Explores existing project structures
- Suggests conventional ADK file organization
- Provides path recommendations for new components

### 🧭 **Dynamic Path Resolution**
- **Session Binding**: Each chat session bound to one root directory
- **Working Directory**: Automatic detection and context provision
- **ADK Source Discovery**: Finds ADK installation dynamically (no hardcoded paths)

## Schema Modes

Choose between two schema handling approaches:

### Embedded Mode (Default)
```python
agent = AgentBuilderAssistant.create_agent(schema_mode="embedded")
```
- Full AgentConfig schema embedded in context
- Faster execution, higher token usage
- Best for comprehensive schema work

### Query Mode
```python
agent = AgentBuilderAssistant.create_agent(schema_mode="query")
```
- Dynamic schema queries via tools
- Lower initial token usage
- Best for targeted schema operations

## Example Interactions

### Create a new agent
```
Create an agent that can roll n-sided number and check whether the rolled number is prime.
```

### Add Capabilities to Existing Agent
```
Could you make the agent under `./config_based/roll_and_check` a multi agent system : root_agent only for request routing and two sub agents responsible for two functions respectively ?
```

### Project Structure Analysis
```
Please analyze my existing project structure at './config_based/roll_and_check' and suggest improvements for better organization.
```

## Tool Ecosystem

### Core File Operations
- **`read_config_files`** - Read multiple YAML configurations with analysis
- **`write_config_files`** - Write multiple YAML files with validation
- **`read_files`** - Read multiple files of any type
- **`write_files`** - Write multiple files with backup options
- **`delete_files`** - Delete multiple files with backup options

### Project Analysis
- **`explore_project`** - Analyze project structure and suggest paths
- **`resolve_root_directory`** - Resolve paths with working directory context

### ADK knowledge Context
- **`google_search`** - Search for ADK examples and documentation
- **`url_context`** - Fetch content from URLs (GitHub, docs, etc.)
- **`search_adk_source`** - Search ADK source code with regex patterns


## File Organization Conventions

### ADK Project Structure
```
my_adk_project/
└── src/
    └── my_app/
        ├── root_agent.yaml
        ├── sub_agent_1.yaml
        ├── sub_agent_2.yaml
        ├── tools/
        │   ├── process_email.py    # No _tool suffix
        │   └── analyze_sentiment.py
        └── callbacks/
            ├── logging.py            # No _callback suffix
            └── security.py
```

### Naming Conventions
- **Agent directories**: `snake_case`
- **Tool files**: `descriptive_action.py`
- **Callback files**: `descriptive_name.py`
- **Tool paths**: `project_name.tools.module.function_name`
- **Callback paths**: `project_name.callbacks.module.function_name`

## Session Management

### Root Directory Binding
Each chat session is bound to a single root directory:

- **Automatic Detection**: Working directory provided to model automatically
- **Session State**: Tracks established root directory across conversations
- **Path Resolution**: All relative paths resolved against session root
- **Directory Switching**: Suggest user starting new session to work in different directory

### Working Directory Context
```python
# The assistant automatically receives working directory context
agent = AgentBuilderAssistant.create_agent(
    working_directory="/path/to/project"
)
# Model instructions include: "Working Directory: /path/to/project"
```

## Advanced Features

### Dynamic ADK Source Discovery
No hardcoded paths - works in any ADK installation:

```python
from google.adk.agent_builder_assistant.utils import (
    find_adk_source_folder,
    get_adk_schema_path,
    load_agent_config_schema
)

# Find ADK source dynamically
adk_path = find_adk_source_folder()

# Load schema with caching
schema = load_agent_config_schema()
```

### Schema Validation
All YAML files validated against AgentConfig schema:

- **Syntax Validation**: YAML parsing with detailed error locations
- **Schema Compliance**: Full AgentConfig.json validation
- **Best Practices**: ADK naming and structure conventions
- **Error Recovery**: Clear suggestions for fixing validation errors

## Performance Optimization

### Efficient Operations
- **Multi-file Processing**: Batch operations reduce overhead
- **Schema Caching**: Global cache prevents repeated file reads
- **Dynamic Discovery**: Efficient ADK source location caching
- **Session Context**: Persistent directory binding across conversations

### Memory Management
- **Lazy Loading**: Schema loaded only when needed
- **Cache Control**: Manual cache clearing for testing/development
- **Resource Cleanup**: Automatic cleanup of temporary files

## Error Handling

### Comprehensive Validation
- **Path Validation**: All paths validated before file operations
- **Schema Compliance**: AgentConfig validation with detailed error reporting
- **Python Syntax**: Syntax validation for generated Python code
- **Backup Creation**: Automatic backups before overwriting files

### Recovery Mechanisms
- **Retry Suggestions**: Clear guidance for fixing validation errors
- **Backup Restoration**: Easy recovery from automatic backups
- **Error Context**: Detailed error messages with file locations and suggestions

This comprehensive assistant provides everything needed for intelligent, efficient ADK agent system creation with proper validation, file management, and project organization.
