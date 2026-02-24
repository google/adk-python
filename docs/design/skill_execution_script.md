

# Summary 	

This outlines the design for adding pre-registered function execution within ADK’s non-bash based skills toolset. 

# Motivation

ADK Skills provide a way to package instructions, resources, and scripts to extend agent capabilities. While skills can guide an agent's reasoning, the ability to execute code within the skill's context unlocks more powerful actions. This requires a secure and well-defined execution environment.  

We also want to support adapting existing `BaseTool` or `FunctionTool` instances, making them available as part of the Skill interface.

# Proposal 

There are two parts to the proposal for pre-registered function execution:

1. Allow SkillToolset to include **additional ADK tools/functions** that are specified by the skill: adapting existing BaseTool and FunctionTool interfaces so that users can feed in existing functions  
2. Enable **script execution** (i.e. files in skill/scripts) with RunSkillScriptTool

## Enabling usage of ADK tools

Users should be able to pass in existing ADK tools (both built-in and custom) as BaseTool objects and have them be used within a skill. To do this, we can use the [allowed\_tools]field of a skill’s frontmatter to determine which tools should be instantiated.

Within SkillToolset, we can modify the signature to accept an optional `additional_tools` argument that allows users to pass in pre-instantiated tools, toolsets, or callables. Then we can update `SkillToolset.get_tools()` to dynamically resolve the tools in allowed\_tools for all skills in the toolset. We will iterate through the skill’s allowed tools and check if any of the `additional_tools` match, and add them to the toolset if so. As a fallback, we will also check if the tools exist as built-in tools in the google.adk.tools directory.

Since get\_tools() is called every invocation, we will also cache the tools so we don’t have to resolve them every time.

```py
class SkillToolset(BaseToolset): 
  def __init__(
    self, 
    skills: list[models.Skill], 
    additional_tools: list[ToolUnion] = None,
  ):
    ...
  
  def get_tools(self, readonly_context) -> list[BaseTool]:
    # Collect allowed tools from skills
    allowed_tool_names = set()
    for skill in self.skills:
      if skill.frontmatter.allowed_tools:
        allowed_tool_names.update([name for name in skill.frontmatter.allowed_tools])
    
   # Resolve tools passed in with `additional_tools`
   tools_by_name = {}
   for tool_union in self._additional_tools:
     if isinstance(tool_union, BaseTool):
       tools_by_name[tool_union.name] = tool_union
     if isinstance(tool_union, BaseToolset):
       ts_tools = await tool_union.get_tools(readonly_context)
       for t in ts_tools:
         tools_by_name[t.name] = t
     elif callable(tool_union):
       tools_by_name[tool_union.name] = FunctionTool(tool_union)

  for allowed_tool in allowed_tool_names:
     # add tools from tools_by_name or if they don't exist, 
     # try to resolve using built-in tools in the google.adk.tools directory
     ...
```

## Enabling Script Execution

We will integrate script execution into `SkillToolset` by using the existing [BaseCodeExecutor] interface.  
We can modify `SkillToolset` to accept an optional code\_executor argument and create a new tool, `RunSkillScriptTool`, that will be used for script execution:

```py
class SkillToolset(BaseToolset): 
  def __init__(
    self, 
    skills: list[models.Skill], 
    code_executor: Optional[base_code_executor.BaseCodeExecutor] = None,
  ):
    super().__init__() 
    self._skills = {skill.name: skill for skill in skills} 
    self._code_executor = code_executor 
    
    self._tools = [LoadSkillTool(self), LoadSkillResourceTool(self)] 
    # Add RunSkillScriptTool for function execution
    if self._code_executor:
      self._tools.append(RunSkillScriptTool(self))
```

`RunSkillScriptTool` will be used for script execution. It will take a toolset and script\_metadata as input.

```py
class RunSkillScriptTool(BaseTool): 
  def __init__(self, toolset: "SkillToolset", script_metadata: Dict[str, Any] = None): 
    # Initialize the tool
    pass
  
  def _get_declaration(self) -> types.FunctionDeclaration | None: 
    params_schema = { 
        "type": "object", 
        "properties": { 
          "skill_name": {
            "type": "string", 
            "description": "The name of the skill."
          },    
          "script_path": {
            "type": "string", 
            "description": "The relative path to the script (e.g., 'scripts/my_script.py' or 'scripts/setup.sh')."
           }, 
           "args": {
             "type": "object", 
             "description": "Optional arguments to pass to the script as key-value pairs."
           }, 
         }, 
         "required": ["skill_name", "script_path"], 
    }
    return types.FunctionDeclaration( 
      name=self.name, 
      description=self.description, 
      parameters_json_schema=params_schema
     ) 
   
  async def run_async(self, *, args: dict[str, Any], tool_context: ToolContext) -> Any:
      # 1. Validate inputs (skill_name, script_path)
      # 2. Extract script and arguments Map
      # 3. Mount all skill files (assets, references, scripts) as input_files
      # 4. Set sandbox_working_dir to the sandbox root '.'
      # 5. Generate safe subprocess wrappers or sys.argv injections
      # 6. Execute via BaseCodeExecutor matching toolset configuration
      ...
```

**Script Invocation:** Arguments are passed to the script similar to command-line arguments, by mocking `sys.argv`. The script's standard output, standard error, and any resulting files will be captured by the executor.

**Executor Choices:** The `SkillToolset` can be configured with any ADK `BaseCodeExecutor` instance. Recommended executors include VertexAiCodeExecutor (executes code in a secure Vertex AI sandbox) and GkeCodeExecutor (executes code in a gVisor sandbox on GKE).

### Adapting BaseCodeExecutor

The current `BaseCodeExecutor` interface is designed for executing LLM-generated code snippets. To effectively support `RunSkillScriptTool`, we should consider file system context and path resolution:

`BaseCodeExecutor`'s `execute_code` method doesn't have a way to inform the execution environment about the skill's file structure. Scripts within a skill will likely need to read from `../references/` or `../assets/` using relative paths. The sandbox needs to honor these paths relative to the skill's root or the `scripts/` directory. 

We can extend the `File` Dataclass in `code_execution_utils.py` to include a `path`:

```py
@dataclasses.dataclass(frozen=True) 
class File: 
  """A structure that contains a file name and its content.""" 
  name: str # Base name of the file 
  content: str | bytes 
  mime_type: str = 'text/plain' 
  path: Optional[str] = None # Native relative path (e.g. 'references/guidelines.md')
```

We will also extend `CodeExecutionInput` to add a `working_dir`:

```py
@dataclasses.dataclass 
class CodeExecutionInput: 
  code: str 
  input_files: list[File] = dataclasses.field(default_factory=list) 
  execution_id: Optional[str] = None 
  working_dir: Optional[str] = None # e.g., '/skill/scripts'
```

`RunSkillScriptTool` will package the Skill’s resources into input\_files to add to the CodeExecutionInput. Within the BaseCodeExecutor implementations (VertexAI, GKE, etc), we will then:

* Read input\_files. For each `File` with sandbox\_path, create the file and any necessary parent directories at the exact path in the sandbox  
* Set the sandbox working directory to sandbox\_working\_dir

#### File Permissions

Files/directories created within the sandbox from the skill’s `references/` and `assets/` should be read-only for the executed script. The script should have write-access to a dedicated temp directory in the sandbox and potentially a designated output directory.

#### Other Considerations

The CodeExecutor and sandbox environment will address most script execution issues:

* Resource limits are enforced by sandbox env  
* Network access disabled by default  
* **Output files:** Files will be created in output directory and returned in `CodeExecutionResult.output_files`  
* **Error handling:** Report script exceptions, exit codes, executor errors through `CodeExecutionResult.stderr`  
  * Add error post-processing: we will implement an ‘LLM-friendly’ error formatter. Instead of returning the full traceback, the tool will extract the specific Exception type and the offending line of code to help the agent self-correct its script invocation.

## RunSkillScriptTool — Design Details & Considerations

### Overview

`RunSkillScriptTool` enables ADK agents to execute scripts bundled inside a
skill's `scripts/` directory via ADK's `BaseCodeExecutor` infrastructure. This
closes the gap between the
[Agent Skills spec](https://agentskills.io/specification) (which defines
`scripts/` as an optional skill resource) and ADK's runtime capabilities. By
mounting skill dependencies including `references/` and `assets/`,
`RunSkillScriptTool` executes skills in sandboxed environments with full access
to their context.

### Architecture

```
LLM calls run_skill_script(skill_name, script_path, args)
        │
        ▼
┌─ RunSkillScriptTool.run_async() ─────────────────────┐
│  1. Validate params & resolve skill/script                │
│  2. Resolve code executor (toolset → agent fallback)      │
│  3. Validate args (handled natively by JSON schema)       │
│  4. Mount dependencies (assets, references, scripts)      │
│  5. _prepare_code() → generate Python wrapper code        │
│  6. code_executor.execute_code(..., input_files=...)      │
│  7. Parse result (JSON envelope for shell scripts)        │
│  8. Return {stdout, stderr, status} to LLM               │
└───────────────────────────────────────────────────────────┘
```

### Parameter Schema Design

The tool employs specific parameter designs to ensure safe sandboxed execution
and high LLM reliability:

1.  **`script_path` vs `script_name`:** \
    Instead of a flat `script_name` (e.g. `setup.sh`), the tool requires the
    full relative `script_path` (e.g. `scripts/setup.sh` or
    `scripts/utils/helper.py`). This is required because the tool mounts the
    entire skill directory into the execution sandbox, meaning the script must
    be invoked from the true sandbox root path so it can reliably access its
    `assets/` and `references/` via relative paths.

2.  **`args` Dictionary:** \
    Instead of taking a raw array of string arguments (`["--verbose",
    "--force"]`), the tool takes a structured key-value `args` object
    (`{"verbose": true, "force": true}`). LLMs are significantly more reliable
    at generating structured JSON objects than raw command-line flag arrays.
    Furthermore, accepting an object moves the burden of secure structural
    flattening (e.g. constructing the `['--verbose', 'True']` array) to the
    Python code, completely eliminating a class of shell-injection
    vulnerabilities.

### Script Type Handling

Type   | Extension      | Execution Method                                      | Timeout                | Args Injection
:----- | :------------- | :---------------------------------------------------- | :--------------------- | :-------------
Python | `.py`          | Direct `exec()` via code executor                     | No (executor-level)    | `sys.argv = [script_path] + mapped_args`
Shell  | `.sh`, `.bash` | `subprocess.run(['bash', script_path] + mapped_args)` | Yes (`script_timeout`) | `args` parsed as sequence of flattened pairs
Other  | any            | Rejected                                              | N/A                    | N/A

**Extensionless files are rejected** (not silently treated as Python) to avoid
unexpected behavior.

### Code Executor Resolution Chain

```
1. SkillToolset(code_executor=...)    ← explicit, highest priority
2. agent.code_executor                ← fallback to agent's executor
3. None → return NO_CODE_EXECUTOR     ← actionable error
```

This design allows a single toolset-level executor to be shared across all
skills, or per-agent executors for different isolation levels.

### Shell Script JSON Envelope

Shell scripts face a unique challenge: `UnsafeLocalCodeExecutor` captures stdout
via `redirect_stdout(StringIO)`, but if the generated code raises an exception,
`stdout.getvalue()` is never called and stdout is lost. This means a naive
`raise RuntimeError(stderr)` approach discards any stdout the script produced.

Crucially, **even when running inside a secure sandbox** (like Vertex AI Code
Interpreter), sandboxes often struggle to cleanly report *why* an arbitrary
script failed if the script loops infinitely or crashes aggressively. An abrupt
exit often yields a generic "sandbox failed" error, denying the LLM the context
it needs to self-correct.

**Solution**: The Python subprocess wrapper we inject *around* the script
executes the shell command safely and serializes both stdout and stderr as a
JSON envelope through the single stdout channel.

Even inside a sandbox environment, our wrapper catches the shell script's output
in real-time. If the shell script times out inside the sandbox, our wrapper
catches the `TimeoutExpired` exception, scoops up whatever output the shell
script produced *before* it hung, packages it into JSON, and returns that
perfectly structured payload to the Executor. This guarantees the LLM always
receives perfectly exact `stdout` and `stderr` logs regardless of script
crashes.

```py
# Generated code for shell scripts:
import subprocess, shlex, json as _json
try:
    _r = subprocess.run(
        ['bash', SCRIPT_PATH] + MAPPED_ARGS,
        capture_output=True, text=True,
        timeout=SCRIPT_TIMEOUT,
    )
    print(_json.dumps({
        '__shell_result__': True,
        'stdout': _r.stdout,
        'stderr': _r.stderr,
        'returncode': _r.returncode,
    }))
except subprocess.TimeoutExpired as _e:
    print(_json.dumps({
        '__shell_result__': True,
        'stdout': _e.stdout or '',
        'stderr': 'Timed out after Ns',
        'returncode': -1,
    }))
```

`run_async()` then parses this JSON envelope (only for shell scripts, keyed on
`__shell_result__`) to extract both streams and the return code. This works
reliably with both `UnsafeLocalCodeExecutor` and container-based executors.

### Three-State Status Model

Status    | Condition
:-------- | :-------------------------------------------------------------
`success` | No stderr
`warning` | Both stdout and stderr present (e.g., deprecation warnings)
`error`   | Stderr only (no stdout), or non-zero returncode with no stdout

### Security Considerations

**Shell injection prevention:**

-   **Structured Argument Arrays:** `args` is passed as a structured dictionary
    by the LLM natively. The tool converts these into strict, flattened string
    arrays `['--key', 'value']` and passes them securely to `subprocess.run`
    with `shell=False`. Because the elements are passed as a strict array, the
    underlying OS treats flags and values as literal parameters passed *into*
    the script, meaning any malicious shell operators (e.g. `&&`, `|`) are
    treated as literal strings and ignored.
-   The script source is executed as an isolated file path inside the sandboxed
    `$working_dir`.

**`SystemExit` handling:**

-   `sys.exit()` raises `SystemExit(BaseException)`, which is NOT caught by
    `except Exception` in executors
-   `run_async()` explicitly catches `SystemExit` to prevent skill scripts from
    terminating the host process
-   `sys.exit(0)` and `sys.exit(None)` are treated as successful termination
-   Non-zero exit codes return `EXECUTION_ERROR`

**Executor security:**

-   `UnsafeLocalCodeExecutor` runs code in the host process via `exec()` —
    suitable only for trusted, first-party skills
-   For third-party or untrusted skills, a sandboxed executor (e.g.,
    `ContainerCodeExecutor`) should be used
-   The sample agent includes explicit warnings about this

### Known Limitations & Future Work

1.  **No timeout for Python scripts**: `exec()` provides no built-in timeout
    mechanism. A malicious/buggy Python script can hang indefinitely. This is an
    executor-level concern — solving it properly requires running Python scripts
    in a subprocess or implementing executor-level cancellation.

2.  **Python script stdout lost on exception**: When a Python script writes to
    stdout and then raises, `UnsafeLocalCodeExecutor` loses the stdout (same
    root cause as the shell fix). This is less critical for Python since
    exceptions are the natural error mechanism, but could be improved at the
    executor level.

### Error Codes Reference

Error Code                | Meaning
:------------------------ | :---------------------------------------------
`MISSING_SKILL_NAME`      | `skill_name` parameter not provided
`MISSING_SCRIPT_PATH`     | `script_path` parameter not provided
`SKILL_NOT_FOUND`         | No skill with that name registered
`SCRIPT_NOT_FOUND`        | No script with that name in the skill
`NO_CODE_EXECUTOR`        | No code executor configured (toolset or agent)
`UNSUPPORTED_SCRIPT_TYPE` | File extension not `.py`, `.sh`, or `.bash`
`EXECUTION_ERROR`         | Runtime error, non-zero exit, or `sys.exit(N)`
