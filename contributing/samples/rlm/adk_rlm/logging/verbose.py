"""
Verbose printing for RLM using rich.

Provides console output for debugging and understanding RLM execution.
Uses a "Tokyo Night" inspired color theme.
"""

from typing import Any

from adk_rlm.types import CodeBlock
from adk_rlm.types import RLMIteration
from adk_rlm.types import RLMMetadata
from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

# Tokyo Night Color Theme
COLORS = {
    "primary": "#7AA2F7",  # Soft blue - headers, titles
    "secondary": "#BB9AF7",  # Soft purple - emphasis
    "success": "#9ECE6A",  # Soft green - success, code
    "warning": "#E0AF68",  # Soft amber - warnings
    "error": "#F7768E",  # Soft red/pink - errors
    "text": "#A9B1D6",  # Soft gray-blue - regular text
    "muted": "#565F89",  # Muted gray - less important
    "accent": "#7DCFFF",  # Bright cyan - accents
    "bg_subtle": "#1A1B26",  # Dark background
    "border": "#3B4261",  # Border color
    "code_bg": "#24283B",  # Code background
}

# Rich styles
STYLE_PRIMARY = Style(color=COLORS["primary"], bold=True)
STYLE_SECONDARY = Style(color=COLORS["secondary"])
STYLE_SUCCESS = Style(color=COLORS["success"])
STYLE_WARNING = Style(color=COLORS["warning"])
STYLE_ERROR = Style(color=COLORS["error"])
STYLE_TEXT = Style(color=COLORS["text"])
STYLE_MUTED = Style(color=COLORS["muted"])
STYLE_ACCENT = Style(color=COLORS["accent"], bold=True)


def _to_str(value: Any) -> str:
  """Convert any value to string safely."""
  if isinstance(value, str):
    return value
  return str(value)


class VerbosePrinter:
  """
  Rich console printer for RLM verbose output.

  Displays beautiful, structured output showing the RLM's execution:
  - Initial configuration panel
  - Each iteration with response summaries
  - Code execution with results
  - Sub-calls to other models
  """

  def __init__(self, enabled: bool = True):
    """
    Initialize the verbose printer.

    Args:
        enabled: Whether verbose printing is enabled.
    """
    self.enabled = enabled
    self.console = Console() if enabled else None
    self._iteration_count = 0

  def print_header(
      self,
      backend: str,
      model: str,
      environment: str,
      max_iterations: int,
      max_depth: int,
      other_backends: list[str] | None = None,
  ) -> None:
    """Print the initial RLM configuration header."""
    if not self.enabled:
      return

    # Main title
    title = Text()
    title.append("◆ ", style=STYLE_ACCENT)
    title.append("RLM", style=Style(color=COLORS["primary"], bold=True))
    title.append(" ━ Recursive Language Model (ADK)", style=STYLE_MUTED)

    # Configuration table
    config_table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        padding=(0, 2),
        expand=True,
    )
    config_table.add_column("key", style=STYLE_MUTED, width=16)
    config_table.add_column("value", style=STYLE_TEXT)
    config_table.add_column("key2", style=STYLE_MUTED, width=16)
    config_table.add_column("value2", style=STYLE_TEXT)

    config_table.add_row(
        "Backend",
        Text(backend, style=STYLE_SECONDARY),
        "Environment",
        Text(environment, style=STYLE_SECONDARY),
    )
    config_table.add_row(
        "Model",
        Text(model, style=STYLE_ACCENT),
        "Max Iterations",
        Text(str(max_iterations), style=STYLE_WARNING),
    )

    if other_backends:
      backends_text = Text(", ".join(other_backends), style=STYLE_SECONDARY)
      config_table.add_row(
          "Sub-models",
          backends_text,
          "Max Depth",
          Text(str(max_depth), style=STYLE_WARNING),
      )
    else:
      config_table.add_row(
          "Max Depth",
          Text(str(max_depth), style=STYLE_WARNING),
          "",
          "",
      )

    panel = Panel(
        config_table,
        title=title,
        title_align="left",
        border_style=COLORS["border"],
        padding=(1, 2),
    )

    self.console.print()
    self.console.print(panel)
    self.console.print()

  def print_metadata(self, metadata: RLMMetadata) -> None:
    """Print RLM metadata as header."""
    if not self.enabled:
      return

    model = metadata.backend_kwargs.get("model_name", "unknown")
    other = list(metadata.other_backends) if metadata.other_backends else None

    self.print_header(
        backend=metadata.backend,
        model=model,
        environment=metadata.environment_type,
        max_iterations=metadata.max_iterations,
        max_depth=metadata.max_depth,
        other_backends=other,
    )

  def print_iteration_start(self, iteration: int) -> None:
    """Print the start of a new iteration."""
    if not self.enabled:
      return

    self._iteration_count = iteration

    rule = Rule(
        Text(f" Iteration {iteration} ", style=STYLE_PRIMARY),
        style=COLORS["border"],
        characters="─",
    )
    self.console.print(rule)

  def print_completion(
      self, response: Any, iteration_time: float | None = None
  ) -> None:
    """Print a completion response."""
    if not self.enabled:
      return

    # Header with timing
    header = Text()
    header.append("◇ ", style=STYLE_ACCENT)
    header.append("LLM Response", style=STYLE_PRIMARY)
    if iteration_time:
      header.append(f"  ({iteration_time:.2f}s)", style=STYLE_MUTED)

    # Response content
    response_str = _to_str(response)
    response_text = Text(response_str, style=STYLE_TEXT)

    # Count words roughly
    word_count = len(response_str.split())
    footer = Text(f"~{word_count} words", style=STYLE_MUTED)

    panel = Panel(
        Group(response_text, Text(), footer),
        title=header,
        title_align="left",
        border_style=COLORS["muted"],
        padding=(0, 1),
    )
    self.console.print(panel)

  def print_code_execution(self, code_block: CodeBlock) -> None:
    """Print code execution details."""
    if not self.enabled:
      return

    result = code_block.result

    # Header
    header = Text()
    header.append("▸ ", style=STYLE_SUCCESS)
    header.append(
        "Code Execution", style=Style(color=COLORS["success"], bold=True)
    )
    if result.execution_time:
      header.append(f"  ({result.execution_time:.3f}s)", style=STYLE_MUTED)

    # Build content
    content_parts = []

    # Code snippet
    code_text = Text()
    code_text.append("Code:\n", style=STYLE_MUTED)
    code_text.append(_to_str(code_block.code), style=STYLE_TEXT)
    content_parts.append(code_text)

    # Stdout if present
    stdout_str = _to_str(result.stdout) if result.stdout else ""
    if stdout_str.strip():
      stdout_text = Text()
      stdout_text.append("\nOutput:\n", style=STYLE_MUTED)
      stdout_text.append(stdout_str, style=STYLE_SUCCESS)
      content_parts.append(stdout_text)

    # Stderr if present (error)
    stderr_str = _to_str(result.stderr) if result.stderr else ""
    if stderr_str.strip():
      stderr_text = Text()
      stderr_text.append("\nError:\n", style=STYLE_MUTED)
      stderr_text.append(stderr_str, style=STYLE_ERROR)
      content_parts.append(stderr_text)

    # Sub-calls summary
    if result.rlm_calls:
      calls_text = Text()
      calls_text.append(
          f"\n↳ {len(result.rlm_calls)} sub-call(s)", style=STYLE_SECONDARY
      )
      content_parts.append(calls_text)

    panel = Panel(
        Group(*content_parts),
        title=header,
        title_align="left",
        border_style=COLORS["success"],
        padding=(0, 1),
    )
    self.console.print(panel)

  def print_subcall(
      self,
      model: str,
      prompt_preview: str,
      response_preview: str,
      execution_time: float | None = None,
  ) -> None:
    """Print a sub-call to another model."""
    if not self.enabled:
      return

    # Header
    header = Text()
    header.append("  ↳ ", style=STYLE_SECONDARY)
    header.append("Sub-call: ", style=STYLE_SECONDARY)
    header.append(_to_str(model), style=STYLE_ACCENT)
    if execution_time:
      header.append(f"  ({execution_time:.2f}s)", style=STYLE_MUTED)

    # Content
    content = Text()
    content.append("Prompt: ", style=STYLE_MUTED)
    content.append(_to_str(prompt_preview)[:200], style=STYLE_TEXT)
    if len(prompt_preview) > 200:
      content.append("...", style=STYLE_MUTED)
    content.append("\nResponse: ", style=STYLE_MUTED)
    content.append(_to_str(response_preview)[:200], style=STYLE_TEXT)
    if len(response_preview) > 200:
      content.append("...", style=STYLE_MUTED)

    panel = Panel(
        content,
        title=header,
        title_align="left",
        border_style=COLORS["secondary"],
        padding=(0, 1),
    )
    self.console.print(panel)

  def print_iteration(
      self, iteration: RLMIteration, iteration_num: int
  ) -> None:
    """
    Print a complete iteration including response and code executions.
    """
    if not self.enabled:
      return

    # Print iteration header
    self.print_iteration_start(iteration_num)

    # Print the LLM response
    self.print_completion(iteration.response, iteration.iteration_time)

    # Print each code block execution
    for code_block in iteration.code_blocks:
      self.print_code_execution(code_block)

      # Print any sub-calls made during this code block
      for call in code_block.result.rlm_calls:
        self.print_subcall(
            model=call.root_model,
            prompt_preview=_to_str(call.prompt) if call.prompt else "",
            response_preview=_to_str(call.response) if call.response else "",
            execution_time=call.execution_time,
        )

  def print_final_answer(self, answer: Any) -> None:
    """Print the final answer."""
    if not self.enabled:
      return

    # Title
    title = Text()
    title.append("★ ", style=STYLE_WARNING)
    title.append(
        "Final Answer", style=Style(color=COLORS["warning"], bold=True)
    )

    # Answer content
    answer_text = Text(_to_str(answer), style=STYLE_TEXT)

    panel = Panel(
        answer_text,
        title=title,
        title_align="left",
        border_style=COLORS["warning"],
        padding=(1, 2),
    )

    self.console.print()
    self.console.print(panel)
    self.console.print()

  def print_summary(
      self,
      total_iterations: int,
      total_time: float,
      usage_summary: dict[str, Any] | None = None,
  ) -> None:
    """Print a summary at the end of execution."""
    if not self.enabled:
      return

    # Summary table
    summary_table = Table(
        show_header=False,
        show_edge=False,
        box=None,
        padding=(0, 2),
    )
    summary_table.add_column("metric", style=STYLE_MUTED)
    summary_table.add_column("value", style=STYLE_ACCENT)

    summary_table.add_row("Iterations", str(total_iterations))
    summary_table.add_row("Total Time", f"{total_time:.2f}s")

    if usage_summary:
      total_input = sum(
          m.get("total_input_tokens", 0)
          for m in usage_summary.get("model_usage_summaries", {}).values()
      )
      total_output = sum(
          m.get("total_output_tokens", 0)
          for m in usage_summary.get("model_usage_summaries", {}).values()
      )
      if total_input or total_output:
        summary_table.add_row("Input Tokens", f"{total_input:,}")
        summary_table.add_row("Output Tokens", f"{total_output:,}")

    self.console.print()
    self.console.print(Rule(style=COLORS["border"], characters="═"))
    self.console.print(summary_table, justify="center")
    self.console.print(Rule(style=COLORS["border"], characters="═"))
    self.console.print()
