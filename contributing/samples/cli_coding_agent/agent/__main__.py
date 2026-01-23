#!/usr/bin/env python3
"""Interactive CLI runner for the coding agent with colorful output."""

import asyncio
import json
from typing import Any

from coding_agent.agent import coding_agent
from google.adk.runners import InMemoryRunner
from google.adk.utils.context_utils import Aclosing
from google.genai import types


# ANSI color codes for terminal output
class Colors:
  """ANSI color codes for terminal formatting."""

  # Text colors
  RED = "\033[91m"
  GREEN = "\033[92m"
  YELLOW = "\033[93m"
  BLUE = "\033[94m"
  MAGENTA = "\033[95m"
  CYAN = "\033[96m"
  WHITE = "\033[97m"
  GRAY = "\033[90m"

  # Styles
  BOLD = "\033[1m"
  UNDERLINE = "\033[4m"

  # Reset
  RESET = "\033[0m"

  @classmethod
  def color(cls, text: str, color: str, bold: bool = False) -> str:
    """Apply color to text."""
    style = cls.BOLD if bold else ""
    return f"{style}{color}{text}{cls.RESET}"


def print_header(text: str):
  """Print a header with formatting."""
  print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
  print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.RESET}")
  print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")


def print_section(title: str):
  """Print a section divider."""
  print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.RESET}")
  print(f"{Colors.GRAY}{'─'*60}{Colors.RESET}")


def print_event(event_type: str, author: str | None = None):
  """Print event metadata."""
  icon_map = {
      "ModelTurn": "🤖",
      "ToolUse": "🔧",
      "ToolResult": "✅",
      "Error": "❌",
      "TurnComplete": "✨",
  }
  icon = icon_map.get(event_type, "📨")

  event_color = Colors.YELLOW if "Tool" in event_type else Colors.CYAN
  print(f"\n{icon} {Colors.color(event_type, event_color, bold=True)}", end="")

  if author:
    print(f" {Colors.GRAY}[{author}]{Colors.RESET}", end="")
  print()


def print_text(text: str):
  """Print assistant text response."""
  print(f"{Colors.GREEN}💬 Response:{Colors.RESET}")
  # Indent the text for better readability
  for line in text.split("\n"):
    print(f"   {line}")


def truncate_arg(value: Any, max_len: int = 20) -> str:
  """Truncate argument value if too long."""
  str_value = str(value)
  if len(str_value) > max_len:
    return f"{str_value[:max_len]}..."
  return str_value


def print_tool_call(name: str, args: dict):
  """Print tool call information in concise function-call format."""
  if args:
    # Format as function call with truncated args
    arg_strs = [f"{k}={truncate_arg(v)}" for k, v in args.items()]
    args_display = ", ".join(arg_strs)
    print(
        f"{Colors.MAGENTA}🔧 {Colors.BOLD}{name}({args_display}){Colors.RESET}"
    )
  else:
    print(f"{Colors.MAGENTA}🔧 {Colors.BOLD}{name}(){Colors.RESET}")


def render_plan(plan_data: dict):
  """Render a plan in a special visual format.

  Args:
      plan_data: Plan data structure with title and tasks
  """
  if not plan_data or "plan" not in plan_data:
    return

  plan = plan_data.get("plan")
  if not plan:
    return

  # Print plan header
  print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}")
  print(f"{Colors.BOLD}{Colors.CYAN}📋 PLAN: {plan['title']}{Colors.RESET}")
  print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.RESET}\n")

  # Calculate progress
  completed_count = sum(1 for t in plan["tasks"] if t["completed"])
  total_count = len(plan["tasks"])
  progress_pct = (completed_count / total_count * 100) if total_count > 0 else 0

  # Progress bar
  bar_width = 40
  filled = (
      int(bar_width * completed_count / total_count) if total_count > 0 else 0
  )
  bar = "█" * filled + "░" * (bar_width - filled)
  print(
      f"{Colors.GRAY}Progress: [{Colors.GREEN}{bar}{Colors.GRAY}]"
      f" {completed_count}/{total_count} ({progress_pct:.0f}%){Colors.RESET}\n"
  )

  # Print tasks
  for task in plan["tasks"]:
    task_id = task["id"]
    description = task["description"]
    completed = task["completed"]

    if completed:
      # Use dim/gray text for completed tasks
      print(
          f"{Colors.GREEN}  ✓ [{task_id}]"
          f" {Colors.GRAY}{description}{Colors.RESET}"
      )
    else:
      print(
          f"{Colors.YELLOW}  ○ [{task_id}]"
          f" {Colors.WHITE}{description}{Colors.RESET}"
      )

  print(f"\n{Colors.GRAY}{'─'*60}{Colors.RESET}\n")


def print_tool_response(name: str, response: Any, verbose: bool = False):
  """Print tool response information.

  Args:
      name: Tool name
      response: Tool response
      verbose: If True, show full response. If False, show checkmark/error only.
  """
  # Check if this is a plan-related response and render specially
  if isinstance(response, dict) and "plan" in response and response.get("plan"):
    # Always render plan visually
    render_plan(response)
    return

  if verbose:
    print(f"{Colors.GREEN}✅ {Colors.BOLD}{name}{Colors.RESET}")
    # Format the response nicely
    if isinstance(response, dict):
      print(f"{Colors.GRAY}   Result:{Colors.RESET}")
      formatted_response = json.dumps(response, indent=4)
      for line in formatted_response.split("\n"):
        print(f"      {line}")
    elif isinstance(response, list):
      print(f"{Colors.GRAY}   Result:{Colors.RESET}")
      formatted_response = json.dumps(response, indent=4)
      for line in formatted_response.split("\n"):
        print(f"      {line}")
    else:
      print(f"{Colors.GRAY}   Result:{Colors.RESET} {response}")
  else:
    # Concise mode: just show success/failure
    if isinstance(response, dict) and "error" in response:
      print(f"{Colors.RED}✗ {name} failed{Colors.RESET}")
    else:
      print(f"{Colors.GREEN}✓ {name}{Colors.RESET}")


def print_error(message: str):
  """Print error message."""
  print(f"{Colors.RED}❌ ERROR:{Colors.RESET} {message}")


def print_finish(reason: str):
  """Print turn completion message."""
  # Simple newline for cleaner output
  print()


async def run_cli():
  """Run the coding agent in interactive CLI mode."""
  # Session IDs for the conversation
  user_id = "cli_user"
  session_id = "coding_session"

  # Verbose mode for tool responses (default: False for concise output)
  verbose_responses = False

  # Create runner with the agent
  runner = InMemoryRunner(agent=coding_agent, app_name=coding_agent.name)

  # Create session
  await runner.session_service.create_session(
      app_name=runner.app_name, user_id=user_id, session_id=session_id
  )

  # Print ASCII art banner
  ascii_art = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ██████╗ ██╗  ██╗     ██████╗ ██████╗ ██████╗ ██████╗  ║
    ║    ██╔══██╗██╔══██╗██║ ██╔╝    ██╔════╝██╔═══██╗██╔══██╗██╔══██╗ ║
    ║    ███████║██║  ██║█████╔╝     ██║     ██║   ██║██║  ██║█████╗   ║
    ║    ██╔══██║██║  ██║██╔═██╗     ██║     ██║   ██║██║  ██║██╔══╝   ║
    ║    ██║  ██║██████╔╝██║  ██╗    ╚██████╗╚██████╔╝██████╔╝███████╗ ║
    ║    ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝     ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝ ║
    ║                                                           ║
    ║                ██████╗██╗     ██╗                        ║
    ║               ██╔════╝██║     ██║                        ║
    ║               ██║     ██║     ██║                        ║
    ║               ██║     ██║     ██║                        ║
    ║               ╚██████╗███████╗██║                        ║
    ║                ╚═════╝╚══════╝╚═╝                        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
  print(f"{Colors.CYAN}{Colors.BOLD}{ascii_art}{Colors.RESET}")

  print(
      f"\n{Colors.CYAN}Agent:{Colors.RESET}"
      f" {Colors.BOLD}{coding_agent.name}{Colors.RESET}"
  )
  print(f"{Colors.CYAN}Model:{Colors.RESET} {coding_agent.model}")
  print(
      f"{Colors.CYAN}Output:{Colors.RESET}"
      f" {'Verbose' if verbose_responses else 'Concise'} mode"
  )
  print(f"\n{Colors.GRAY}Commands:{Colors.RESET}")
  print(f"  • {Colors.YELLOW}exit/quit{Colors.RESET} - Exit the assistant")
  print(
      f"  • {Colors.YELLOW}verbose{Colors.RESET} - Toggle verbose tool"
      " responses"
  )
  print(f"\n{Colors.GRAY}File tools:{Colors.RESET}")
  print(f"  • read_file, write_file, update_file, list_directory")
  print(f"\n{Colors.GRAY}Planning tools:{Colors.RESET}")
  print(f"  • create_plan, update_plan, get_plan, reset_plan")
  print()

  while True:
    # Get user input
    try:
      user_input = input(f"{Colors.BOLD}{Colors.BLUE}[You]:{Colors.RESET} ")
    except (EOFError, KeyboardInterrupt):
      print(f"\n{Colors.YELLOW}Exiting...{Colors.RESET}")
      break

    # Check for exit command
    if user_input.strip().lower() in ["exit", "quit", "q"]:
      print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
      break

    # Check for verbose toggle
    if user_input.strip().lower() == "verbose":
      verbose_responses = not verbose_responses
      mode = "verbose" if verbose_responses else "concise"
      print(
          f"{Colors.CYAN}Output mode set to:"
          f" {Colors.BOLD}{mode}{Colors.RESET}\n"
      )
      continue

    # Skip empty inputs
    if not user_input.strip():
      continue

    # Create message content
    message = types.Content(role="user", parts=[types.Part(text=user_input)])

    # Run the agent and stream events
    try:
      print()  # Add blank line before output

      async with Aclosing(
          runner.run_async(
              user_id=user_id, session_id=session_id, new_message=message
          )
      ) as event_stream:
        async for event in event_stream:

          # Handle content parts
          if event.content and event.content.parts:
            for part in event.content.parts:
              # Print text content
              if part.text:
                print_text(part.text)

              # Print function calls
              if part.function_call:
                print_tool_call(
                    part.function_call.name, part.function_call.args
                )

              # Print function responses
              if part.function_response:
                response = getattr(part.function_response, "response", None)
                print_tool_response(
                    part.function_response.name,
                    response,
                    verbose=verbose_responses,
                )

          # Print error information if present
          if event.error_message:
            print_error(event.error_message)

          # Print finish reason if turn is complete
          if event.turn_complete and event.finish_reason:
            print_finish(event.finish_reason)

    except Exception as e:
      print_error(str(e))
      import traceback

      print(f"{Colors.GRAY}{traceback.format_exc()}{Colors.RESET}")

  # Close the runner
  await runner.close()
  print(f"\n{Colors.CYAN}Session closed. Goodbye!{Colors.RESET}\n")


def main():
  """Entry point for the CLI."""
  asyncio.run(run_cli())


if __name__ == "__main__":
  main()
