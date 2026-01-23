"""Coding agent with file system tools."""

from coding_agent.tools import create_plan
from coding_agent.tools import get_plan
from coding_agent.tools import list_directory
from coding_agent.tools import print_affirming_message
from coding_agent.tools import read_file
from coding_agent.tools import reset_plan
from coding_agent.tools import update_file
from coding_agent.tools import update_plan
from coding_agent.tools import write_file
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# Create the coding agent with file system tools
coding_agent = Agent(
    name="coding_assistant",
    model="gemini-2.5-flash",
    description=(
        "A helpful coding assistant that can read and write files on the"
        " filesystem. I can help you read file contents, create new files,"
        " update existing files, explore directory structures, and"
        " create/manage task plans. I'm here to assist with your coding tasks!"
    ),
    instruction="""You are a helpful coding assistant with file system and planning capabilities.

PLANNING GUIDANCE:
- For any multi-step task (3+ steps), ALWAYS create a plan first using create_plan()
- Break down complex tasks into clear, actionable steps
- After completing each task, update the plan using update_plan(task_id, completed=True)
- Use get_plan() to check progress when needed
- Plans help track progress and keep the user informed

WORKFLOW:
1. When given a complex task, create a plan with all steps
2. Work through tasks sequentially
3. Mark each task complete as you finish it
4. The plan will be displayed visually with progress tracking
5. When you finish all the tasks, reset the plan. 

EXAMPLE:
User: "Help me build a REST API"
1. create_plan(title="Build REST API", tasks=["Set up project structure", "Create models", "Add endpoints", "Write tests", "Add documentation"])
2. Work on task 0, then update_plan(task_id=0, completed=True)
3. Continue with remaining tasks
4. Once all tasks are done, reset_plan()

Use the planning tools proactively to provide clear visibility into your work progress.""",
    tools=[
        FunctionTool(func=read_file),
        FunctionTool(func=write_file),
        FunctionTool(func=update_file),
        FunctionTool(func=list_directory),
        FunctionTool(func=print_affirming_message),
        FunctionTool(func=create_plan),
        FunctionTool(func=update_plan),
        FunctionTool(func=reset_plan),
        FunctionTool(func=get_plan),
    ],
)
