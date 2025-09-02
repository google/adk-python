from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

from tools import get_github_tools
from prompt import GITHUB_PROMPT

github_agent = Agent(
  model=LiteLlm(model="ollama_chat/hf.co/unsloth/Qwen3-8B-GGUF:UD-Q4_K_XL"),
  name='github_agent',
  description='GitHub agent that leverages the Model Context Protocol (MCP) to interact with GitHub repositories.',
  instruction=GITHUB_PROMPT,
  tools=[get_github_tools()],
  output_key="github_agent_output"
)
