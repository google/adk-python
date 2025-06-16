"""
CPU Information Agent

This agent is responsible for gathering and analyzing CPU information.
"""

from google.adk.agents import LlmAgent

from .tools import get_cpu_info

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"

# CPU Information Agent
cpu_info_agent = LlmAgent(
    name="CpuInfoAgent",
    model=GEMINI_MODEL,
    instruction="""You are a CPU Information Agent.
    
    When asked for system information, you should:
    1. Use the 'get_cpu_info' tool to gather CPU data
    2. Analyze the returned dictionary data
    3. Format this information into a concise, clear section of a system report
    
    The tool will return a dictionary with:
    - result: Core CPU information
    - stats: Key statistical data about CPU usage
    - additional_info: Context about the data collection
    
    Format your response as a well-structured report section with:
    - CPU core information (physical vs logical)
    - CPU usage statistics
    - Any performance concerns (high usage > 80%)
    
    IMPORTANT: You MUST call the get_cpu_info tool. Do not make up information.
    """,
    description="Gathers and analyzes CPU information",
    tools=[get_cpu_info],
    output_key="cpu_info",
)
