"""
System Report Synthesizer Agent

This agent is responsible for synthesizing information from other agents
to create a comprehensive system health report.
"""

from google.adk.agents import LlmAgent

# --- Constants ---
GEMINI_MODEL = "gemini-2.0-flash"

# System Report Synthesizer Agent
system_report_synthesizer = LlmAgent(
    name="SystemReportSynthesizer",
    model=GEMINI_MODEL,
    instruction="""You are a System Report Synthesizer.
    
    Your task is to create a comprehensive system health report by combining information from:
    - CPU information: {cpu_info}
    - Memory information: {memory_info}
    - Disk information: {disk_info}
    
    Create a well-formatted report with:
    1. An executive summary at the top with overall system health status
    2. Sections for each component with their respective information
    3. Recommendations based on any concerning metrics
    
    Use markdown formatting to make the report readable and professional.
    Highlight any concerning values and provide practical recommendations.
    """,
    description="Synthesizes all system information into a comprehensive report",
)
