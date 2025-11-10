from google.adk.agents.llm_agent import Agent
import datetime

def get_current_time(city: str) -> dict:
    """
    Returns the current time in the specified city.
    """
    # For example purpose only. Ideally, integrate with a world time API.
    time_now = datetime.datetime.now().strftime('%I:%M %p')
    return {"status": "success", "city": city, "time": time_now}

root_agent = Agent(
    model='gemini-2.5-flash',  # Change to your configured model if needed
    name='root_agent',
    description="Tells the current time in any city.",
    instruction=(
        "You are a helpful assistant that tells the current time. "
        "Use the 'get_current_time' tool for this purpose."
    ),
    tools=[get_current_time],
)
