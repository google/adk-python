import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """

    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

def get_city_population(city: str) -> dict:
    """Returns a hardcoded population estimate for a few cities."""
    city_data = {
        "new york": "New York has a population of approximately 8.5 million people.",
        "tokyo": "Tokyo has a population of approximately 14 million people.",
        "paris": "Paris has a population of approximately 2.1 million people.",
        "london": "London has a population of approximately 9 million people.",
        "beijing": "Beijing has a population of approximately 21 million people."
    }

    city_key = city.lower()
    if city_key in city_data:
        return {
            "status": "success",
            "report": city_data[city_key]
        }
    else:
        return {
            "status": "error",
            "error_message": f"Population data for '{city}' is not available."
        }

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions about the time, weather, and city populations.",
    instruction="You are a helpful agent who can answer user questions about the time, weather, and city populations.",
    tools=[get_weather, get_current_time, get_city_population],
)
