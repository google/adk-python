# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk import Agent

def book_flight(destination: str, date: str, passenger_name: str) -> str:
    """Book a flight to a destination for a passenger on a specific date.

    Args:
        destination: The destination city of the flight.
        date: The date of the flight (YYYY-MM-DD).
        passenger_name: The full name of the passenger.

    Returns:
        A string confirming the booking.
    """
    return f"Successfully booked flight to {destination} on {date} for {passenger_name}."

root_agent = Agent(
    model='gemini-2.5-flash',
    name='booking_assistant',
    allow_elicitation=True,
    elicitation_max_turns=3,
    instruction="""
        You are a flight booking assistant.
        If the user asks you to book a flight, you MUST gather all three required parameters:
        1. destination
        2. date
        3. passenger_name

        If the user does not specify ALL three parameters, you MUST call the `trigger_elicitation` tool 
        to request the missing information from the user. Do not attempt to book a flight without 
        all three details.

        Once you have all three details, call the `book_flight` tool.
    """,
    tools=[book_flight]
)
