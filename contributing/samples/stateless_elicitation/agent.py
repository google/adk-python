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

def book_hotel(location: str, checkin_date: str, guest_name: str) -> str:
    """Book a hotel reservation for a guest at a specific location and check-in date.

    Args:
        location: The city or location of the hotel.
        checkin_date: The check-in date (YYYY-MM-DD).
        guest_name: The full name of the guest booking the hotel.

    Returns:
        A string confirmation of the booking.
    """
    return f"Successfully booked hotel in {location} on {checkin_date} for {guest_name}."

root_agent = Agent(
    model='gemini-2.5-flash',
    name='hotel_booking_agent',
    allow_elicitation=True,
    elicitation_max_turns=3,
    instruction="""
        You are a helpful hotel booking assistant.
        If the user asks you to book a hotel, you MUST gather all three required parameters:
        1. location
        2. checkin_date
        3. guest_name

        If the user does not specify ALL three parameters, you MUST call the `trigger_elicitation` tool 
        to request the missing information. Do not attempt to book a hotel without all three details.

        Once you have gathered all three details, call the `book_hotel` tool to complete the booking.
    """,
    tools=[book_hotel]
)
