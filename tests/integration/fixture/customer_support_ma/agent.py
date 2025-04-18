# Copyright 2025 Google LLC
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

import os
import sys

from google.adk import Agent
from google.adk.agents import RemoteAgent
from google.adk.examples import Example
from google.adk.sessions import Session
from google.genai import types


def reset_data():
  pass


def fetch_user_flight_information(customer_email: str) -> str:
  """Fetch user flight information."""
  return """
[{"ticket_no": "7240005432906569", "book_ref": "C46E9F", "flight_id": 19250, "flight_no": "LX0112", "departure_airport": "CDG", "arrival_airport": "BSL", "scheduled_departure": "2024-12-30 12:09:03.561731-04:00", "scheduled_arrival": "2024-12-30 13:39:03.561731-04:00", "seat_no": "18E", "fare_conditions": "Economy"}]
"""


def list_customer_flights(customer_email: str) -> str:
  return "{'flights': [{'book_ref': 'C46E9F'}]}"


def update_ticket_to_new_flight(ticket_no: str, new_flight_id: str) -> str:
  return 'OK, your ticket has been updated.'


def lookup_company_policy(topic: str) -> str:
  """Lookup policies for flight cancelation and rebooking."""
  return """
1. How can I change my booking?
	* The ticket number must start with 724 (SWISS ticket no./plate).
	* The ticket was not paid for by barter or voucher (there are exceptions to voucher payments; if the ticket was paid for in full by voucher, then it may be possible to rebook online under certain circumstances. If it is not possible to rebook online because of the payment method, then you will be informed accordingly during the rebooking process).
	* There must be an active flight booking for your ticket. It is not possible to rebook open tickets or tickets without the corresponding flight segments online at the moment.
	* It is currently only possible to rebook outbound (one-way) tickets or return tickets with single flight routes (point-to-point).
"""


def search_flights(
    departure_airport: str = None,
    arrival_airport: str = None,
    start_time: str = None,
    end_time: str = None,
) -> list[dict]:
  return """
[{"flight_id": 19238, "flight_no": "LX0112", "scheduled_departure": "2024-05-08 12:09:03.561731-04:00", "scheduled_arrival": "2024-05-08 13:39:03.561731-04:00", "departure_airport": "CDG", "arrival_airport": "BSL", "status": "Scheduled", "aircraft_code": "SU9", "actual_departure": null, "actual_arrival": null}, {"flight_id": 19242, "flight_no": "LX0112", "scheduled_departure": "2024-05-09 12:09:03.561731-04:00", "scheduled_arrival": "2024-05-09 13:39:03.561731-04:00", "departure_airport": "CDG", "arrival_airport": "BSL", "status": "Scheduled", "aircraft_code": "SU9", "actual_departure": null, "actual_arrival": null}]"""


def search_hotels(
    location: str = None,
    price_tier: str = None,
    checkin_date: str = None,
    checkout_date: str = None,
) -> list[dict]:
  return """
[{"id": 1, "name": "Hilton Basel", "location": "Basel", "price_tier": "Luxury"}, {"id": 3, "name": "Hyatt Regency Basel", "location": "Basel", "price_tier": "Upper Upscale"}, {"id": 8, "name": "Holiday Inn Basel", "location": "Basel", "price_tier": "Upper Midscale"}]
"""


def book_hotel(hotel_name: str) -> str:
  return 'OK, your hotel has been booked.'


def before_model_call(agent: Agent, session: Session, user_message):
  if 'expedia' in user_message.lower():
    response = types.Content(
        role='model',
        parts=[types.Part(text="Sorry, I can't answer this question.")],
    )
    return response
  return None


def after_model_call(
    agent: Agent, session: Session, content: types.Content
) -> bool:
  model_message = content.parts[0].text
  if 'expedia' in model_message.lower():
    response = types.Content(
        role='model',
        parts=[types.Part(text="Sorry, I can't answer this question.")],
    )
    return response
  return None


flight_agent = Agent(
    model='gemini-1.5-pro',
    name='flight_agent',
    description='Handles flight information, policy and updates',
    instruction="""
      You are a specialized assistant for handling flight updates.
        The primary assistant delegates work to you whenever the user needs help updating their bookings.
      Confirm the updated flight details with the customer and inform them of any additional fees.
        When searching, be persistent. Expand your query bounds if the first search returns no results.
        Remember that a booking isn't completed until after the relevant tool has successfully been used.
      Do not waste the user's time. Do not make up invalid tools or functions.
""",
    tools=[
        list_customer_flights,
        lookup_company_policy,
        fetch_user_flight_information,
        search_flights,
        update_ticket_to_new_flight,
    ],
)

hotel_agent = Agent(
    model='gemini-1.5-pro',
    name='hotel_agent',
    description='Handles hotel information and booking',
    instruction="""
      You are a specialized assistant for handling hotel bookings.
      The primary assistant delegates work to you whenever the user needs help booking a hotel.
      Search for available hotels based on the user's preferences and confirm the booking details with the customer.
        When searching, be persistent. Expand your query bounds if the first search returns no results.
""",
    tools=[search_hotels, book_hotel],
)


idea_agent = RemoteAgent(
    model='gemini-1.5-pro',
    name='idea_agent',
    description='Provide travel ideas base on the destination.',
    url='http://localhost:8000/agent/run',
)


root_agent = Agent(
    model='gemini-1.5-pro',
    name='root_agent',
    instruction="""
      You are a helpful customer support assistant for Swiss Airlines.
""",
    sub_agents=[flight_agent, hotel_agent, idea_agent],
    flow='auto',
    examples=[
        Example(
            input=types.Content(
                role='user',
                parts=[types.Part(text='How were you built?')],
            ),
            output=[
                types.Content(
                    role='model',
                    parts=[
                        types.Part(
                            text='I was built with the best agent framework.'
                        )
                    ],
                )
            ],
        ),
    ],
)
