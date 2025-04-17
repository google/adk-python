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

# https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner

from google.adk import Agent

# Agent that selects the best city for the trip.
identify_agent = Agent(
    name='identify_agent',
    description='Select the best city based on weather, season, and prices.',
    instruction="""
  Analyze and select the best city for the trip based
  on specific criteria such as weather patterns, seasonal
  events, and travel costs. This task involves comparing
  multiple cities, considering factors like current weather
  conditions, upcoming cultural or seasonal events, and
  overall travel expenses.

  Your final answer must be a detailed
  report on the chosen city, and everything you found out
  about it, including the actual flight costs, weather
  forecast and attractions.

  Traveling from: {origin}
  City Options: {cities}
  Trip Date: {range}
  Traveler Interests: {interests}
""",
)

# Agent that gathers information about the city.
gather_agent = Agent(
    name='gather_agent',
    description='Provide the BEST insights about the selected city',
    instruction="""
  As a local expert on this city you must compile an
  in-depth guide for someone traveling there and wanting
  to have THE BEST trip ever!
  Gather information about key attractions, local customs,
  special events, and daily activity recommendations.
  Find the best spots to go to, the kind of place only a
  local would know.
  This guide should provide a thorough overview of what
  the city has to offer, including hidden gems, cultural
  hotspots, must-visit landmarks, weather forecasts, and
  high level costs.

  The final answer must be a comprehensive city guide,
  rich in cultural insights and practical tips,
  tailored to enhance the travel experience.

  Trip Date: {range}
  Traveling from: {origin}
  Traveler Interests: {interests}
""",
)

# Agent that plans the trip.
plan_agent = Agent(
    name='plan_agent',
    description="""Create the most amazing travel itineraries with budget and
    packing suggestions for the city""",
    instruction="""
  Expand this guide into a full 7-day travel
  itinerary with detailed per-day plans, including
  weather forecasts, places to eat, packing suggestions,
  and a budget breakdown.

  You MUST suggest actual places to visit, actual hotels
  to stay and actual restaurants to go to.

  This itinerary should cover all aspects of the trip,
  from arrival to departure, integrating the city guide
  information with practical travel logistics.

  Your final answer MUST be a complete expanded travel plan,
  formatted as markdown, encompassing a daily schedule,
  anticipated weather conditions, recommended clothing and
  items to pack, and a detailed budget, ensuring THE BEST
  TRIP EVER. Be specific and give it a reason why you picked
  each place, what makes them special!

  Trip Date: {range}
  Traveling from: {origin}
  Traveler Interests: {interests}
""",
)

root_agent = Agent(
    model='gemini-2.0-flash-001',
    name='trip_planner',
    description='Plan the best trip ever',
    instruction="""
  Your goal is to plan the best trip according to information listed above.
  You describe why did you choose the city, list top 3
  attactions and provide a detailed itinerary for each day.""",
    sub_agents=[identify_agent, gather_agent, plan_agent],
)
