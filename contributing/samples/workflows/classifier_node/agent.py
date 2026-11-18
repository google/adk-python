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
from google.adk import Context
from google.adk import Event
from google.adk import Workflow
from google.adk.labs.classifiers import Binary
from google.adk.labs.classifiers import Choice
from google.adk.labs.classifiers import ClassifierNode
from google.adk.labs.classifiers import LlmClassifier

# Any BaseClassifier works here, such as a dedicated decision model.
classifier = LlmClassifier(model="gemini-3.5-flash")


def process_input(node_input: str):
  return Event(output=node_input, state={"ticket": node_input})


triage = ClassifierNode(
    name="triage",
    classifier=classifier,
    question=Choice(
        instructions="Which team should handle this support ticket?",
        criteria={
            "billing": "Invoices, refunds, or subscription charges",
            "technical": "Bugs, errors, outages, or API problems",
            "general": "Product questions and anything else",
        },
    ),
    answer_key="triage_answer",
)


is_urgent = ClassifierNode(
    name="is_urgent",
    classifier=classifier,
    question=Binary(
        instructions=(
            "Is this an outage, or does it block the customer's production use?"
        ),
    ),
    threshold=0.7,
    answer_key="urgency_answer",
)


billing_agent = Agent(
    name="billing_agent",
    instruction="You are a billing specialist. Reply to this ticket: {ticket}",
)


faq_agent = Agent(
    name="faq_agent",
    instruction="You answer general product questions. Reply to: {ticket}",
)


def page_oncall(node_input: str, ctx: Context):
  probability = ctx.state["urgency_answer"]["probability"]
  yield Event(
      message=(
          f"Paged the on-call engineer (urgency {probability:.0%}):"
          f" {node_input}"
      )
  )


def eng_queue(node_input: str):
  yield Event(message=f"Filed in the engineering queue: {node_input}")


root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", process_input, triage),
        (
            triage,
            {
                "billing": billing_agent,
                "technical": is_urgent,
                "general": faq_agent,
            },
        ),
        (is_urgent, {True: page_oncall, False: eng_queue}),
    ],
)
