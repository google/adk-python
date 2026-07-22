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

import re
from typing import Any

from google.adk import Agent
from google.adk import Event
from google.adk import Workflow
from google.adk.workflow import JoinNode

# Shared verbatim by every council agent so the system instruction is
# byte-identical across all calls in a session, keeping the cacheable
# prefix stable. Per-seat persona and stage tasks live in the dynamic
# `instruction`, which ADK sends as user content when `static_instruction`
# is set.
_COUNCIL_PROTOCOL = (
    "You are a seat on an LLM council that answers questions by"
    " deliberating in three stages: each member answers the question"
    " independently (opinion), each member ranks the anonymized answers of"
    " the whole council (review), and the chairman synthesizes the"
    " opinions, reviews, and tallied leaderboard into one final answer"
    " (synthesis). Your persona and current task are described in the"
    " conversation. Follow them exactly."
)

_PERSONAS = {
    "pragmatist": (
        "a pragmatic engineer. You care about shipping on time, real-world"
        " constraints, and the simplest thing that works today."
    ),
    "skeptic": (
        "a risk-focused skeptic. You hunt for security holes, failure modes,"
        " and hidden costs that everyone else missed."
    ),
    "visionary": (
        "a first-principles visionary. You reason about long-term"
        " consequences, architecture, and what the ideal solution looks like."
    ),
}


def _make_opinion_agent(name: str, persona: str) -> Agent:
  """Creates a council member that gives an independent first opinion."""
  return Agent(
      name=name,
      description=f"Council member giving a first opinion: {name}.",
      static_instruction=_COUNCIL_PROTOCOL,
      instruction=(
          f"You are {persona}\n\n"
          "Your task: answer the question in the user message from your own"
          " point of view. Be concrete and keep it under 150 words."
      ),
  )


def _make_review_agent(name: str, persona: str) -> Agent:
  """Creates a council member that reviews the anonymized opinions."""
  return Agent(
      name=f"{name}_review",
      description=f"Council member reviewing peer answers: {name}.",
      static_instruction=_COUNCIL_PROTOCOL,
      instruction=(
          f"You are {persona}\n\n"
          "Your task: review the council's anonymized responses to the"
          " question in the user message:\n"
          "{transcript}\n\n"
          "One of these responses is yours, but you cannot tell which."
          " Judge each response by how well it serves the person asking, not"
          " by how similar it is to the answer you would have written."
          " Evaluate each response for accuracy and insight in 1-2 sentences,"
          " judging on merit only. Do not write a new answer of your own."
          " Then end with EXACTLY one final line in this format, best first,"
          " labels only:\n"
          "FINAL RANKING: B > A > C"
      ),
  )


def record_question(node_input: str):
  """Stores the user's question in state for instruction templating."""
  return Event(state={"question": node_input})


join_opinions = JoinNode(name="join_opinions")
join_reviews = JoinNode(name="join_reviews")


def anonymize_opinions(node_input: dict[str, Any]):
  """Relabels the opinions so reviewers cannot tell who wrote what."""
  members = sorted(node_input)
  labels = {member: chr(ord("A") + i) for i, member in enumerate(members)}
  transcript = "\n\n".join(
      f"### Response {labels[member]}\n{node_input[member]}"
      for member in members
  )
  roster = "\n".join(
      f"Response {labels[member]}: {member}" for member in members
  )
  return Event(state={"transcript": transcript, "roster": roster})


_RANKING_RE = re.compile(r"FINAL RANKING:\s*([A-Z](?:\s*>\s*[A-Z])+)")


def tally_ballots(node_input: dict[str, Any]):
  """Tallies the FINAL RANKING ballots deterministically, in code.

  The average-rank math is done here rather than by the chairman so the
  leaderboard can never be miscalculated or hallucinated.
  """
  reviews = [node_input[reviewer] for reviewer in sorted(node_input)]
  positions: dict[str, list[int]] = {}
  ballots = 0
  for review in reviews:
    match = _RANKING_RE.search(review)
    if not match:
      continue
    ballots += 1
    for position, label in enumerate(match.group(1).split(">"), start=1):
      positions.setdefault(label.strip(), []).append(position)

  if positions:
    rows = "\n".join(
        f"| Response {label} | {sum(ranks) / len(ranks):.2f} |"
        for label, ranks in sorted(
            positions.items(), key=lambda item: sum(item[1]) / len(item[1])
        )
    )
    leaderboard = (
        f"Council leaderboard, tallied from {ballots} ballots (lower average"
        " rank is better):\n\n"
        "| Response | Average rank |\n| :--- | :--- |\n"
        f"{rows}"
    )
  else:
    leaderboard = "No parseable FINAL RANKING ballots were returned."

  full_reviews = "\n\n".join(
      f"### Peer review {i}\n{review}"
      for i, review in enumerate(reviews, start=1)
  )
  return f"{leaderboard}\n\n{full_reviews}"


chairman = Agent(
    name="council_chairman",
    description="Synthesizes the council's opinions into a final answer.",
    static_instruction=_COUNCIL_PROTOCOL,
    instruction=(
        "You are the chairman of the council.\n\n"
        "The question under discussion is:\n{question}\n\n"
        "The anonymized council responses are:\n{transcript}\n\n"
        "The identity behind each response is:\n{roster}\n\n"
        "The user message contains the pre-tallied council leaderboard and"
        " the peer reviews. Reply in Markdown with exactly these sections:\n"
        "## Chairman's Answer — the complete answer to the question,"
        " synthesized from the council's material. Weight the top-ranked"
        " responses most, but adopt any uniquely good point regardless of"
        " rank.\n"
        "## Where the Council Agreed — bullets of consensus points.\n"
        "## Where the Council Split — real disagreements and your ruling.\n"
        "## Council Verdict — the leaderboard result, de-anonymized using"
        " the roster. Reproduce the leaderboard table exactly as given; do"
        " not recompute it.\n"
        "Keep the entire reply under 500 words."
    ),
)

opinion_agents = tuple(
    _make_opinion_agent(name, persona) for name, persona in _PERSONAS.items()
)
review_agents = tuple(
    _make_review_agent(name, persona) for name, persona in _PERSONAS.items()
)

root_agent = Workflow(
    name="llm_council",
    edges=[
        (
            "START",
            record_question,
            opinion_agents,
            join_opinions,
            anonymize_opinions,
        ),
        (
            anonymize_opinions,
            review_agents,
            join_reviews,
            tally_ballots,
            chairman,
        ),
    ],
)
