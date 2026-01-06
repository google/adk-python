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
"""Tools for Vote Taker Agent."""

from datetime import datetime
import os
from typing import Any
from typing import Dict
from typing import Optional

from google.adk.tools import ToolContext
from google.cloud import bigquery

# Configuration
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
BQ_DATASET = os.getenv("BQ_DATASET", "")
BQ_VOTES_TABLE = os.getenv("BQ_VOTES_TABLE", "")
LOCAL_MODE = os.getenv("LOCAL_MODE", "true").lower() == "true"

# In-memory storage for local development
local_votes = []

# Voting options for multiple rounds
VOTING_ROUNDS = {
    "round1": {
        "question": "What would you like to see next?",
        "options": {
            "A": {
                "title": "Computer Use",
                "description": "Autonomous browser control with Gemini 2.5",
            },
            "B": {
                "title": "A2A Multi-Agent",
                "description": "Agent-to-Agent coordination patterns",
            },
            "C": {
                "title": "Production Observability",
                "description": "Monitoring and debugging at scale",
            },
        },
    },
    "round2": {
        "question": "What shall we add to this image now?",
        "options": {
            "A": {
                "title": "Add butterflies",
                "description": "Add colorful butterflies around the dog",
            },
            "B": {
                "title": "Add a rainbow",
                "description": "Add a vibrant rainbow in the sky",
            },
            "C": {
                "title": "Add flowers",
                "description": "Add blooming flowers in the grass",
            },
        },
    },
}

# Default to round 1 options for backward compatibility
VOTING_OPTIONS = VOTING_ROUNDS["round1"]["options"]
CURRENT_ROUND = "round1"


def get_voting_options(
    tool_context: ToolContext, round_id: Optional[str] = None
) -> Dict[str, Any]:
  """Returns the current voting options available to the user.

  Args:
      tool_context: ADK tool context
      round_id: Optional round ID (round1, round2, etc.)

  Returns:
      dict: Voting options with titles and descriptions
  """
  print(f"Tool called: get_voting_options - round={round_id or CURRENT_ROUND}")

  active_round = round_id or CURRENT_ROUND

  if active_round not in VOTING_ROUNDS:
    return {"success": False, "error": f"Invalid round ID: {active_round}"}

  round_data = VOTING_ROUNDS[active_round]

  return {
      "success": True,
      "round": active_round,
      "question": round_data["question"],
      "image_url": round_data.get("image_url"),
      "options": round_data["options"],
      "message": round_data["question"],
  }


def set_voting_round(
    round_id: str, tool_context: ToolContext
) -> Dict[str, Any]:
  """Sets the current voting round.

  Args:
      round_id: The round ID to set (round1, round2, etc.)
      tool_context: ADK tool context

  Returns:
      dict: Confirmation with new round details
  """
  global CURRENT_ROUND, VOTING_OPTIONS

  print(f"Tool called: set_voting_round - round={round_id}")

  if round_id not in VOTING_ROUNDS:
    return {"success": False, "error": f"Invalid round ID: {round_id}"}

  CURRENT_ROUND = round_id
  VOTING_OPTIONS = VOTING_ROUNDS[round_id]["options"]

  return {
      "success": True,
      "round": round_id,
      "question": VOTING_ROUNDS[round_id]["question"],
      "message": f"Voting round changed to: {round_id}",
  }


def store_vote_to_bigquery(
    vote_choice: str,
    user_id: str,
    additional_feedback: Optional[str],
    tool_context: ToolContext,
    round_id: Optional[str] = None,
) -> Dict[str, Any]:
  """Stores a validated vote to BigQuery (or local storage in dev mode).

  Args:
      vote_choice: The vote option (A, B, or C)
      user_id: Unique identifier for the voter
      additional_feedback: Optional feedback from the user
      tool_context: ADK tool context
      round_id: Optional round ID for the vote

  Returns:
      dict: Confirmation with vote details
  """
  print(
      f"Tool called: store_vote_to_bigquery - vote={vote_choice},"
      f" user={user_id}, round={round_id or CURRENT_ROUND}"
  )

  active_round = round_id or CURRENT_ROUND
  active_options = VOTING_ROUNDS[active_round]["options"]

  # Validate vote choice
  vote = vote_choice.upper()
  if vote not in active_options:
    return {
        "success": False,
        "error": "Invalid vote choice. Must be A, B, or C.",
        "vote": vote,
    }

  # Create vote record
  vote_record = {
      "vote": vote,
      "user_id": user_id,
      "additional_feedback": additional_feedback or "",
      "timestamp": datetime.utcnow().isoformat(),
      "round": active_round,
      "option_title": active_options[vote]["title"],
  }

  if LOCAL_MODE:
    # Store locally for development
    local_votes.append(vote_record)

    return {
        "success": True,
        "message": (
            f"✅ Vote recorded for Option {vote}:"
            f" {active_options[vote]['title']}!"
        ),
        "vote_details": vote_record,
        "total_votes": len(local_votes),
    }
  else:
    # Store to BigQuery for production
    try:
      client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)
      table_id = f"{GOOGLE_CLOUD_PROJECT}.{BQ_DATASET}.{BQ_VOTES_TABLE}"

      errors = client.insert_rows_json(table_id, [vote_record])

      if errors:
        return {
            "success": False,
            "error": "Failed to store vote to database",
            "details": str(errors),
        }

      return {
          "success": True,
          "message": (
              f"✅ Vote recorded for Option {vote}:"
              f" {active_options[vote]['title']}!"
          ),
          "vote_details": vote_record,
      }

    except Exception as e:
      return {
          "success": False,
          "error": "Database error occurred",
          "details": str(e),
      }


def get_vote_summary(tool_context: ToolContext) -> Dict[str, Any]:
  """Returns a summary of all votes collected so far.

  Returns:
      dict: Vote counts and summary statistics
  """
  print("Tool called: get_vote_summary")

  if LOCAL_MODE:
    # Calculate summary from local storage
    vote_counts = {"A": 0, "B": 0, "C": 0}

    for vote_record in local_votes:
      vote = vote_record.get("vote")
      if vote in vote_counts:
        vote_counts[vote] += 1

    total_votes = len(local_votes)

    # Determine winner
    winner = None
    if total_votes > 0:
      winner = max(vote_counts, key=vote_counts.get)

    return {
        "success": True,
        "total_votes": total_votes,
        "breakdown": vote_counts,
        "winner": winner,
        "winner_title": VOTING_OPTIONS[winner]["title"] if winner else None,
        "message": (
            f"Total votes: {total_votes}. Leading option: {winner}"
            if winner
            else "No votes yet."
        ),
    }
  else:
    # Query BigQuery for production
    try:
      client = bigquery.Client(project=GOOGLE_CLOUD_PROJECT)

      query = f"""
                SELECT
                    vote,
                    COUNT(*) as count
                FROM `{GOOGLE_CLOUD_PROJECT}.{BQ_DATASET}.{BQ_VOTES_TABLE}`
                GROUP BY vote
                ORDER BY count DESC
            """

      results = client.query(query).result()

      vote_counts = {"A": 0, "B": 0, "C": 0}
      for row in results:
        vote_counts[row.vote] = row.count

      total_votes = sum(vote_counts.values())
      winner = (
          max(vote_counts, key=vote_counts.get) if total_votes > 0 else None
      )

      return {
          "success": True,
          "total_votes": total_votes,
          "breakdown": vote_counts,
          "winner": winner,
          "winner_title": VOTING_OPTIONS[winner]["title"] if winner else None,
          "message": (
              f"Total votes: {total_votes}. Leading option: {winner}"
              if winner
              else "No votes yet."
          ),
      }

    except Exception as e:
      return {
          "success": False,
          "error": "Failed to retrieve vote summary",
          "details": str(e),
      }
