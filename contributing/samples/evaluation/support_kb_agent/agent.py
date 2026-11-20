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

"""A small support agent that answers from a fixed knowledge base.

The `search_docs` tool ranks six in-memory articles by word overlap with the
query, so retrieval is deterministic and needs no external service. It is used
by the retrieval_recall evaluation sample.
"""

import re

from google.adk import Agent

_TOP_K = 2

_ARTICLES = [
    {
        "id": "returns-policy",
        "title": "Returns policy",
        "text": (
            "You can return most items within 30 days of delivery for a full"
            " refund. Items must be unused and in their original packaging."
        ),
    },
    {
        "id": "refund-timing",
        "title": "When refunds arrive",
        "text": (
            "Refunds are issued to the original payment method within 5"
            " business days after we receive the returned item."
        ),
    },
    {
        "id": "shipping-times",
        "title": "Domestic shipping times",
        "text": (
            "Standard domestic shipping takes 3 to 5 business days. Express"
            " shipping takes 1 to 2 business days."
        ),
    },
    {
        "id": "international-shipping",
        "title": "International shipping",
        "text": (
            "International shipping takes 7 to 14 business days. Customs"
            " duties are paid by the recipient."
        ),
    },
    {
        "id": "warranty",
        "title": "Warranty cover",
        "text": (
            "All electronics carry a 12 month warranty against manufacturing"
            " defects. Accidental damage is not covered."
        ),
    },
    {
        "id": "account-password",
        "title": "Resetting your password",
        "text": (
            "Use the Forgot password link on the sign-in page. A reset link"
            " is emailed to you and expires after 1 hour."
        ),
    },
]


def _stem(word: str) -> str:
  """Strips a plural or past-tense ending, so `refunds` matches `refund`."""
  for suffix in ("ed", "s"):
    if word.endswith(suffix) and len(word) > len(suffix) + 2:
      return word[: -len(suffix)]
  return word


def _words(text: str) -> set[str]:
  """Returns the stemmed lowercase words of `text`."""
  return {_stem(word) for word in re.findall(r"[a-z0-9]+", text.lower())}


def search_docs(query: str) -> dict:
  """Searches the support knowledge base.

  Args:
    query: What to look for, in a few keywords.

  Returns:
    A dict with a `documents` list. Each document has an `id`, a `title` and
    the article `text`. At most two documents are returned, best match first.
  """
  query_words = _words(query)
  scored = []
  for position, article in enumerate(_ARTICLES):
    overlap = len(
        query_words & _words(article["title"] + " " + article["text"])
    )
    if overlap:
      # Ties keep the order of _ARTICLES, so results are reproducible.
      scored.append((-overlap, position, article))
  scored.sort()
  return {"documents": [article for _, _, article in scored[:_TOP_K]]}


root_agent = Agent(
    name="support_kb_agent",
    description="Answers customer support questions from a knowledge base.",
    instruction=(
        "You are a customer support assistant. Before answering, call"
        " search_docs once with a few keywords from the user's question. Answer"
        " in one or two sentences using only the returned documents. If they"
        " do not contain the answer, say you do not know."
    ),
    tools=[search_docs],
)
