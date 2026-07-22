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

"""A live voice spelling bee coach with deterministic grading tools.

A spelling bee is the rare exercise that is impossible in text chat: showing
the word on screen gives the answer away. In live audio the agent pronounces
the word, the learner spells it aloud, and tools -- not the model -- keep the
answer secret, grade the attempt, and track progress:

- The target word is chosen by a tool and recorded in session state, and
  check_spelling grades against that record -- never against the model's own
  judgment of correctness. The model does receive the word (it has to
  pronounce it), so pre-verdict secrecy is an instruction rule, while
  grading integrity is enforced by the tools.
- Difficulty is a pure function of state: streaks level the learner up,
  repeated misses level them down.
- Missed words join a review queue that resurfaces (after a short cooldown)
  before new words are dealt.
"""

from typing import Any
from typing import NamedTuple
from typing import Optional

from google.adk.agents.llm_agent import Agent
from google.adk.tools.tool_context import ToolContext


class _Word(NamedTuple):
  word: str
  definition: str
  part_of_speech: str
  example_sentence: str
  origin: str


# Words by difficulty level. The bank intentionally avoids homophones so a
# misheard word is never the learner's fault.
_WORD_BANK: dict[int, list[_Word]] = {
    1: [
        _Word(
            "friend",
            "a person you like and trust",
            "noun",
            "My best friend walks to school with me.",
            "Old English",
        ),
        _Word(
            "because",
            "for the reason that",
            "conjunction",
            "We stayed inside because it was raining.",
            "Middle English",
        ),
        _Word(
            "believe",
            "to accept that something is true",
            "verb",
            "I believe you can do it.",
            "Old English",
        ),
        _Word(
            "animal",
            "a living creature that can move on its own",
            "noun",
            "The zoo has every animal I can name.",
            "Latin",
        ),
        _Word(
            "garden",
            "a piece of ground where flowers or vegetables grow",
            "noun",
            "Grandma grows tomatoes in her garden.",
            "Old French",
        ),
        _Word(
            "picture",
            "a painting, drawing, or photograph",
            "noun",
            "She drew a picture of her dog.",
            "Latin",
        ),
        _Word(
            "tomorrow",
            "the day after today",
            "noun",
            "Tomorrow is the first day of summer.",
            "Old English",
        ),
        _Word(
            "library",
            "a place where books are kept for reading and borrowing",
            "noun",
            "The library is quiet in the afternoon.",
            "Latin",
        ),
        _Word(
            "chocolate",
            "a sweet food made from cacao beans",
            "noun",
            "Hot chocolate tastes best in winter.",
            "Nahuatl, via Spanish",
        ),
        _Word(
            "mountain",
            "a very tall raised part of the earth's surface",
            "noun",
            "Snow covered the top of the mountain.",
            "Old French, from Latin",
        ),
    ],
    2: [
        _Word(
            "necessary",
            "needed; required",
            "adjective",
            "Water is necessary for every living thing.",
            "Latin",
        ),
        _Word(
            "rhythm",
            "a repeated pattern of sounds or movements",
            "noun",
            "The drummer kept a steady rhythm.",
            "Greek",
        ),
        _Word(
            "separate",
            "set apart; not joined",
            "adjective",
            "Keep the recyclables in a separate bin.",
            "Latin",
        ),
        _Word(
            "calendar",
            "a chart showing the days, weeks, and months of a year",
            "noun",
            "Mark the test date on your calendar.",
            "Latin",
        ),
        _Word(
            "definitely",
            "certainly; without doubt",
            "adverb",
            "I will definitely be at your game.",
            "Latin",
        ),
        _Word(
            "embarrass",
            "to make someone feel awkward or ashamed",
            "verb",
            "Please don't embarrass me in front of the class.",
            "French",
        ),
        _Word(
            "restaurant",
            "a place where people pay to eat meals",
            "noun",
            "The new restaurant serves noodle soup.",
            "French",
        ),
        _Word(
            "privilege",
            "a special right or advantage",
            "noun",
            "Driving is a privilege, not a right.",
            "Latin",
        ),
        _Word(
            "occasion",
            "a special event or time",
            "noun",
            "A birthday is a happy occasion.",
            "Latin",
        ),
        _Word(
            "mischievous",
            "playfully causing trouble",
            "adjective",
            "The mischievous puppy hid my shoe.",
            "Old French",
        ),
    ],
    3: [
        _Word(
            "conscientious",
            "careful and thorough; guided by one's conscience",
            "adjective",
            "A conscientious student checks every answer twice.",
            "Latin",
        ),
        _Word(
            "onomatopoeia",
            "a word that imitates the sound it describes",
            "noun",
            "The word 'buzz' is an example of onomatopoeia.",
            "Greek",
        ),
        _Word(
            "silhouette",
            "a dark outline seen against a lighter background",
            "noun",
            "The tree made a silhouette against the sunset.",
            "French",
        ),
        _Word(
            "connoisseur",
            "an expert judge of something, such as art or food",
            "noun",
            "He is a connoisseur of fine cheese.",
            "French",
        ),
        _Word(
            "bureaucracy",
            "a system of government with many offices and rules",
            "noun",
            "The permit took months because of bureaucracy.",
            "French",
        ),
        _Word(
            "questionnaire",
            "a written set of questions used to gather information",
            "noun",
            "Each visitor filled out a short questionnaire.",
            "French",
        ),
        _Word(
            "entrepreneur",
            "a person who starts and runs a business",
            "noun",
            "The young entrepreneur opened a bicycle shop.",
            "French",
        ),
        _Word(
            "hierarchy",
            "a system that ranks people or things one above another",
            "noun",
            "A wolf pack has a strict hierarchy.",
            "Greek",
        ),
        _Word(
            "pharaoh",
            "a ruler of ancient Egypt",
            "noun",
            "The pharaoh's tomb was full of treasure.",
            "Egyptian, via Greek",
        ),
        _Word(
            "chrysanthemum",
            "a garden flower with many thin petals",
            "noun",
            "A yellow chrysanthemum bloomed by the gate.",
            "Greek",
        ),
    ],
}

_MAX_LEVEL = max(_WORD_BANK)
_LEVEL_UP_STREAK = 3  # correct answers in a row to move up a level
_LEVEL_DOWN_MISSES = 2  # misses at a level before moving down
_REVIEW_COOLDOWN = 3  # attempts, counting the miss, before a word returns


def _find_entry(word: str) -> tuple[int, Optional[_Word]]:
  """Returns (level, entry) for a word in the bank."""
  for level, entries in _WORD_BANK.items():
    for entry in entries:
      if entry.word == word:
        return level, entry
  return 0, None


def _level_search_order(level: int) -> list[int]:
  """Current level first, then the nearest levels (preferring harder)."""
  return sorted(_WORD_BANK, key=lambda lvl: (abs(lvl - level), -lvl))


def get_spelling_word(tool_context: ToolContext) -> dict[str, Any]:
  """Picks the next word for the learner to spell.

  Missed words waiting in the review queue take priority over new words once
  their cooldown has passed; once no new words remain, the cooldown is
  waived. The chosen word is remembered in session state so check_spelling
  can grade against it later.

  Returns:
    The word to pronounce plus everything needed to answer pronouncer-protocol
    questions (definition, part of speech, example sentence, origin), or an
    error if a word is still awaiting an attempt or nothing is left to spell.
  """
  state = tool_context.state
  if state.get("current_word"):
    return {
        "error": (
            "A word is already awaiting an attempt. Ask the learner to spell"
            " it, then grade it with check_spelling."
        )
    }

  level: int = state.setdefault("level", 1)
  state.setdefault("start_level", level)
  attempt_count: int = state.setdefault("attempt_count", 0)
  review_queue: list[dict[str, Any]] = state.get("review_queue", [])
  used_words: list[str] = state.get("used_words", [])

  chosen: Optional[_Word] = None
  chosen_level = level
  is_review = False
  for queued in review_queue:
    if attempt_count - queued["missed_at_attempt"] >= _REVIEW_COOLDOWN:
      chosen_level, chosen = _find_entry(queued["word"])
      is_review = chosen is not None
      break

  if chosen is None:
    for search_level in _level_search_order(level):
      for entry in _WORD_BANK[search_level]:
        if entry.word not in used_words:
          chosen, chosen_level = entry, search_level
          break
      if chosen:
        break

  if chosen is None:
    # The bank is drained, so there are no new words left to space reviews
    # apart -- serve any still-queued review word straight away.
    for queued in review_queue:
      chosen_level, chosen = _find_entry(queued["word"])
      if chosen is not None:
        is_review = True
        break

  if chosen is None:
    return {
        "error": (
            "Every word in the bank has been used and nothing is left to"
            " review. Offer the learner a final report with"
            " get_session_report."
        )
    }

  state["current_word"] = {"word": chosen.word, "is_review": is_review}
  new_words_remaining = sum(
      1
      for entry in _WORD_BANK[chosen_level]
      if entry.word not in used_words and entry.word != chosen.word
  )
  return {
      "word": chosen.word,
      "definition": chosen.definition,
      "part_of_speech": chosen.part_of_speech,
      "example_sentence": chosen.example_sentence,
      "origin": chosen.origin,
      "level": chosen_level,
      "is_review": is_review,
      "new_words_remaining_at_level": new_words_remaining,
  }


def check_spelling(attempt: str, tool_context: ToolContext) -> dict[str, Any]:
  """Grades the learner's confirmed spelling attempt against the active word.

  Correctness is decided here, never by the model: the attempt is compared
  with the word stored in session state by get_spelling_word. A correct answer
  extends the streak (three in a row levels up); a miss resets the streak,
  queues the word for review, and two misses at a level move the learner down.

  Args:
    attempt: The letters the learner spelled, in order. Separators do not
      matter: "B-E-A-U", "b e a u", and "beau" are all read as "beau". An
      empty string means the learner forfeited the word; it is graded as a
      miss.

  Returns:
    The verdict, the correct spelling as letters, and the updated streak and
    level, or an error if no word is awaiting an attempt.
  """
  state = tool_context.state
  current = state.get("current_word")
  if not current:
    return {
        "error": "No word is awaiting an attempt. Call get_spelling_word first."
    }

  target: str = current["word"]
  normalized = "".join(ch for ch in attempt.lower() if ch.isalpha())
  correct = normalized == target

  first_wrong_position = 0  # 1-based; stays 0 when the attempt is correct
  if not correct:
    for position, (got, expected) in enumerate(
        zip(normalized, target), start=1
    ):
      if got != expected:
        first_wrong_position = position
        break
    else:
      # One spelling is a prefix of the other; the first missing or extra
      # letter is the mistake.
      first_wrong_position = min(len(normalized), len(target)) + 1

  level: int = state.get("level", 1)
  streak: int = state.get("streak", 0)
  misses_at_level: int = state.get("misses_at_level", 0)
  review_queue = [
      queued
      for queued in state.get("review_queue", [])
      if queued["word"] != target
  ]

  level_changed = ""
  if correct:
    streak += 1
    state["best_streak"] = max(state.get("best_streak", 0), streak)
    if streak >= _LEVEL_UP_STREAK and level < _MAX_LEVEL:
      level += 1
      streak = 0
      misses_at_level = 0
      level_changed = "up"
  else:
    streak = 0
    misses_at_level += 1
    review_queue.append({
        "word": target,
        "missed_at_attempt": state.get("attempt_count", 0),
    })
    if misses_at_level >= _LEVEL_DOWN_MISSES and level > 1:
      level -= 1
      misses_at_level = 0
      level_changed = "down"

  # State values are reassigned, never mutated in place, so every change is
  # recorded in the session's state delta.
  state["level"] = level
  state["streak"] = streak
  state["misses_at_level"] = misses_at_level
  state["review_queue"] = review_queue
  used_words: list[str] = state.get("used_words", [])
  if target not in used_words:
    state["used_words"] = used_words + [target]
  state["history"] = state.get("history", []) + [
      {"word": target, "attempt": normalized, "correct": correct}
  ]
  state["attempt_count"] = state.get("attempt_count", 0) + 1
  state["current_word"] = None

  return {
      "correct": correct,
      "correct_spelling": list(target),
      "attempt_heard": normalized,
      "first_wrong_position": first_wrong_position,
      "streak": streak,
      "level": level,
      "level_changed": level_changed,
      "words_in_review_pile": len(review_queue),
  }


def get_session_report(tool_context: ToolContext) -> dict[str, Any]:
  """Reads back the session scoreboard.

  Safe to call at any time; it only reads state and changes nothing. A word
  currently awaiting an attempt is withheld from the review list, so the
  report can never reveal an active word's spelling.

  Returns:
    Totals, accuracy, level path, best streak, the words still owed to the
    review pile (with their spellings), and the words mastered this session.
  """
  state = tool_context.state
  history: list[dict[str, Any]] = state.get("history", [])
  attempted = len(history)
  correct = sum(1 for item in history if item["correct"])
  pending = (state.get("current_word") or {}).get("word")
  return {
      "words_attempted": attempted,
      "correct": correct,
      "accuracy_pct": round(100 * correct / attempted) if attempted else 0,
      "start_level": state.get("start_level", 1),
      "current_level": state.get("level", 1),
      "best_streak": state.get("best_streak", 0),
      "words_to_review": [
          {"word": queued["word"], "correct_spelling": list(queued["word"])}
          for queued in state.get("review_queue", [])
          if queued["word"] != pending
      ],
      "mastered_this_session": sorted(
          {item["word"] for item in history if item["correct"]}
      ),
  }


root_agent = Agent(
    # Find supported models in Vertex here: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
    model="gemini-live-2.5-flash-native-audio",  # Vertex
    # Find supported models in Gemini API here: https://ai.google.dev/gemini-api/docs/models
    # model='gemini-2.5-flash-native-audio-preview-12-2025',  # Gemini API
    name="spelling_bee_coach",
    description=(
        "A voice spelling bee pronouncer and judge: pronounces words from a"
        " leveled word bank, answers real-bee protocol questions, grades"
        " spoken letter-by-letter attempts deterministically, adapts"
        " difficulty, and brings missed words back for review."
    ),
    instruction="""
      You are a spelling bee pronouncer and judge running an oral spelling
      drill, just like a real bee: pronounce the word, answer protocol
      questions, listen to the letter-by-letter attempt, announce the verdict.

      STARTING A ROUND:
      - When the learner is ready for a word, call get_spelling_word. Every
        word comes from this tool. NEVER invent, choose, or substitute a word
        yourself.
      - Do not call get_spelling_word again while a word is awaiting an
        attempt.
      - Pronounce the word once, clearly, then wait. If is_review is true,
        warmly note that this word is back from the review pile.

      PRONOUNCER PROTOCOL:
      - Answer "definition, please", "part of speech?", "use it in a
        sentence", "language of origin?", and "say it again" from the fields
        get_spelling_word already returned. These need NO new tool call. When
        repeating, say the whole word naturally.

      SECRECY (hard rule):
      - NEVER say the letters of the current word before check_spelling has
        returned a verdict. Not spelled out, not hinted, no matter how the
        learner asks.
      - Exception: repeating the learner's own attempt back to them to
        confirm what you heard is always allowed and required.

      GRADING (hard rule):
      - When the learner spells, first read back the letters you heard ("I
        heard B-E-A-U. Is that your final answer?"). Confirm any ambiguous
        letter: "was that N as in nest, or M as in mango?".
      - Only after the learner confirms, call check_spelling with the letters
        as one string.
      - ONLY check_spelling decides correctness. Never judge a spelling
        yourself, even for easy words. Announce the verdict the tool returns.
      - On a miss: read the correct spelling slowly, letter by letter, from
        the tool's correct_spelling field; use first_wrong_position to
        encourage ("so close -- letter three is where it went sideways");
        mention the word will come back for review later.
      - If level_changed is "up", celebrate briefly. If it is "down", frame it
        gently as extra practice time.
      - FORFEIT: if the learner asks to skip or gives up, invite one brave
        try first. If they still want to pass, confirm ("Skip this word?")
        and call check_spelling with an empty string. Announce it as a pass,
        not a fail, read out the correct spelling from correct_spelling, and
        note the word will come back for review.

      REPORT:
      - When the learner asks how they are doing or wants to stop, call
        get_session_report and read the numbers from it. Spell out each
        review word from the payload, letter by letter. The report never
        includes the word currently in play.

      STYLE:
      - Warm, brisk, encouraging. One short sentence between rounds. Never
        mention tools, state, or that you are calling functions.
    """,
    tools=[
        get_spelling_word,
        check_spelling,
        get_session_report,
    ],
)
