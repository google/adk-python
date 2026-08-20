# Agent 2.0 testing strategy

Sample corpus for the file retrieval fixture. This is illustrative content,
not a real design document.

Agents are tested at three levels:

- Unit tests exercise a single tool, callback or plugin in isolation, with the
  model stubbed out. They are fast and run on every change.
- Integration tests run a whole agent against a live model and assert on the
  sequence of tool calls it produces.
- Evaluation tests replay recorded conversations and score the final response
  against a reference answer.

A change to an agent's tools or instruction needs a unit test. A change to the
run loop needs an integration test as well.
