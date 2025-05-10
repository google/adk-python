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

import pytest
import copy
from google.genai import types
from google.adk.flows.llm_flows import contents as contents_mod

# --- Fixtures and helpers ---
@pytest.fixture
def sample_content():
    return types.Content(role="user", parts=[types.Part(text="hello")])

@pytest.fixture
def sample_event(sample_content):
    from google.adk.events.event import Event
    return Event(author="user", content=copy.deepcopy(sample_content), branch="main", timestamp=1)

@pytest.fixture
def sample_state():
    return {"user_profile": "Alice", "current_task": "testing", "irrelevant": 123}

@pytest.fixture
def sample_config():
    from google.adk.agents.content_config import ContentConfig, SummarizationConfig
    return ContentConfig(
        enabled=True,
        max_events=3,
        summarize=False,
        always_include_last_n=1,
        context_from_state=["user_profile", "current_task"],
        state_template="User: {context}"
    )

@pytest.fixture
def event_with_author(sample_content):
    from google.adk.events.event import Event
    def _make(author, branch, invocation_id):
        return Event(author=author, content=copy.deepcopy(sample_content), branch=branch, invocation_id=invocation_id, timestamp=1)
    return _make

# --- Unit tests for utility functions ---
def test_normalize_limits_basic():
    norm = contents_mod._normalize_limits(5, 2, 10)
    assert norm == (5, 2)
    norm = contents_mod._normalize_limits(None, None, 10)
    assert norm == (None, 0)
    norm = contents_mod._normalize_limits(2, 5, 10)
    assert norm == (2, 2)  # always_include_last_n capped by max_events
    norm = contents_mod._normalize_limits(5, 10, 8)
    assert norm == (5, 5)  # always_include_last_n capped by max_events and total
    norm = contents_mod._normalize_limits(-1, -1, 10)
    assert norm == (None, 0)

def test_apply_max_events_with_always_include():
    contents = [f"c{i}" for i in range(10)]
    # Only max_events
    out = contents_mod._apply_max_events_with_always_include(contents, 3, None)
    assert out == contents[-3:]
    # Only always_include_last_n
    out = contents_mod._apply_max_events_with_always_include(contents, None, 2)
    assert out == contents[-2:]
    # Both, always_include_last_n > max_events
    out = contents_mod._apply_max_events_with_always_include(contents, 2, 5)
    assert out == contents[-2:]
    # Both, always_include_last_n < max_events
    out = contents_mod._apply_max_events_with_always_include(contents, 5, 2)
    assert out == contents[-5:]
    # None
    out = contents_mod._apply_max_events_with_always_include(contents, None, None)
    assert out == contents

def test_get_contents_to_summarize_and_always_include():
    contents = [f"c{i}" for i in range(5)]
    to_sum = contents_mod._get_contents_to_summarize(contents, 2)
    always = contents_mod._get_always_include_contents(contents, 2)
    assert to_sum == contents[:-2]
    assert always == contents[-2:]
    # Edge: always_include_last_n >= len(contents)
    to_sum = contents_mod._get_contents_to_summarize(contents, 10)
    always = contents_mod._get_always_include_contents(contents, 10)
    assert to_sum == []
    assert always == contents

def test_prepare_final_contents():
    contents = [f"c{i}" for i in range(4)]
    to_sum, always = contents_mod._prepare_final_contents(contents, 2)
    assert to_sum == contents[:-2]
    assert always == contents[-2:]

# --- Test for _get_state_context_content ---
def test_get_state_context_content_basic(sample_state):
    content = contents_mod._get_state_context_content(sample_state, ["user_profile", "current_task"])
    assert content is not None
    assert "user_profile" in content.parts[0].text
    assert "current_task" in content.parts[0].text

def test_get_state_context_content_missing_keys(sample_state):
    content = contents_mod._get_state_context_content(sample_state, ["not_found"])
    assert content is None

def test_get_state_context_content_custom_template(sample_state):
    tpl = "CTX: {context}"
    content = contents_mod._get_state_context_content(sample_state, ["user_profile"], tpl)
    assert content is not None
    assert content.parts[0].text.startswith("CTX:")

# --- Async test for _get_contents (main integration) ---
import asyncio
@pytest.mark.asyncio
async def test_get_contents_full_flow(sample_event, sample_state, sample_config):
    # Prepare events: 3 events, only 1 will be always included
    events = []
    for i in range(3):
        ev = copy.deepcopy(sample_event)
        ev.content.parts[0].text = f"msg{i}"
        events.append(ev)
    # Simulate config with state context and always_include_last_n=1
    out = await contents_mod._get_contents(
        config=sample_config,
        current_branch="main",
        events=events,
        agent_name="user",
        session_state=sample_state,
    )
    # Should include state context first, then last event
    assert out[0].parts[0].text.startswith("User:")
    assert out[1].parts[0].text == "msg2"
    assert len(out) == 2

@pytest.mark.asyncio
async def test_get_contents_no_state(sample_event, sample_config):
    # No state context, just always_include_last_n
    events = []
    for i in range(2):
        ev = copy.deepcopy(sample_event)
        ev.content.parts[0].text = f"msg{i}"
        events.append(ev)
    out = await contents_mod._get_contents(
        config=sample_config,
        current_branch="main",
        events=events,
        agent_name="user",
        session_state=None,
    )
    # Only the last event should be included
    assert out[0].parts[0].text == "msg1"
    assert len(out) == 1

@pytest.mark.asyncio
async def test_get_contents_empty_events(sample_config, sample_state):
    out = await contents_mod._get_contents(
        config=sample_config,
        current_branch="main",
        events=[],
        agent_name="user",
        session_state=sample_state,
    )
    # Only state context should be present
    assert len(out) == 1
    assert "user_profile" in out[0].parts[0].text

@pytest.mark.asyncio
async def test_exclude_authors_does_not_exclude_own_flow(event_with_author, sample_config):
    # Agent is 'agentA', branch is 'agentA.sub', invocation_id is 'abc'
    config = copy.deepcopy(sample_config)
    config.exclude_authors = ["agentA"]
    config.include_authors = None
    config.always_include_last_n = 2
    events = [
        event_with_author("agentA", "otherbranch", "abc"), # não será incluído (branch incompatível)
        event_with_author("other", "agentA", "abc"),       # contexto
        event_with_author("agentA", "agentA.sub", "abc"),  # own flow (agora último)
    ]
    out = await contents_mod._get_contents(
        config=config,
        current_branch="agentA.sub",
        events=events,
        agent_name="agentA",
        session_state=None,
        current_invocation_id="abc",
    )
    texts = [c.parts[0].text for c in out]
    assert any("hello" in t for t in texts)
    assert any("For context:" in t for t in texts)
    assert "hello" in texts and "For context:" in texts
    assert len(out) == 2

@pytest.mark.asyncio
async def test_exclude_authors_excludes_other(event_with_author, sample_config):
    config = copy.deepcopy(sample_config)
    config.exclude_authors = ["other"]
    config.include_authors = None
    config.always_include_last_n = 2
    events = [
        event_with_author("other", "agentA", "abc"),  # contexto
        event_with_author("agentA", "agentA.sub", "abc"),  # own flow (agora último)
    ]
    out = await contents_mod._get_contents(
        config=config,
        current_branch="agentA.sub",
        events=events,
        agent_name="agentA",
        session_state=None,
        current_invocation_id="abc",
    )
    texts = [c.parts[0].text for c in out]
    assert any("hello" in t for t in texts)
    assert all("For context:" not in t for t in texts)
    assert texts == ["hello"]
    assert len(out) == 1

@pytest.mark.asyncio
async def test_include_authors_includes_own_flow(event_with_author, sample_config):
    config = copy.deepcopy(sample_config)
    config.include_authors = ["other"]
    config.exclude_authors = None
    config.always_include_last_n = 2
    events = [
        event_with_author("agentA", "agentA.sub", "abc"),  # own flow, not in include_authors
        event_with_author("other", "agentA", "abc"),  # in include_authors, branch compatible
    ]
    out = await contents_mod._get_contents(
        config=config,
        current_branch="agentA.sub",
        events=events,
        agent_name="agentA",
        session_state=None,
        current_invocation_id="abc",
    )
    texts = [c.parts[0].text for c in out]
    assert any("hello" in t for t in texts)
    assert any("For context:" in t for t in texts)
    assert len(out) == 2

@pytest.mark.asyncio
async def test_include_authors_excludes_non_included(event_with_author, sample_config):
    config = copy.deepcopy(sample_config)
    config.include_authors = ["other"]
    config.exclude_authors = None
    config.always_include_last_n = 2
    events = [
        event_with_author("not_included", "agentA", "abc"),
        event_with_author("other", "agentA", "abc"),
    ]
    out = await contents_mod._get_contents(
        config=config,
        current_branch="agentA.sub",
        events=events,
        agent_name="agentA",
        session_state=None,
        current_invocation_id="abc",
    )
    texts = [c.parts[0].text for c in out]
    assert any("For context:" in t for t in texts)
    assert len(out) == 1

@pytest.mark.asyncio
async def test_exclude_and_include_none(event_with_author, sample_config):
    config = copy.deepcopy(sample_config)
    config.include_authors = None
    config.exclude_authors = None
    config.always_include_last_n = 2
    events = [
        event_with_author("agentA", "agentA.sub", "abc"),
        event_with_author("other", "agentA", "abc"),
    ]
    out = await contents_mod._get_contents(
        config=config,
        current_branch="agentA.sub",
        events=events,
        agent_name="agentA",
        session_state=None,
        current_invocation_id="abc",
    )
    texts = [c.parts[0].text for c in out]
    assert any("hello" in t for t in texts)
    assert any("For context:" in t for t in texts)
    assert len(out) == 2

# Additional tests for summarization, edge cases, etc. can be added as needed.

def make_event(author, branch, invocation_id):
    from google.adk.events.event import Event
    from google.genai import types
    return Event(author=author, content=types.Content(role="user", parts=[]), branch=branch, invocation_id=invocation_id, timestamp=1)

def make_config(include=None, exclude=None):
    from google.adk.agents.content_config import ContentConfig
    return ContentConfig(enabled=True, include_authors=include, exclude_authors=exclude)

def test_is_event_from_agent_flow_true():
    from google.adk.flows.llm_flows import contents as contents_mod
    event = make_event("agentA", "agentA.sub", "abc")
    assert contents_mod._is_event_from_agent_flow(event, "agentA", "agentA.sub", "abc")

def test_is_event_from_agent_flow_false_branch():
    from google.adk.flows.llm_flows import contents as contents_mod
    event = make_event("agentA", "otherbranch", "abc")
    assert not contents_mod._is_event_from_agent_flow(event, "agentA", "agentA.sub", "abc")

def test_is_event_from_agent_flow_false_invocation():
    from google.adk.flows.llm_flows import contents as contents_mod
    event = make_event("agentA", "agentA.sub", "def")
    assert not contents_mod._is_event_from_agent_flow(event, "agentA", "agentA.sub", "abc")

def test_should_exclude_event_excludes():
    from google.adk.flows.llm_flows import contents as contents_mod
    config = make_config(exclude=["other"])
    event = make_event("other", "branch", "abc")
    assert contents_mod._should_exclude_event(event, config, "agentA", "agentA.sub", "abc")

def test_should_exclude_event_not_exclude_own_flow():
    from google.adk.flows.llm_flows import contents as contents_mod
    config = make_config(exclude=["agentA"])
    event = make_event("agentA", "agentA.sub", "abc")
    assert not contents_mod._should_exclude_event(event, config, "agentA", "agentA.sub", "abc")

def test_should_include_event_includes():
    from google.adk.flows.llm_flows import contents as contents_mod
    config = make_config(include=["other"])
    event = make_event("other", "branch", "abc")
    assert contents_mod._should_include_event(event, config, "agentA", "agentA.sub", "abc")

def test_should_include_event_not_include():
    from google.adk.flows.llm_flows import contents as contents_mod
    config = make_config(include=["other"])
    event = make_event("not_included", "branch", "abc")
    assert not contents_mod._should_include_event(event, config, "agentA", "agentA.sub", "abc")

def test_should_include_event_always_include_own_flow():
    from google.adk.flows.llm_flows import contents as contents_mod
    config = make_config(include=["other"])
    event = make_event("agentA", "agentA.sub", "abc")
    assert contents_mod._should_include_event(event, config, "agentA", "agentA.sub", "abc")

@pytest.mark.asyncio
def test_get_contents_enabled_false(sample_event, sample_state, sample_config):
    # Should return only state context (if configured) or an empty list
    config = copy.deepcopy(sample_config)
    config.enabled = False
    events = [copy.deepcopy(sample_event) for _ in range(3)]
    out = asyncio.run(contents_mod._get_contents(
        config=config,
        current_branch="main",
        events=events,
        agent_name="user",
        session_state=sample_state,
    ))
    assert len(out) == 1
    assert "user_profile" in out[0].parts[0].text
    assert "current_task" in out[0].parts[0].text

@pytest.mark.asyncio
def test_get_contents_convert_foreign_events_false(event_with_author, sample_config):
    # Events from other agents should not be converted to context
    config = copy.deepcopy(sample_config)
    config.convert_foreign_events = False
    config.always_include_last_n = 2
    events = [
        event_with_author("other", "agentA", "abc"),
        event_with_author("agentA", "agentA.sub", "abc"),
    ]
    out = asyncio.run(contents_mod._get_contents(
        config=config,
        current_branch="agentA.sub",
        events=events,
        agent_name="agentA",
        session_state=None,
        current_invocation_id="abc",
    ))
    texts = [c.parts[0].text for c in out]
    assert any("hello" in t for t in texts)
    assert all("For context:" not in t for t in texts)
    assert len(out) == 2

@pytest.mark.asyncio
def test_get_contents_summarization_window(event_with_author, sample_config):
    # Only the most recent events (within the window) should be summarized
    from google.adk.agents.content_config import SummarizationConfig
    config = copy.deepcopy(sample_config)
    config.summarize = True
    config.summarization_config = SummarizationConfig(model="dummy-model")
    config.always_include_last_n = 2
    config.summarization_window = 3
    # Create 7 events: 2 old (should be discarded), 3 to summarize, 2 always_include
    events = [event_with_author("user", "main", "id") for _ in range(2)]  # old
    events += [event_with_author("user", "main", "id") for _ in range(3)] # to summarize
    events += [event_with_author("user", "main", "id") for _ in range(2)] # always_include
    # Mock summarizer to avoid real LLM call
    async def fake_summarize(*args, **kwargs):
        return "RESUMO"
    contents_mod._summarize_contents_with_llm = fake_summarize
    out = asyncio.run(contents_mod._get_contents(
        config=config,
        current_branch="main",
        events=events,
        agent_name="user",
        session_state=None,
    ))
    texts = [c.parts[0].text for c in out]
    assert any("RESUMO" in t for t in texts)
    assert len(out) == 3  # 1 summary + 2 always_include
