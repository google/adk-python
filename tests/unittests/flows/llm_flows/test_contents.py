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
from unittest.mock import AsyncMock
from google.adk.agents.content_config import ContentConfig, SummarizationConfig
from google.adk.flows.llm_flows import contents as contents_mod
from google.genai import types
from google.adk.events.event import Event

# Project utilities for creating events and agents
from ... import utils

class DummyAgent:
    def __init__(self, name="agent", model="gemini-2.0-flash"):
        self.name = name
        self.model = model
        self.canonical_model = utils.MockModel.create(responses=["summary"])

@pytest.mark.asyncio
async def test_get_contents_basic_filtering():
    # Create events from different authors and branches
    events = [
        Event(author="user", content=types.Content(role="user", parts=[types.Part(text="msg1")])),
        Event(author="agent", content=types.Content(role="user", parts=[types.Part(text="msg2")])),
        Event(author="system", content=types.Content(role="user", parts=[types.Part(text="msg3")])),
    ]
    config = ContentConfig(include_authors=["user", "agent"])
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name="agent", session_state=None, agent=DummyAgent())
    # Should filter only user and agent
    assert len(result) == 2
    assert all(c.parts[0].text in ["msg1", "msg2"] for c in result)

@pytest.mark.asyncio
async def test_get_contents_max_events():
    events = [
        Event(author="user", content=types.Content(role="user", parts=[types.Part(text=f"msg{i}")])) for i in range(5)
    ]
    config = ContentConfig(max_events=3)
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name="agent", session_state=None, agent=DummyAgent())
    # Should only get the last 3
    assert len(result) == 3
    assert all(c.parts[0].text in ["msg2", "msg3", "msg4"] for c in result)

@pytest.mark.asyncio
async def test_get_contents_summarization(monkeypatch):
    # Create 4 events, with always_include_last_n=2, summarize=True
    events = [
        Event(author="user", content=types.Content(role="user", parts=[types.Part(text=f"msg{i}")])) for i in range(4)
    ]
    config = ContentConfig(summarize=True, always_include_last_n=2, summarization_config=SummarizationConfig(model="gemini-2.0-flash"))
    # Mock summarize_events_with_llm to return a fixed text
    monkeypatch.setattr(contents_mod, "summarize_events_with_llm", AsyncMock(return_value="[SUMMARY]"))
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name="agent", session_state=None, agent=DummyAgent())
    # Should contain the summary and the last 2 events
    assert any("[SUMMARY]" in (p.text if hasattr(p, "text") else str(p)) for c in result for p in c.parts)
    assert any("msg2" in (p.text if hasattr(p, "text") else str(p)) for c in result for p in c.parts)
    assert any("msg3" in (p.text if hasattr(p, "text") else str(p)) for c in result for p in c.parts)

@pytest.mark.asyncio
async def test_get_contents_context_injection():
    events = [Event(author="user", content=types.Content(role="user", parts=[types.Part(text="msg")]))]
    config = ContentConfig(context_from_state=["foo", "bar"], state_template="CTX: {context}")
    session_state = {"foo": "abc", "bar": "xyz"}
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name="agent", session_state=session_state, agent=DummyAgent())
    # Should inject context as the first event
    assert result[0].parts[0].text.startswith("CTX:")
    assert "abc" in result[0].parts[0].text and "xyz" in result[0].parts[0].text

@pytest.mark.asyncio
async def test_get_contents_empty_events():
    config = ContentConfig()
    result = await contents_mod._get_contents(config, current_branch=None, events=[], agent_name="agent", session_state=None, agent=DummyAgent())
    assert result == []

# Tests for helper functions

def test_is_other_agent_reply():
    event = Event(author="other_agent", content=types.Content(role="user", parts=[types.Part(text="hi")]))
    assert contents_mod._is_other_agent_reply("agent", event)
    event2 = Event(author="agent", content=types.Content(role="user", parts=[types.Part(text="hi")]))
    assert not contents_mod._is_other_agent_reply("agent", event2)
    event3 = Event(author="user", content=types.Content(role="user", parts=[types.Part(text="hi")]))
    assert not contents_mod._is_other_agent_reply("agent", event3)

def test_convert_foreign_event():
    event = Event(author="other_agent", content=types.Content(role="user", parts=[types.Part(text="hi")]))
    converted = contents_mod._convert_foreign_event(event)
    assert converted.author == "user"
    assert any("other_agent" in p.text for p in converted.content.parts if p.text)

# _merge_function_response_events requires events with function_response, can be tested separately if needed. 

# More detailed tests for windowing and summarization
@pytest.mark.asyncio
async def test_get_contents_summarization_window_only(monkeypatch):
    """Tests that summarization_window limits events for summarization when always_include_last_n is 0."""
    events = [Event(author="user", content=utils.UserContent(f"msg{i}")) for i in range(10)] # 10 events
    # Summarize, window of 5, keep 0 last messages (so all 5 in window are for summary)
    config = ContentConfig(summarize=True, summarization_window=5, always_include_last_n=0, summarization_config=SummarizationConfig(model="gemini-2.0-flash"))
    
    mock_summarize = AsyncMock(return_value="[WINDOW_SUMMARY]")
    monkeypatch.setattr(contents_mod, "summarize_events_with_llm", mock_summarize)
    
    dummy_agent = DummyAgent()
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    # summarize_events_with_llm should have been called with the 5 events from the window (msg5 to msg9)
    assert mock_summarize.call_count == 1
    summarized_events_arg = mock_summarize.call_args[1]['events']
    assert len(summarized_events_arg) == 5 
    assert summarized_events_arg[0].content.parts[0].text == "msg5"
    assert summarized_events_arg[-1].content.parts[0].text == "msg9"
    
    assert len(result) == 1
    assert result[0].parts[0].text == "[WINDOW_SUMMARY]"

@pytest.mark.asyncio
async def test_get_contents_always_include_last_n_only_no_summarization_window(monkeypatch):
    """Tests always_include_last_n when summarization_window is not set."""
    events = [Event(author="user", content=utils.UserContent(f"msg{i}")) for i in range(5)] # 5 events
    # Summarize, keep last 2, no specific window (so first 3 should be summarized)
    config = ContentConfig(summarize=True, always_include_last_n=2, summarization_config=SummarizationConfig(model="gemini-2.0-flash"))

    mock_summarize = AsyncMock(return_value="[LAST_N_SUMMARY]")
    monkeypatch.setattr(contents_mod, "summarize_events_with_llm", mock_summarize)

    dummy_agent = DummyAgent()
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)

    assert mock_summarize.call_count == 1
    summarized_events_arg = mock_summarize.call_args[1]['events']
    assert len(summarized_events_arg) == 3 # msg0, msg1, msg2
    assert summarized_events_arg[0].content.parts[0].text == "msg0"
    assert summarized_events_arg[-1].content.parts[0].text == "msg2"

    assert len(result) == 3 # Summary + msg3 + msg4
    assert result[0].parts[0].text == "[LAST_N_SUMMARY]"
    assert result[1].parts[0].text == "msg3"
    assert result[2].parts[0].text == "msg4"

@pytest.mark.asyncio
async def test_get_contents_summarization_window_smaller_than_always_include(monkeypatch):
    """Tests behavior when always_include_last_n is larger than what summarization_window might process alone."""
    events = [Event(author="user", content=utils.UserContent(f"msg{i}")) for i in range(10)] # 10 events
    # always_include_last_n=5 (Y=5). summarization_window=3 (N=3)
    # Expected: last 5 events (msg5-msg9) are kept.
    # Remaining events (msg0-msg4) are processed. Window N=3 applies to these -> (msg2,msg3,msg4) are summarized.
    # Final: [summary(msg2,msg3,msg4), msg5, msg6, msg7, msg8, msg9]
    config = ContentConfig(summarize=True, summarization_window=3, always_include_last_n=5, summarization_config=SummarizationConfig(model="gemini-2.0-flash"))

    mock_summarize = AsyncMock(return_value="[SUMMARY_OF_WINDOW_BEFORE_LAST_N]")
    monkeypatch.setattr(contents_mod, "summarize_events_with_llm", mock_summarize)

    dummy_agent = DummyAgent()
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    assert mock_summarize.call_count == 1
    summarized_events_arg = mock_summarize.call_args[1]['events']
    assert len(summarized_events_arg) == 3 # msg2, msg3, msg4
    assert summarized_events_arg[0].content.parts[0].text == "msg2"
    assert summarized_events_arg[1].content.parts[0].text == "msg3"
    assert summarized_events_arg[2].content.parts[0].text == "msg4"
    
    assert len(result) == 1 + 5 # 1 summary event + 5 always_include_last_n events
    assert result[0].parts[0].text == "[SUMMARY_OF_WINDOW_BEFORE_LAST_N]"
    for i in range(5):
        assert result[i+1].parts[0].text == f"msg{i+5}" # msg5 to msg9

@pytest.mark.asyncio
async def test_get_contents_summarization_not_enough_events_for_window_or_last_n(monkeypatch):
    """Tests when total events are less than window or always_include_last_n."""
    events = [Event(author="user", content=utils.UserContent(f"msg{i}")) for i in range(3)] # Only 3 events
    config = ContentConfig(summarize=True, summarization_window=5, always_include_last_n=5, summarization_config=SummarizationConfig(model="gemini-2.0-flash"))

    mock_summarize = AsyncMock(return_value="[FEW_EVENTS_SUMMARY]")
    monkeypatch.setattr(contents_mod, "summarize_events_with_llm", mock_summarize)

    dummy_agent = DummyAgent()
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    # Summarization should not be called as there are not enough events to be outside always_include_last_n within the window
    assert mock_summarize.call_count == 0
    
    assert len(result) == 3 # All original events should be present
    for i in range(3):
        assert result[i].parts[0].text == f"msg{i}"

# More detailed tests for filtering
@pytest.mark.asyncio
async def test_get_contents_exclude_authors():
    events = [
        Event(author="user", content=utils.UserContent("user_msg")), # Kept
        Event(author="agent1_excluded", content=utils.UserContent("agent1_excluded_msg")), # Excluded
        Event(author="system_excluded", content=utils.UserContent("system_excluded_msg")), # Excluded
        Event(author="agent2_passes", content=utils.UserContent("agent2_passes_msg")), # Kept, will be converted by default
    ]
    # Default: convert_foreign_events=True
    config = ContentConfig(exclude_authors=["system_excluded", "agent1_excluded"])
    dummy_agent = DummyAgent(name="current_agent") # DummyAgent different from event authors that pass
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    assert len(result) == 2
    # Check user event (unchanged)
    assert result[0].parts[0].text == "user_msg"
    
    # Check converted event (agent2_passes)
    assert len(result[1].parts) >= 2 # Should have at least "For context:" and the message part
    assert result[1].parts[0].text == "For context:" # Check the first part
    assert result[1].parts[1].text == "[agent2_passes] said: agent2_passes_msg" # Check the second part

@pytest.mark.asyncio
async def test_get_contents_exclude_authors_conversion_false():
    events = [
        Event(author="user", content=utils.UserContent("user_msg")), 
        Event(author="agent1_excluded", content=utils.UserContent("agent1_excluded_msg")), 
        Event(author="system_excluded", content=utils.UserContent("system_excluded_msg")),
        Event(author="agent2_passes", content=utils.UserContent("agent2_passes_msg")), 
    ]
    config = ContentConfig(exclude_authors=["system_excluded", "agent1_excluded"], convert_foreign_events=False)
    dummy_agent = DummyAgent(name="current_agent")
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    assert len(result) == 2
    assert result[0].parts[0].text == "user_msg"
    assert result[1].parts[0].text == "agent2_passes_msg" # Not converted

@pytest.mark.asyncio
async def test_get_contents_include_and_exclude_authors_interaction():
    events = [
        Event(author="user", content=utils.UserContent("user_msg")), 
        Event(author="included_but_foreign", content=utils.UserContent("included_but_foreign_msg")), 
        Event(author="excluded_agent", content=utils.UserContent("excluded_agent_msg")), 
        Event(author="both_excluded", content=utils.UserContent("both_excluded_msg")), 
        Event(author="other_user_not_explicitly_included", content=utils.UserContent("other_user_not_explicitly_included_msg")), 
    ]
    # Default: convert_foreign_events=True
    # Include user, included_but_foreign. Exclude excluded_agent, both_excluded.
    config = ContentConfig(include_authors=["user", "included_but_foreign"], exclude_authors=["excluded_agent", "both_excluded"])
    dummy_agent = DummyAgent(name="current_agent")
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    assert len(result) == 2
    
    # Find the user message and the converted message by checking the FIRST part
    user_msg_content = None
    converted_content = None
    for content_obj in result:
        if content_obj.parts and content_obj.parts[0].text == "user_msg":
            user_msg_content = content_obj
        elif content_obj.parts and content_obj.parts[0].text == "For context:":
            converted_content = content_obj
            
    assert user_msg_content is not None, "User message content not found in result"
    assert converted_content is not None, "Converted foreign message content not found in result"
    
    # Verify the converted content structure
    assert len(converted_content.parts) >= 2
    assert converted_content.parts[0].text == "For context:"
    assert converted_content.parts[1].text == "[included_but_foreign] said: included_but_foreign_msg"
    
    # Ensure others are not present
    raw_texts_from_events = [e.content.parts[0].text for e in events]
    # Check that the original text of excluded messages is not present in ANY part of the results
    all_result_texts = [p.text for res in result for p in res.parts]
    assert "excluded_agent_msg" not in all_result_texts and "excluded_agent_msg" in raw_texts_from_events
    assert "both_excluded_msg" not in all_result_texts and "both_excluded_msg" in raw_texts_from_events
    assert "other_user_not_explicitly_included_msg" not in all_result_texts and "other_user_not_explicitly_included_msg" in raw_texts_from_events

@pytest.mark.asyncio
async def test_get_contents_include_and_exclude_authors_interaction_conversion_false():
    events = [
        Event(author="user", content=utils.UserContent("user_msg")), 
        Event(author="included_author", content=utils.UserContent("included_author_msg")), 
        Event(author="excluded_agent", content=utils.UserContent("excluded_agent_msg")),
        Event(author="both_excluded", content=utils.UserContent("both_excluded_msg")), 
        Event(author="other_user_not_explicitly_included", content=utils.UserContent("other_user_not_explicitly_included_msg")), 
    ]
    config = ContentConfig(include_authors=["user", "included_author"], 
                           exclude_authors=["excluded_agent", "both_excluded"], 
                           convert_foreign_events=False)
    dummy_agent = DummyAgent(name="current_agent")
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    assert len(result) == 2
    texts = [r.parts[0].text for r in result]
    assert "user_msg" in texts
    assert "included_author_msg" in texts # Not converted

    raw_texts_from_events = [e.content.parts[0].text for e in events]
    assert "excluded_agent_msg" not in texts and "excluded_agent_msg" in raw_texts_from_events
    assert "both_excluded_msg" not in texts and "both_excluded_msg" in raw_texts_from_events
    assert "other_user_not_explicitly_included_msg" not in texts and "other_user_not_explicitly_included_msg" in raw_texts_from_events


# New tests for convert_foreign_events flag
@pytest.mark.asyncio
async def test_get_contents_convert_foreign_events_true_by_default():
    events = [Event(author="foreign_agent", content=utils.UserContent("foreign_msg"))]
    config = ContentConfig() # convert_foreign_events is True by default
    dummy_agent = DummyAgent(name="current_agent")
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    assert len(result) == 1
    # Verify the converted content structure
    assert len(result[0].parts) >= 2
    assert result[0].parts[0].text == "For context:"
    assert result[0].parts[1].text == "[foreign_agent] said: foreign_msg"

@pytest.mark.asyncio
async def test_get_contents_convert_foreign_events_false():
    events = [Event(author="foreign_agent", content=utils.UserContent("foreign_msg"))]
    config = ContentConfig(convert_foreign_events=False)
    dummy_agent = DummyAgent(name="current_agent")
    result = await contents_mod._get_contents(config, current_branch=None, events=events, agent_name=dummy_agent.name, session_state=None, agent=dummy_agent)
    
    assert len(result) == 1
    assert result[0].parts[0].text == "foreign_msg" # Not converted 