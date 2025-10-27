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

"""Unit tests for the News Podcast Agent."""

import pytest

import agent


def test_agent_initialization():
    """Test that the root agent is properly initialized."""
    assert agent.root_agent is not None
    assert agent.root_agent.name == "newsletter_podcast_producer"
    assert len(agent.root_agent.tools) > 0


def test_podcaster_agent_exists():
    """Test that the podcaster agent exists."""
    assert hasattr(agent, 'podcaster_agent')
    assert agent.podcaster_agent is not None
    assert agent.podcaster_agent.name == "podcaster_agent"


def test_newsletter_senders_config():
    """Test that newsletter senders are configured."""
    assert len(agent.NEWSLETTER_SENDERS) > 0
    assert 'tldr.tech' in agent.NEWSLETTER_SENDERS
    assert 'morningbrew.com' in agent.NEWSLETTER_SENDERS


def test_get_today_date():
    """Test the date utility function."""
    result = agent.get_today_date()
    assert result["status"] == "success"
    assert "primary_date" in result
    assert result["primary_date"] is not None


def test_is_valid_newsletter():
    """Test newsletter validation logic."""
    # Test valid TLDR newsletter
    sender = "dan@tldrnewsletter.com"
    subject = "TLDR AI - Daily Newsletter"
    content = "Here are today's top stories..."
    
    result = agent.is_valid_newsletter(sender, subject, content, [])
    assert result == True  # TLDR newsletters should always be valid
    
    # Test promotional email (should be invalid)
    sender_promo = "events@company.com"
    subject_promo = "Register for our webinar"
    content_promo = "Join us for a live webinar..."
    
    result_promo = agent.is_valid_newsletter(sender_promo, subject_promo, content_promo, [])
    assert result_promo == False  # Promotional content should be filtered out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

