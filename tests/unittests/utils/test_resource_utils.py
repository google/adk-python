"""Tests for resource utility functions - Issue #2940 fix"""

import sys
import os

# Add src directory to Python path for imports
test_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(test_dir, '..', '..', '..')
src_dir = os.path.join(root_dir, 'src')
sys.path.insert(0, os.path.abspath(src_dir))

import pytest
from src.google.adk.utils.resource_utils import (
    extract_agent_engine_id, 
    validate_agent_engine_resource_name,
    get_project_from_resource_name,
    get_location_from_resource_name
)

class TestResourceUtils:
    """Test cases for resource utility functions."""
    
    def test_extract_agent_engine_id_valid(self):
        """Test extracting agent ID from valid resource name."""
        resource_name = "projects/test-project/locations/us-central1/reasoningEngines/123456789"
        result = extract_agent_engine_id(resource_name)
        assert result == "123456789"
    
    def test_extract_agent_engine_id_alphanumeric(self):
        """Test extracting alphanumeric agent ID."""
        resource_name = "projects/my-proj/locations/us-west1/reasoningEngines/abc123def456"
        result = extract_agent_engine_id(resource_name)
        assert result == "abc123def456"
    
    def test_extract_agent_engine_id_empty_string(self):
        """Test handling empty resource name."""
        with pytest.raises(ValueError, match="Agent Engine resource name cannot be empty"):
            extract_agent_engine_id("")
    
    def test_extract_agent_engine_id_none(self):
        """Test handling None input."""
        with pytest.raises(ValueError, match="Agent Engine resource name cannot be empty"):
            extract_agent_engine_id(None)
    
    def test_validate_resource_name_valid(self):
        """Test validating correct resource name format."""
        valid_names = [
            "projects/test/locations/us-central1/reasoningEngines/123abc",
            "projects/my-project-123/locations/europe-west1/reasoningEngines/abc123def",
        ]
        
        for name in valid_names:
            assert validate_agent_engine_resource_name(name) == True
    
    def test_validate_resource_name_invalid(self):
        """Test validating incorrect resource name formats."""
        invalid_names = [
            "invalid-format",
            "projects/test/reasoningEngines/123",  # Missing locations
            "",
            None
        ]
        
        for name in invalid_names:
            assert validate_agent_engine_resource_name(name) == False
    
    def test_real_world_scenario(self):
        """Test complete real-world scenario matching issue #2940."""
        # This simulates what users get from agent_engine.api_resource.name
        full_resource_name = "projects/my-gcp-project/locations/us-central1/reasoningEngines/1234567890abcdef"
        
        # Extract agent ID (the fix for issue #2940)
        agent_id = extract_agent_engine_id(full_resource_name)
        
        # Verify it's just the ID, not the full path
        assert agent_id == "1234567890abcdef"
        assert "projects/" not in agent_id
        assert "locations/" not in agent_id
        assert "reasoningEngines/" not in agent_id
