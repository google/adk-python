#!/usr/bin/env python3
"""
Direct test of MSA extraction logic
"""

import sys
import os
sys.path.insert(0, 'backend')
sys.path.insert(0, 'backend/api')

# Mock Vertex AI to force pattern-based extraction
import unittest.mock as mock
with mock.patch('backend.api.msa_analyzer.VERTEX_AI_AVAILABLE', False):
    from backend.api.msa_analyzer import extract_structured_changes

BIGQUERY_ACL_MSA = """We're writing to remind you that starting March 17, 2026, we'll be introducing more granular permission requirements for managing BigQuery dataset Access Control Lists (ACLs)."""

print("🧪 Testing Direct Extraction")
print("=" * 50)

# Test extraction
changes = extract_structured_changes(BIGQUERY_ACL_MSA)

print(f"✅ Extracted {len(changes)} changes\n")

for i, change in enumerate(changes, 1):
    print(f"{i}. {change.service} - {change.change_type}")
    print(f"   Impact: {change.impact_level}")
    print(f"   Date: {change.effective_date}")
    print(f"   Description: {change.description[:100]}...")
    print(f"   Action: {change.required_action[:100]}...")
    print(f"   Resources: {', '.join(change.affected_resources)}")
    print()

print("=" * 50)
print("✅ Extraction test complete!")