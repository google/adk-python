#!/usr/bin/env python3
"""
Playwright MCP Test Runner
==========================

This script integrates with Claude's Playwright MCP tools to run browser tests.
It can be invoked through a slash command for automated UI testing.

Usage:
    /test-ui                    # Run all UI tests
    /test-ui service-evaluation # Run specific test
    /test-ui --screenshot       # Take screenshots during tests
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlaywrightMCPRunner:
    """Runner that coordinates with Claude's Playwright MCP tools."""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.results = []
        self.screenshots_dir = Path("tests/screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        
    def navigate_to_app(self) -> Dict[str, Any]:
        """Navigate to the Streamlit application."""
        return {
            "action": "navigate",
            "url": self.base_url,
            "expected": "Page loaded successfully"
        }
        
    def take_screenshot(self, name: str) -> Dict[str, Any]:
        """Take a screenshot of current state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.png"
        return {
            "action": "screenshot",
            "filename": str(self.screenshots_dir / filename),
            "fullPage": True
        }
        
    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture accessibility snapshot for analysis."""
        return {
            "action": "snapshot",
            "purpose": "Analyze page structure and content"
        }
        
    def test_executive_dashboard(self) -> List[Dict[str, Any]]:
        """Test steps for executive dashboard."""
        return [
            {
                "name": "Navigate to Dashboard",
                "action": "navigate",
                "url": self.base_url
            },
            {
                "name": "Wait for Dashboard Load",
                "action": "wait",
                "text": "Executive Security Dashboard",
                "timeout": 5
            },
            {
                "name": "Capture Dashboard Snapshot",
                "action": "snapshot"
            },
            {
                "name": "Check Critical Findings Metric",
                "action": "evaluate",
                "function": "() => document.querySelector('[data-testid=\"metric-container\"]') !== null",
                "expected": True
            },
            {
                "name": "Screenshot Dashboard",
                "action": "screenshot",
                "filename": "executive_dashboard.png"
            }
        ]
        
    def test_service_evaluation(self) -> List[Dict[str, Any]]:
        """Test steps for service evaluation feature."""
        return [
            {
                "name": "Navigate to App",
                "action": "navigate",
                "url": self.base_url
            },
            {
                "name": "Wait for Page Load",
                "action": "wait",
                "time": 2
            },
            {
                "name": "Click Service Evaluation Tab",
                "action": "click",
                "element": "Service Evaluation tab",
                "ref": "tab containing 'Service Evaluation' text"
            },
            {
                "name": "Wait for Tab Content",
                "action": "wait",
                "text": "Service Evaluation",
                "timeout": 3
            },
            {
                "name": "Select Example Service",
                "action": "select_option",
                "element": "Service dropdown",
                "ref": "select box for example services",
                "values": ["vertex-ai-memory-store"]
            },
            {
                "name": "Click Evaluate Button",
                "action": "click",
                "element": "Evaluate Service button",
                "ref": "button with text 'Evaluate Service'"
            },
            {
                "name": "Wait for Results",
                "action": "wait",
                "text": "Evaluation Complete",
                "timeout": 10
            },
            {
                "name": "Verify Risk Score Display",
                "action": "evaluate",
                "function": "() => document.body.textContent.includes('Risk Score')",
                "expected": True
            },
            {
                "name": "Screenshot Results",
                "action": "screenshot",
                "filename": "service_evaluation_results.png"
            }
        ]
        
    def test_security_chat(self) -> List[Dict[str, Any]]:
        """Test steps for security chat interface."""
        return [
            {
                "name": "Navigate to App",
                "action": "navigate",
                "url": self.base_url
            },
            {
                "name": "Click Security Chat Tab",
                "action": "click",
                "element": "Security Chat tab",
                "ref": "tab containing 'Security Chat'"
            },
            {
                "name": "Wait for Chat Interface",
                "action": "wait",
                "text": "Security Agent Chat",
                "timeout": 3
            },
            {
                "name": "Type Test Query",
                "action": "type",
                "element": "Chat input field",
                "ref": "text input for chat messages",
                "text": "What tables are available in the security database?"
            },
            {
                "name": "Submit Query",
                "action": "press_key",
                "key": "Enter"
            },
            {
                "name": "Wait for Response",
                "action": "wait",
                "time": 5
            },
            {
                "name": "Verify Response Displayed",
                "action": "evaluate",
                "function": "() => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length > 1",
                "expected": True
            },
            {
                "name": "Screenshot Chat",
                "action": "screenshot",
                "filename": "security_chat.png"
            }
        ]
        
    def test_msa_analyzer(self) -> List[Dict[str, Any]]:
        """Test steps for MSA analyzer."""
        return [
            {
                "name": "Navigate to App",
                "action": "navigate",
                "url": self.base_url
            },
            {
                "name": "Click MSA Analyzer Tab",
                "action": "click",
                "element": "MSA Analyzer tab",
                "ref": "tab containing 'MSA Analyzer'"
            },
            {
                "name": "Wait for MSA Interface",
                "action": "wait",
                "text": "Master Service Agreement",
                "timeout": 3
            },
            {
                "name": "Verify Upload Interface",
                "action": "evaluate",
                "function": "() => document.body.textContent.includes('Upload')",
                "expected": True
            },
            {
                "name": "Screenshot MSA Interface",
                "action": "screenshot",
                "filename": "msa_analyzer.png"
            }
        ]
        
    def test_responsive_design(self) -> List[Dict[str, Any]]:
        """Test steps for responsive design."""
        viewports = [
            {"width": 1920, "height": 1080, "name": "desktop"},
            {"width": 768, "height": 1024, "name": "tablet"},
            {"width": 375, "height": 667, "name": "mobile"}
        ]
        
        steps = []
        for viewport in viewports:
            steps.extend([
                {
                    "name": f"Resize to {viewport['name']}",
                    "action": "resize",
                    "width": viewport["width"],
                    "height": viewport["height"]
                },
                {
                    "name": f"Navigate at {viewport['name']} size",
                    "action": "navigate",
                    "url": self.base_url
                },
                {
                    "name": f"Wait for render at {viewport['name']}",
                    "action": "wait",
                    "time": 2
                },
                {
                    "name": f"Screenshot {viewport['name']} view",
                    "action": "screenshot",
                    "filename": f"responsive_{viewport['name']}.png"
                }
            ])
            
        return steps
        
    def generate_test_plan(self, test_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a test plan for Claude to execute."""
        
        test_suites = {
            "dashboard": self.test_executive_dashboard(),
            "service-evaluation": self.test_service_evaluation(),
            "security-chat": self.test_security_chat(),
            "msa-analyzer": self.test_msa_analyzer(),
            "responsive": self.test_responsive_design()
        }
        
        if test_name and test_name in test_suites:
            # Run specific test
            return {
                "name": test_name,
                "steps": test_suites[test_name],
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Run all tests
            all_steps = []
            for suite_name, steps in test_suites.items():
                all_steps.append({
                    "suite": suite_name,
                    "steps": steps
                })
                
            return {
                "name": "full-suite",
                "suites": all_steps,
                "timestamp": datetime.now().isoformat()
            }
            
    def format_for_claude(self, test_plan: Dict[str, Any]) -> str:
        """Format test plan as instructions for Claude."""
        instructions = [
            "# Playwright UI Test Execution Plan",
            f"Generated: {test_plan['timestamp']}",
            "",
            "Please execute the following browser automation tests using Playwright MCP tools:",
            ""
        ]
        
        if "suites" in test_plan:
            # Multiple test suites
            for suite in test_plan["suites"]:
                instructions.append(f"## Test Suite: {suite['suite']}")
                instructions.append("")
                for i, step in enumerate(suite["steps"], 1):
                    instructions.append(f"{i}. **{step['name']}**")
                    instructions.append(f"   - Action: `{step['action']}`")
                    for key, value in step.items():
                        if key not in ["name", "action"]:
                            instructions.append(f"   - {key}: {value}")
                    instructions.append("")
        else:
            # Single test suite
            instructions.append(f"## Test: {test_plan['name']}")
            instructions.append("")
            for i, step in enumerate(test_plan["steps"], 1):
                instructions.append(f"{i}. **{step['name']}**")
                instructions.append(f"   - Action: `{step['action']}`")
                for key, value in step.items():
                    if key not in ["name", "action"]:
                        instructions.append(f"   - {key}: {value}")
                instructions.append("")
                
        instructions.extend([
            "",
            "## Expected Outcomes:",
            "- All navigation actions should load successfully",
            "- All element interactions should complete without errors",
            "- Screenshots should be captured at specified points",
            "- All verification steps should pass",
            "",
            "Please report the results of each step and provide a summary at the end."
        ])
        
        return "\n".join(instructions)


def main():
    """Main entry point for generating test plans."""
    import sys
    
    runner = PlaywrightMCPRunner()
    
    # Parse command line arguments
    test_name = None
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
    # Generate test plan
    test_plan = runner.generate_test_plan(test_name)
    
    # Format for Claude
    instructions = runner.format_for_claude(test_plan)
    
    # Save to file
    output_file = Path("tests/test_plan.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(instructions)
        
    print(instructions)
    print(f"\n✅ Test plan saved to: {output_file}")
    

if __name__ == "__main__":
    main()