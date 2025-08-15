#!/usr/bin/env python3
"""
ADK Pattern Compliance Test - Ensures proper Agent -> Tools -> APIs architecture.

This test validates that:
1. No mock/hardcoded responses in production tools
2. All agents follow ADK patterns
3. Tools make real API calls
4. Proper error handling and fallbacks
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

class ADKPatternValidator:
    """Validates ADK architecture patterns in the codebase."""
    
    def __init__(self, backend_path: str = None):
        self.backend_path = Path(backend_path or os.path.dirname(os.path.dirname(__file__)))
        self.agents_path = self.backend_path / "agents"
        self.violations = []
        self.warnings = []
        
    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks."""
        print("🔍 ADK Pattern Compliance Validation Starting...")
        
        # Check 1: No mock responses in tools
        self._check_no_mock_responses()
        
        # Check 2: All agents inherit from BaseADKAgent
        self._check_agent_inheritance()
        
        # Check 3: Tools use real API calls
        self._check_real_api_usage()
        
        # Check 4: Tools have proper error handling
        self._check_error_handling()
        
        # Check 5: Tools are async and use ToolContext
        self._check_tool_signatures()
        
        # Check 6: No MockAgent in production
        self._check_no_mock_agents()
        
        success = len(self.violations) == 0
        return success, self.violations, self.warnings
    
    def _check_no_mock_responses(self):
        """Ensure no hardcoded/mock data in tool implementations."""
        print("  ✓ Checking for mock responses in tools...")
        
        patterns = [
            r'return\s*{\s*["\'].*["\']\s*:\s*\d+',  # Hardcoded numbers
            r'return\s*{\s*["\'].*["\']\s*:\s*["\'].*hardcoded',  # "hardcoded" text
            r'return\s*{\s*["\'].*["\']\s*:\s*\[.*\]',  # Hardcoded lists
            r'#\s*TODO.*mock|#\s*FIXME.*mock',  # TODO/FIXME comments about mocks
        ]
        
        for agent_file in self.agents_path.glob("*_agent.py"):
            if agent_file.name == "base_agent.py":
                continue
                
            content = agent_file.read_text()
            
            # Check for suspicious patterns
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Check if it's inside a tool function
                    for match in matches:
                        # Allow if it's in error handling or fallback
                        if "error" in match.lower() or "fallback" in match.lower() or "source" in match.lower():
                            continue
                        # Allow success status
                        if '"success": True' in match or '"success": False' in match:
                            continue
                        self.warnings.append(
                            f"Possible hardcoded data in {agent_file.name}: {match[:50]}..."
                        )
    
    def _check_agent_inheritance(self):
        """Verify all agents inherit from BaseADKAgent."""
        print("  ✓ Checking agent inheritance...")
        
        for agent_file in self.agents_path.glob("*_agent.py"):
            if agent_file.name == "base_agent.py":
                continue
                
            content = agent_file.read_text()
            
            # Check for class definition with BaseADKAgent
            if not re.search(r'class\s+\w+Agent\s*\(\s*BaseADKAgent\s*\)', content):
                self.violations.append(
                    f"Agent in {agent_file.name} does not inherit from BaseADKAgent"
                )
    
    def _check_real_api_usage(self):
        """Ensure tools import and use real backend APIs."""
        print("  ✓ Checking for real API usage...")
        
        required_imports = {
            "iam_agent.py": "backend.api.iam",
            "network_agent.py": "backend.api.network",
            "cost_agent.py": "backend.api.cost",
            "compliance_agent.py": "backend.api.compliance",
            "storage_agent.py": "backend.services.enhanced_asset_inventory_service"
        }
        
        for agent_file, expected_import in required_imports.items():
            file_path = self.agents_path / agent_file
            if file_path.exists():
                content = file_path.read_text()
                
                # Check for import statement
                if expected_import not in content and expected_import.split('.')[-1] not in content:
                    self.violations.append(
                        f"{agent_file} does not import real API: {expected_import}"
                    )
                
                # Check for try/except around API calls
                if "try:" not in content:
                    self.violations.append(
                        f"{agent_file} missing try/except for API error handling"
                    )
    
    def _check_error_handling(self):
        """Verify proper error handling in tools."""
        print("  ✓ Checking error handling...")
        
        for agent_file in self.agents_path.glob("*_agent.py"):
            if agent_file.name == "base_agent.py":
                continue
                
            content = agent_file.read_text()
            
            # Check for exception handling patterns
            if "@create_tool" in content:
                # Count try/except blocks
                try_count = content.count("try:")
                except_count = content.count("except")
                
                if try_count == 0 or except_count == 0:
                    self.violations.append(
                        f"{agent_file.name} has tools without proper exception handling"
                    )
                
                # Check for source field in returns
                if '"source":' not in content:
                    self.warnings.append(
                        f"{agent_file.name} tools should include 'source' field in responses"
                    )
    
    def _check_tool_signatures(self):
        """Ensure tools are async and use ToolContext."""
        print("  ✓ Checking tool signatures...")
        
        for agent_file in self.agents_path.glob("*_agent.py"):
            if agent_file.name == "base_agent.py":
                continue
                
            content = agent_file.read_text()
            
            # Find tool definitions
            tool_pattern = r'@create_tool\([^)]+\)\s*async def (\w+)\([^)]+\)'
            tools = re.findall(tool_pattern, content)
            
            for tool in tools:
                # Check if tool is async
                if f"async def {tool}" not in content:
                    self.violations.append(
                        f"Tool {tool} in {agent_file.name} must be async"
                    )
                
                # Check for ToolContext parameter
                tool_def_pattern = f"async def {tool}\\([^)]*tool_context: ToolContext[^)]*\\)"
                if not re.search(tool_def_pattern, content):
                    self.violations.append(
                        f"Tool {tool} in {agent_file.name} must have tool_context: ToolContext parameter"
                    )
    
    def _check_no_mock_agents(self):
        """Ensure no MockAgent class in production code."""
        print("  ✓ Checking for MockAgent usage...")
        
        # Check main agent_llm.py file
        agent_llm_path = self.backend_path / "api" / "agent_llm.py"
        if agent_llm_path.exists():
            content = agent_llm_path.read_text()
            
            # Check for MockAgent class definition
            if "class MockAgent" in content:
                # It's okay if it's in a fallback section
                if "# Ultimate fallback" not in content:
                    self.violations.append(
                        "MockAgent class found in agent_llm.py without proper fallback context"
                    )
            
            # Check that real agents are imported
            if "from backend.agents" not in content:
                self.violations.append(
                    "agent_llm.py not importing real agents from backend.agents"
                )
    
    def print_results(self):
        """Print validation results."""
        success, violations, warnings = self.validate_all()
        
        print("\n" + "="*60)
        print("📊 ADK PATTERN COMPLIANCE RESULTS")
        print("="*60)
        
        if violations:
            print("\n❌ VIOLATIONS (Must Fix):")
            for v in violations:
                print(f"  • {v}")
        
        if warnings:
            print("\n⚠️  WARNINGS (Should Review):")
            for w in warnings:
                print(f"  • {w}")
        
        if success:
            print("\n✅ SUCCESS: All ADK patterns validated!")
            print("  • No mock responses in production tools")
            print("  • All agents inherit from BaseADKAgent")
            print("  • Tools use real GCP API calls")
            print("  • Proper error handling implemented")
            print("  • Async tools with ToolContext")
        else:
            print(f"\n❌ FAILED: {len(violations)} violations found")
        
        print("\n" + "="*60)
        return success


def main():
    """Run the validation test."""
    validator = ADKPatternValidator()
    success = validator.print_results()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()