#!/usr/bin/env python3
"""
Example: Web UI Evaluation Integration

Demonstrates how to integrate with ADK's web UI for visual evaluation.
Following the pattern: adk web
"""

import asyncio
import json
from pathlib import Path
import sys
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from adk_evaluator import ADKEvaluator, EvaluationCriteria


class WebUIEvaluator:
    """
    Evaluator that formats results for web UI display.
    
    This would integrate with ADK's web interface for visual evaluation
    and interactive testing.
    """
    
    def __init__(self):
        self.evaluator = ADKEvaluator()
        
    async def evaluate_for_ui(
        self,
        agent_module: str,
        test_file: str
    ) -> Dict[str, Any]:
        """
        Evaluate agent and format results for web UI.
        
        Returns JSON-formatted results suitable for display in web interface.
        """
        # Run evaluation
        results = await self.evaluator.evaluate(
            agent_module=agent_module,
            eval_dataset_file_path_or_dir=test_file
        )
        
        # Format for UI
        ui_results = {
            "summary": {
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
                "pass_rate": self._calculate_pass_rate(results)
            },
            "tests": []
        }
        
        # Add individual test results
        for result in results:
            test_info = {
                "id": result.eval_id,
                "passed": result.passed,
                "scores": result.scores,
                "details": result.details
            }
            
            # Add visual indicators
            test_info["visual"] = self._generate_visual_indicators(result)
            
            # Add errors if any
            if result.errors:
                test_info["errors"] = result.errors
            
            ui_results["tests"].append(test_info)
        
        # Add charts data
        ui_results["charts"] = self._generate_chart_data(results)
        
        return ui_results
    
    def _calculate_pass_rate(self, results):
        """Calculate pass rate percentage"""
        if not results:
            return 0.0
        return (sum(1 for r in results if r.passed) / len(results)) * 100
    
    def _generate_visual_indicators(self, result):
        """Generate visual indicators for UI display"""
        indicators = {}
        
        # Status icon
        indicators["status_icon"] = "✅" if result.passed else "❌"
        
        # Score bars (for progress bar display)
        indicators["score_bars"] = {}
        for metric, score in result.scores.items():
            indicators["score_bars"][metric] = {
                "value": score,
                "percentage": score * 100,
                "color": self._get_score_color(score)
            }
        
        return indicators
    
    def _get_score_color(self, score):
        """Get color based on score value"""
        if score >= 0.9:
            return "green"
        elif score >= 0.7:
            return "yellow"
        else:
            return "red"
    
    def _generate_chart_data(self, results):
        """Generate data for charts in web UI"""
        chart_data = {
            "metrics_distribution": {},
            "pass_fail_pie": {
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed)
            },
            "scores_timeline": []
        }
        
        # Aggregate metrics
        metrics_totals = {}
        for result in results:
            for metric, score in result.scores.items():
                if metric not in metrics_totals:
                    metrics_totals[metric] = []
                metrics_totals[metric].append(score)
        
        # Calculate averages for bar chart
        for metric, scores in metrics_totals.items():
            chart_data["metrics_distribution"][metric] = {
                "average": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0
            }
        
        # Timeline data (for line chart)
        for i, result in enumerate(results):
            chart_data["scores_timeline"].append({
                "test_number": i + 1,
                "scores": result.scores
            })
        
        return chart_data


async def simulate_web_ui_session():
    """
    Simulate a web UI evaluation session.
    
    This demonstrates how the evaluation would work in the ADK web interface.
    """
    print("=" * 50)
    print("ADK Web UI Evaluation Simulation")
    print("=" * 50)
    
    ui_evaluator = WebUIEvaluator()
    
    # Test different datasets
    test_files = [
        "datasets/vulnerability_assessment.test.json",
        "datasets/compliance_check.test.json",
        "datasets/incident_response.test.json"
    ]
    
    all_results = {}
    
    for test_file in test_files:
        print(f"\n📊 Evaluating: {test_file}")
        
        try:
            ui_results = await ui_evaluator.evaluate_for_ui(
                agent_module="security_agent",
                test_file=test_file
            )
            
            # Display summary (would be shown in web UI)
            summary = ui_results["summary"]
            print(f"   Total Tests: {summary['total_tests']}")
            print(f"   Passed: {summary['passed']} ✅")
            print(f"   Failed: {summary['failed']} ❌")
            print(f"   Pass Rate: {summary['pass_rate']:.1f}%")
            
            # Show sample visual indicators
            if ui_results["tests"]:
                test = ui_results["tests"][0]
                print(f"\n   Sample Test: {test['id']}")
                visual = test["visual"]
                print(f"   Status: {visual['status_icon']}")
                
                # Show score bars (in web UI these would be progress bars)
                for metric, bar_data in visual["score_bars"].items():
                    bar = "█" * int(bar_data["percentage"] / 10)
                    print(f"   {metric}: {bar} {bar_data['percentage']:.0f}%")
            
            all_results[test_file] = ui_results
            
        except Exception as e:
            print(f"   Error: {e}")
    
    # Generate overall dashboard data
    print("\n" + "=" * 50)
    print("Dashboard Summary")
    print("=" * 50)
    
    total_passed = sum(r["summary"]["passed"] for r in all_results.values())
    total_tests = sum(r["summary"]["total_tests"] for r in all_results.values())
    
    print(f"Overall Pass Rate: {(total_passed/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    
    # Export results for web UI (would be served via API)
    export_path = Path("evaluation_results_ui.json")
    with open(export_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results exported to: {export_path}")
    print("   (In real ADK, these would be displayed in the web interface)")


async def interactive_evaluation():
    """
    Simulate interactive evaluation session.
    
    This shows how users would interact with the evaluation system
    through the web UI.
    """
    print("\n" + "=" * 50)
    print("Interactive Evaluation Mode")
    print("=" * 50)
    
    ui_evaluator = WebUIEvaluator()
    
    # Simulate user selecting options in web UI
    print("\n🎯 Evaluation Configuration:")
    print("1. Agent: security_agent")
    print("2. Dataset: vulnerability_assessment")
    print("3. Criteria: Standard (ADK defaults)")
    
    # Run evaluation with progress updates
    print("\n⏳ Running evaluation...")
    
    # Simulate progress updates (in web UI, this would be real-time)
    steps = [
        "Loading test dataset...",
        "Initializing agent...",
        "Running test cases...",
        "Calculating metrics...",
        "Generating report..."
    ]
    
    for step in steps:
        print(f"   {step}")
        await asyncio.sleep(0.5)  # Simulate processing time
    
    # Get results
    results = await ui_evaluator.evaluate_for_ui(
        agent_module="security_agent",
        test_file="datasets/vulnerability_assessment.test.json"
    )
    
    print("\n✅ Evaluation Complete!")
    
    # Show interactive options (in web UI, these would be buttons/links)
    print("\n📋 Available Actions:")
    print("   [View Details] - See detailed test results")
    print("   [Export Report] - Download evaluation report")
    print("   [Run Again] - Re-run with different settings")
    print("   [Compare Results] - Compare with previous runs")
    
    return results


async def main():
    """Run web UI examples"""
    try:
        # Run simulated web UI session
        await simulate_web_ui_session()
        
        # Run interactive evaluation
        await interactive_evaluation()
        
        print("\n" + "=" * 50)
        print("Web UI examples completed!")
        print("In production, access via: adk web")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)