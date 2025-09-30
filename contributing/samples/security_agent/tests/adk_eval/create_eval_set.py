#!/usr/bin/env python3
"""
Create ADK 1.15.0 compliant evaluation sets for the BigQuery Security Agent
"""

import json
from datetime import datetime
from google.adk.evaluation.local_eval_sets_manager import EvalSet, EvalCase, Invocation
from google.adk.evaluation.eval_rubrics import Rubric

def create_security_agent_eval_set():
    """Create evaluation set for BigQuery Security Agent"""

    # Create eval cases
    eval_cases = []

    # Test 1: List datasets (tests basic tool usage)
    eval_cases.append(EvalCase(
        eval_id="test_list_datasets",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'List my BigQuery datasets'}]},
                rubrics=[
                    Rubric(
                        rubric_id="check_datasets",
                        rubricContent={
                            "textProperty": "The response should mention the security_insights dataset"
                        }
                    )
                ]
            )
        ]
    ))

    # Test 2: Security focus (tests agent's primary dataset understanding)
    eval_cases.append(EvalCase(
        eval_id="test_security_focus",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'What is your primary dataset for security analysis?'}]},
                rubrics=[
                    Rubric(
                        rubric_id="primary_dataset",
                        rubricContent={
                            "textProperty": "Response should clearly identify security_insights as the primary dataset and mention security_findings table"
                        }
                    )
                ]
            )
        ]
    ))

    # Test 3: Query security data (tests security tool usage)
    eval_cases.append(EvalCase(
        eval_id="test_security_query",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'Show me security issues'}]},
                rubrics=[
                    Rubric(
                        rubric_id="security_query",
                        rubricContent={
                            "textProperty": "Should use query_security_insights or get_security_insights_summary and query from security_insights dataset"
                        }
                    )
                ]
            )
        ]
    ))

    # Test 4: Table exploration (tests exploration tools)
    eval_cases.append(EvalCase(
        eval_id="test_table_exploration",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'What tables are in the security_insights dataset?'}]},
                rubrics=[
                    Rubric(
                        rubric_id="list_tables",
                        rubricContent={
                            "textProperty": "Should use list_tables tool and list tables including security_findings, firewall_rules, iam_accounts"
                        }
                    )
                ]
            )
        ]
    ))

    # Test 5: Greeting with security context
    eval_cases.append(EvalCase(
        eval_id="test_greeting",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'Hello, what can you help me with?'}]},
                rubrics=[
                    Rubric(
                        rubric_id="greeting_response",
                        rubricContent={
                            "textProperty": "Should mention security analysis capabilities, reference security_insights dataset, and be conversational and helpful"
                        }
                    )
                ]
            )
        ]
    ))

    # Create the eval set
    eval_set = EvalSet(
        eval_set_id="bigquery_security_agent_eval_v1",
        eval_cases=eval_cases
    )

    return eval_set

def create_tool_specific_eval_set():
    """Create evaluation set focused on specific tool testing"""

    eval_cases = []

    # Test each security tool
    eval_cases.append(EvalCase(
        eval_id="test_security_summary_tool",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'Give me a security summary'}]},
                rubrics=[
                    Rubric(
                        rubric_id="summary_tool",
                        rubricContent={
                            "textProperty": "Must call get_security_insights_summary and provide statistics from security_insights dataset"
                        }
                    )
                ]
            )
        ]
    ))

    eval_cases.append(EvalCase(
        eval_id="test_security_statistics_tool",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'Show me security statistics grouped by severity'}]},
                rubrics=[
                    Rubric(
                        rubric_id="statistics_tool",
                        rubricContent={
                            "textProperty": "Must call get_security_statistics with group_by='severity' parameter"
                        }
                    )
                ]
            )
        ]
    ))

    eval_cases.append(EvalCase(
        eval_id="test_query_critical_issues",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'Find all critical security issues'}]},
                rubrics=[
                    Rubric(
                        rubric_id="critical_query",
                        rubricContent={
                            "textProperty": "Should call query_security_insights and filter by severity='CRITICAL'"
                        }
                    )
                ]
            )
        ]
    ))

    # Test BigQuery tools
    eval_cases.append(EvalCase(
        eval_id="test_hello_world",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'Run hello world in BigQuery'}]},
                rubrics=[
                    Rubric(
                        rubric_id="hello_world",
                        rubricContent={
                            "textProperty": "Should call hello_world tool and return BigQuery hello world message"
                        }
                    )
                ]
            )
        ]
    ))

    eval_cases.append(EvalCase(
        eval_id="test_table_schema",
        conversation=[
            Invocation(
                invocation_id="1",
                userContent={'parts': [{'text': 'Show me the schema for security_findings table'}]},
                rubrics=[
                    Rubric(
                        rubric_id="schema_check",
                        rubricContent={
                            "textProperty": "Should call get_table_schema and reference security_insights.security_findings"
                        }
                    )
                ]
            )
        ]
    ))

    eval_set = EvalSet(
        eval_set_id="security_agent_tools_eval_v1",
        eval_cases=eval_cases
    )

    return eval_set

def save_eval_set(eval_set, filename):
    """Save eval set to JSON file"""
    # Convert to dict for JSON serialization
    eval_dict = eval_set.model_dump(mode='json')

    with open(filename, 'w') as f:
        json.dump(eval_dict, f, indent=2, default=str)

    print(f"✅ Created eval set: {filename}")
    print(f"   - Eval set ID: {eval_set.eval_set_id}")
    print(f"   - Number of test cases: {len(eval_set.eval_cases)}")
    print(f"   - Test IDs: {[case.eval_id for case in eval_set.eval_cases]}")

if __name__ == "__main__":
    # Create general evaluation set
    general_eval_set = create_security_agent_eval_set()
    save_eval_set(general_eval_set, "tests/adk_eval/eval_security_agent.json")

    # Create tool-specific evaluation set
    tools_eval_set = create_tool_specific_eval_set()
    save_eval_set(tools_eval_set, "tests/adk_eval/eval_tools_test.json")

    print("\n🎯 Evaluation sets created successfully!")
    print("Run with: adk eval agents tests/adk_eval/eval_security_agent.json --print_detailed_results")