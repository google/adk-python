# STORY-301: Comprehensive Multi-Turn Integration Testing

**ID**: STORY-301
**EPIC**: SEC-003 - Testing and Validation
**Status**: Not Started
**Priority**: P0
**Size**: L

## Description

**As a** Quality Assurance Engineer,
**I want to** create a comprehensive integration test that simulates a complex, multi-turn conversation,
**So that** I can validate that the agent can maintain context, switch between different tools, and synthesize information from multiple backend services.

## Acceptance Criteria

1.  **Cross-Domain Conversation Flow**:
    *   The test must simulate a conversation that starts in one domain (e.g., Storage), moves to another (e.g., IAM), and then asks a follow-up question that relates to both.
    *   Example flow:
        1.  "Which of my storage buckets are public?" (Storage Tool)
        2.  "Show me the IAM roles for the owners of those buckets." (IAM Tool + context from previous turn)
        3.  "Generate a remediation plan for the top critical issue." (Remediation Tool)

2.  **Tool Routing Validation**:
    *   The test must verify that the correct backend service (e.g., `StorageSecurityAnalyzer`, `IAMSecurityAnalyzer`) is invoked in response to each user query.
    *   Mocking will be used to intercept service calls and assert that the correct functions were called with the expected parameters.

3.  **Context Preservation**:
    *   The test must confirm that context (e.g., specific resource names, identified vulnerabilities) is maintained across multiple turns.
    *   The agent's responses should reflect an understanding of the conversation history.

4.  **Data Synthesis**:
    *   The test will include a query that requires the agent to synthesize information from at least two different backend services to formulate an answer.

5.  **Reliable and Fast Execution**:
    *   The test must use FastAPI's `TestClient` to run in memory without a live server.
    *   All external dependencies (GCP APIs) must be mocked to ensure the test is fast, reliable, and independent of the network.

## Technical Design

*   **Test File**: A new test file will be created at `contributing/samples/security_agent/backend/tests/test_comprehensive_multiturn_flow.py`.
*   **Testing Framework**: The test will use `pytest` and FastAPI's `TestClient`.
*   **Mocking**: Python's `unittest.mock.patch` will be used to mock the service methods within the chat endpoint logic (e.g., `analyze_storage`, `analyze_iam`). This isolates the test to the agent's routing and context management logic.
*   **Assertions**: The test will assert:
    *   HTTP status codes (e.g., `200 OK`).
    *   Content of the JSON response.
    *   Which mocked backend services were called and with what arguments.

## Test Plan

*   Develop a test case that covers a realistic, complex conversation flow.
*   The test will simulate a single user session across all turns.
*   The test will validate both the agent's final response and the internal tool calls it makes.
