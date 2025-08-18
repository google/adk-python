# Improved Architecture Diagrams

This document provides visual diagrams for the proposed ADK-aligned security agent architecture. The new design emphasizes a clear separation of concerns, with an intelligent Router Agent at the core of the system.

## 1. High-Level System Architecture

This diagram illustrates the overall structure, showing how the new Router Agent becomes the central hub for processing user queries and delegating to specialist agents.

```mermaid
graph TD
    subgraph Frontend
        A[Streamlit UI]
    end

    subgraph Backend (ADK Platform)
        B(API Endpoint)
        C{Router Agent}
        D[Storage Agent]
        E[IAM Agent]
        F[Network Agent]
        G[Compliance Agent]
        H[GCP Tools]
    end

    subgraph Google Cloud
        I[GCP APIs]
    end

    A -- HTTP/WebSocket --> B;
    B -- Forwards Query --> C;
    C -- Analyzes Intent & Routes --> D;
    C -- Analyzes Intent & Routes --> E;
    C -- Analyzes Intent & Routes --> F;
    C -- Analyzes Intent & Routes --> G;
    D -- Uses --> H;
    E -- Uses --> H;
    F -- Uses --> H;
    G -- Uses --> H;
    H -- Calls --> I;
```

## 2. Agent Interaction Flow (Sequence Diagram)

This sequence diagram details the step-by-step interaction for a complex query that requires orchestration between multiple specialist agents.

**Query**: "Do any of my public storage buckets have IAM users with excessive permissions?"

```mermaid
sequenceDiagram
    participant User
    participant API_Endpoint as API Endpoint
    participant RouterAgent as Router Agent
    participant StorageAgent as Storage Agent
    participant IAMAgent as IAM Agent
    participant GCP_Tools as GCP Tools

    User->>API_Endpoint: "Check public buckets for risky IAM roles"

    API_Endpoint->>RouterAgent: Forward query

    RouterAgent->>RouterAgent: Analyze query intent (LLM)
    Note over RouterAgent: Intent: Find public buckets AND analyze their IAM policies.

    RouterAgent->>StorageAgent: 1. Find public buckets
    StorageAgent->>GCP_Tools: list_buckets(public=True)
    GCP_Tools-->>StorageAgent: [bucket_A, bucket_C]

    StorageAgent-->>RouterAgent: Public buckets: [bucket_A, bucket_C]

    RouterAgent->>IAMAgent: 2. Analyze IAM for [bucket_A, bucket_C]
    IAMAgent->>GCP_Tools: get_iam_policy(bucket_A), get_iam_policy(bucket_C)
    GCP_Tools-->>IAMAgent: IAM policies

    IAMAgent->>IAMAgent: Analyze policies for risky roles (e.g., editor, owner)
    IAMAgent-->>RouterAgent: Findings: bucket_C has 'allUsers' with 'objectViewer'

    RouterAgent->>RouterAgent: Synthesize findings into a final response
    RouterAgent-->>API_Endpoint: "Bucket 'bucket_C' is public and allows all users to view objects. This is a high-risk configuration."

    API_Endpoint-->>User: Return synthesized response
```

## 3. Comparison: Old vs. New Architecture

### Old Architecture (Keyword-Based Routing)

```mermaid
graph TD
    A[User Query] --> B{API Endpoint};
    B -- "if 'bucket' in query" --> C[Storage Agent];
    B -- "if 'iam' in query" --> D[IAM Agent];
    B -- "else" --> E[Coordinator Agent];
    C --> F[GCP Tools];
    D --> F;
    E --> F;
```
**Problem**: Brittle, not intelligent, and bypasses the coordinator.

### New Architecture (LLM-Powered Routing)

```mermaid
graph TD
    A[User Query] --> B{API Endpoint};
    B --> C[Router Agent];
    C -- LLM-based Delegation --> D[Specialist Agents];
    D --> E[GCP Tools];
```
**Improvement**: Flexible, intelligent, and centralized routing logic that aligns with ADK best practices.