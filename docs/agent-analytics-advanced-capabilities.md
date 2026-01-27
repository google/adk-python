# **ADK Agent Analytics: SQL-First Advanced Capabilities Design Document**

## **Executive Summary**

This document outlines the architecture for the ADK BigQuery Agent Analytics Plugin, transitioned to a fully SQL-native implementation. By leveraging **BigQuery AI Functions** (AI.GENERATE, AI.EMBED, AI.EXTRACT), **BigQuery Graph**, and **BigQuery ML**, we enable sophisticated agent evaluation and memory management directly within the data warehouse.

This approach addresses the primary challenges in the agentic space—observability, stateful memory, and trajectory evaluation—without the overhead of external Python middleware.

## **Part 1: Trace-Based Evaluation Harness**

### **1.1 Background & Motivation**

Agent evaluation has shifted from binary success/failure metrics to comprehensive trajectory analysis. According to recent research ([Evaluation and Benchmarking of LLM Agents](https://arxiv.org/abs/2507.21504)), modern evaluation requires examining:

* **Reasoning Layer**: Planning quality and dependency handling.  
* **Action Layer**: Tool selection accuracy and argument correctness.  
* **Overall Execution**: Step efficiency and task completion.

### **1.2 SQL-Native LLM-as-Judge**

We utilize AI.GENERATE to perform "forensic" analysis of traces. This replaces the need for external evaluation harnesses.

```sql

-- Batch Evaluate Agent Trajectories for Efficiency and Reasoning
CREATE OR REPLACE TABLE `{project}.{dataset}.eval_results` AS
WITH session_trajectories AS (
  SELECT
    session_id,
    STRING_AGG(
      FORMAT("Step %d: Tool=%s, Args=%s, Status=%s", 
             step_index, 
             JSON_EXTRACT_SCALAR(attributes, '$.tool_name'),
             JSON_EXTRACT_SCALAR(attributes, '$.tool_args'),
             JSON_EXTRACT_SCALAR(attributes, '$.status')),
      "\n" ORDER BY timestamp ASC
    ) AS trajectory_str
  FROM `{project}.{dataset}.agent_events`
  WHERE event_type IN ('TOOL_STARTING', 'TOOL_COMPLETED', 'TOOL_ERROR')
  GROUP BY session_id
)
SELECT
  session_id,
  AI.GENERATE(
    FORMAT("""
      You are an expert AI Agent Evaluator. 
      Analyze the following execution trajectory against the goal of step-efficiency.
      Trajectory:
      %s
      
      Provide a structured JSON response:
      {
        "task_completion": float (0-1),
        "step_efficiency": float (0-1),
        "tool_usage_accuracy": float (0-1),
        "critique": string
      }
    """, trajectory_str)
  ) AS evaluation_json
FROM session_trajectories;

```

## **Part 2: Long-Horizon Agent Memory**

### **2.1 Memory Architecture**

According to [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564), memory serves as the cornerstone for long-horizon reasoning. We implement three memory types natively:

| Memory Type | Description | BigQuery Feature |
| :---- | :---- | :---- |
| **Episodic** | Recall of past similar interactions | AI.EMBED \+ VECTOR\_SEARCH |
| **Semantic** | Learned facts and user preferences | AI.EXTRACT |
| **Working** | Current session context | CTE-based windowing |

### **2.2 Semantic Retrieval and Vector Search**

This replaces traditional RAG pipelines by keeping embeddings and retrieval logic inside BigQuery.

```sql

-- 1. Create a Native Vector Store
CREATE OR REPLACE TABLE `{project}.{dataset}.trace_vector_store` AS
SELECT 
  session_id, user_id, timestamp, content,
  AI.EMBED(content) AS embedding
FROM `{project}.{dataset}.agent_events` 
WHERE event_type = 'AGENT_COMPLETED';

-- 2. Retrieve Cross-Session Context for Current User
CREATE OR REPLACE TABLE `{project}.{dataset}.user_memory_context` AS
SELECT 
  base.trace_text,
  distance
FROM VECTOR_SEARCH(
  TABLE `{project}.{dataset}.trace_vector_store`,
  'embedding',
  (SELECT AI.EMBED("How did the user want their reports formatted in the past?")),
  top_k => 3
) AS search_results
WHERE user_id = 'user_99';

```

### **2.3 Knowledge Extraction via AI.EXTRACT**

Instead of storing raw chat logs, we "compress" interactions into structured facts using AI.EXTRACT, as suggested by the [CORAL framework](https://openreview.net/forum?id=NBGlItueYE).

```sql

-- Extracting User Preferences into Permanent Knowledge Store
INSERT INTO `{project}.{dataset}.user_profiles` (user_id, profile_json)
SELECT
  user_id,
  AI.EXTRACT(
    STRING_AGG(content, " | "),
    ['preferred language', 'reporting frequency', 'technical expertise level']
  )
FROM `{project}.{dataset}.agent_events`
WHERE event_type = 'USER_MESSAGE_RECEIVED'
GROUP BY user_id;

```

## **Part 3: Behavioral Graph Analytics**

### **3.1 Topology Analysis**

Modeling traces as a graph allows us to detect structural failures like "Delegation Loops" or "Dependency Deadlocks" which are difficult to query in flat tables.

```sql

-- Define the Trace Property Graph
CREATE OR REPLACE PROPERTY GRAPH `{project}.{dataset}.agent_trace_graph`
NODE TABLES (
  `{project}.{dataset}.agents` KEY (agent_id),
  `{project}.{dataset}.tools` KEY (tool_name),
  `{project}.{dataset}.sessions` KEY (session_id)
)
EDGE TABLES (
  `{project}.{dataset}.delegations`
    SOURCE KEY (parent_id) REFERENCES agents (agent_id)
    DESTINATION KEY (child_id) REFERENCES agents (agent_id)
    LABEL delegates,
  `{project}.{dataset}.tool_invocations`
    SOURCE KEY (agent_id) REFERENCES agents (agent_id)
    DESTINATION KEY (tool_name) REFERENCES tools (tool_name)
    LABEL calls
);

```

### **3.2 Scenario: Detecting Infinite Delegation Loops (Cycles)**

In multi-agent systems, agents may enter an infinite loop by delegating back and forth. BigQuery Graph identifies these cycles instantly.

```sql

-- Detect cycles of length 2 to 5 in agent delegations
SELECT *
FROM GRAPH_TABLE(
  `{project}.{dataset}.agent_trace_graph`,
  MATCH (a)-[e:delegates]->{2,5}(a)
  COLUMNS (a.agent_id, "Circular Delegation Detected" as issue_type)
);

```

### **3.3 Scenario: Execution Bottleneck Detection (Centrality)**

Identify "Hub" tools that are central to most failing traces. If a tool has high degree centrality in sessions that end with AGENT\_ERROR, it is a systemic bottleneck.

```sql

-- Find tools with the highest number of calls in failed sessions
SELECT 
  tool_name, 
  COUNT(*) as call_count
FROM GRAPH_TABLE(
  `{project}.{dataset}.agent_trace_graph`,
  MATCH (s:sessions)-[:includes]->(a:agents)-[c:calls]->(t:tools)
  WHERE s.final_status = 'ERROR'
  COLUMNS (t.tool_name)
)
GROUP BY tool_name
ORDER BY call_count DESC;

```

### **3.4 Scenario: Data Lineage & Entity Propagation**

Track how a specific entity (e.g., order\_id) propagates across different tools. This visualizes the "lineage" of a data point as it is transformed by the agent.

```sql

-- Trace the flow of a specific entity across the tool graph
SELECT 
  path
FROM GRAPH_TABLE(
  `{project}.{dataset}.agent_trace_graph`,
  MATCH p = (t1:tools)-[:calls*]->(t2:tools)
  WHERE t1.output_json LIKE '%order_123%' AND t2.input_json LIKE '%order_123%'
  COLUMNS (JSON_ARRAY_AGG(t2.tool_name) as path)
);

```

## **Part 4: Predictive & Diagnostic Analytics**

### **4.1 Anomaly Detection**

Using BigQuery ML's AUTOENCODER, we identify sessions that deviate from "normal" behavioral patterns (e.g., unusual tool-calling frequency).

```sql

-- Train Anomaly Detection Model
CREATE OR REPLACE MODEL `{project}.{dataset}.behavior_anomaly_model`
OPTIONS(model_type='AUTOENCODER') AS
SELECT 
  COUNTIF(event_type = 'TOOL_STARTING') as tools,
  COUNTIF(event_type = 'LLM_REQUEST') as llms,
  AVG(latency_ms) as lat
FROM `{project}.{dataset}.agent_events`
GROUP BY session_id;

-- Detect and Explain Anomalies
WITH outliers AS (
  SELECT * FROM ML.DETECT_ANOMALIES(
    MODEL `{project}.{dataset}.behavior_anomaly_model`,
    STRUCT(0.01 AS contamination),
    TABLE `{project}.{dataset}.session_metrics`
  ) WHERE is_anomaly = TRUE
)
SELECT 
  session_id,
  AI.GENERATE(FORMAT("Explain why this session with %d tool calls is anomalous.", tool_count))
FROM outliers;

```

## 

## **References & Citations**

### **Academic Research**

* **Evaluation**: [TRAJECT-Bench: A Trajectory-Aware Benchmark](https://arxiv.org/html/2510.04550v1)  
* **Memory**: [CORAL: Cognitive Resource Self-Allocation](https://openreview.net/forum?id=NBGlItueYE)  
* **Context Management**: [JetBrains Research on Long-Horizon Agents (2025)](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)

### **Industry Documentation**

* [BigQuery AI Functions Overview](https://cloud.google.com/bigquery/docs/ai-introduction)  
* [BigQuery ML.DETECT\_ANOMALIES](https://cloud.google.com/bigquery/docs/anomaly-detection-overview)  
* [BigQuery Property Graph Documentation](https://www.google.com/search?q=https://cloud.google.com/bigquery/docs/graph-introduction)
