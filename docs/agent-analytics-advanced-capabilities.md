# ADK Agent Analytics: Advanced Capabilities Design Document

## Executive Summary

This document outlines three advanced capabilities for the ADK BigQuery Agent Analytics Plugin:

1. **Trace-Based Evaluation Harness** - Automated evaluation of agent behavior using stored traces
2. **Long-Horizon Agent Memory** - Context and memory management for agents using historical trace data
3. **BigQuery AI/ML Integration** - Leveraging BigQuery's advanced features for agent analytics

---

## Part 1: Trace-Based Evaluation Harness for ADK Agents

### 1.1 Background & Motivation

Agent evaluation has evolved from simple task completion metrics to comprehensive trajectory analysis. According to recent research surveys ([Evaluation and Benchmarking of LLM Agents](https://arxiv.org/html/2507.16504v1)), modern agent evaluation requires examining:

- **Reasoning Layer**: Planning quality, dependency handling, plan adherence
- **Action Layer**: Tool selection accuracy, argument correctness, call ordering
- **Overall Execution**: Task completion, step efficiency, staying on-task

The ADK BigQuery Agent Analytics Plugin already captures rich trace data that can power such evaluation. The plugin stores events including:
- `USER_MESSAGE_RECEIVED` - User inputs
- `AGENT_STARTING/COMPLETED` - Agent lifecycle
- `LLM_REQUEST/RESPONSE` - Model interactions with prompts and completions
- `TOOL_STARTING/COMPLETED/ERROR` - Tool execution details
- `INVOCATION_STARTING/COMPLETED` - Full invocation lifecycle

### 1.2 Evaluation Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ADK Trace-Based Evaluation                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │  BigQuery   │───▶│   Trace      │───▶│   Evaluation        │   │
│  │  Analytics  │    │   Retriever  │    │   Engine            │   │
│  │  Store      │    │              │    │                     │   │
│  └─────────────┘    └──────────────┘    └─────────────────────┘   │
│         │                                        │                 │
│         │                                        ▼                 │
│         │                              ┌─────────────────────┐     │
│         │                              │   Metric Scorers    │     │
│         │                              ├─────────────────────┤     │
│         │                              │ • Trajectory Match  │     │
│         │                              │ • LLM Judge         │     │
│         │                              │ • Tool Accuracy     │     │
│         │                              │ • Task Completion   │     │
│         │                              │ • Step Efficiency   │     │
│         │                              └─────────────────────┘     │
│         │                                        │                 │
│         ▼                                        ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Evaluation Results Store                  │  │
│  │              (BigQuery ML.EVALUATE Integration)              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Core Evaluation Metrics

Based on the ADK evaluation framework (`src/google/adk/evaluation/`) and industry standards ([LangChain Trajectory Evals](https://docs.langchain.com/langsmith/trajectory-evals), [DeepEval Agent Evaluation](https://deepeval.com/guides/guides-ai-agent-evaluation)):

#### 1.3.1 Trajectory Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| `TOOL_TRAJECTORY_AVG_SCORE` | How closely agent's tool calls match expected trajectory | Compare tool sequences from trace vs golden trajectory |
| `TOOL_TRAJECTORY_IN_ORDER_SCORE` | Whether tools were called in correct order | Order-aware sequence matching |
| `RESPONSE_MATCH_SCORE` | Final response similarity to expected | Embedding similarity or LLM judge |
| `STEP_EFFICIENCY_SCORE` | Ratio of necessary vs actual steps | Count trace events vs optimal path |

#### 1.3.2 Quality Metrics

| Metric | Description | Data Source |
|--------|-------------|-------------|
| `PLAN_QUALITY` | Quality of agent's reasoning/planning | LLM_REQUEST content analysis |
| `PLAN_ADHERENCE` | Whether agent followed its plan | Compare stated plan vs executed tools |
| `TOOL_SELECTION_ACCURACY` | Correct tool chosen for task | TOOL_STARTING events vs expected |
| `ARGUMENT_CORRECTNESS` | Tool arguments match requirements | TOOL_STARTING attributes |

### 1.4 Implementation Design

#### 1.4.1 Trace Retrieval SQL

```sql
-- Retrieve complete session trace for evaluation
SELECT
  event_type,
  agent,
  timestamp,
  latency_ms,
  JSON_EXTRACT_SCALAR(content, '$.summary') as content_summary,
  JSON_EXTRACT_SCALAR(attributes, '$.tool_name') as tool_name,
  JSON_EXTRACT_SCALAR(attributes, '$.tool_args') as tool_args,
  JSON_EXTRACT_SCALAR(attributes, '$.status') as status
FROM `{project}.{dataset}.{table}`
WHERE session_id = @session_id
  AND event_type IN (
    'USER_MESSAGE_RECEIVED',
    'AGENT_STARTING', 'AGENT_COMPLETED',
    'TOOL_STARTING', 'TOOL_COMPLETED', 'TOOL_ERROR',
    'LLM_REQUEST', 'LLM_RESPONSE'
  )
ORDER BY timestamp ASC
```

#### 1.4.2 Evaluation Harness Class

```python
from google.adk.evaluation import AgentEvaluator, EvalMetric
from google.cloud import bigquery

class BigQueryTraceEvaluator:
    """Evaluate agent traces stored in BigQuery."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
        eval_metrics: list[EvalMetric] = None
    ):
        self.client = bigquery.Client(project=project_id)
        self.table_ref = f"{project_id}.{dataset_id}.{table_id}"
        self.metrics = eval_metrics or [
            EvalMetric.TOOL_TRAJECTORY_AVG_SCORE,
            EvalMetric.RESPONSE_MATCH_SCORE,
        ]

    async def evaluate_session(
        self,
        session_id: str,
        golden_trajectory: list[dict],
        golden_response: str = None
    ) -> dict[str, float]:
        """Evaluate a single session against golden data."""
        # 1. Retrieve trace from BigQuery
        trace = await self._get_session_trace(session_id)

        # 2. Extract tool trajectory
        actual_trajectory = self._extract_tool_trajectory(trace)

        # 3. Compute metrics
        results = {}
        for metric in self.metrics:
            if metric == EvalMetric.TOOL_TRAJECTORY_AVG_SCORE:
                results[metric.name] = self._compute_trajectory_score(
                    actual_trajectory, golden_trajectory
                )
            elif metric == EvalMetric.RESPONSE_MATCH_SCORE:
                actual_response = self._extract_final_response(trace)
                results[metric.name] = await self._compute_response_match(
                    actual_response, golden_response
                )

        return results

    async def evaluate_batch(
        self,
        eval_dataset: list[dict]
    ) -> pd.DataFrame:
        """Evaluate multiple sessions from an eval dataset."""
        results = []
        for item in eval_dataset:
            scores = await self.evaluate_session(
                session_id=item['session_id'],
                golden_trajectory=item['expected_trajectory'],
                golden_response=item.get('expected_response')
            )
            results.append({
                'session_id': item['session_id'],
                **scores
            })
        return pd.DataFrame(results)
```

#### 1.4.3 LLM-as-Judge Evaluation

Following [TRAJECT-Bench](https://arxiv.org/html/2510.04550v1) methodology:

```python
TRAJECTORY_JUDGE_PROMPT = """
You are evaluating an AI agent's task execution trajectory.

## Task Description
{task_description}

## Agent Trajectory
{trajectory_json}

## Evaluation Criteria
1. **Task Completion** (0-10): Did the agent successfully complete the task?
2. **Efficiency** (0-10): Were the steps taken necessary and minimal?
3. **Tool Usage** (0-10): Were the right tools used with correct arguments?
4. **Reasoning Quality** (0-10): Was the agent's reasoning sound?

Provide scores and brief justification for each criterion.
Output as JSON: {"task_completion": X, "efficiency": X, "tool_usage": X, "reasoning": X, "overall": X, "justification": "..."}
"""

async def llm_judge_trajectory(
    trajectory: list[dict],
    task_description: str,
    model: str = "gemini-2.5-flash"
) -> dict:
    """Use LLM to judge trajectory quality."""
    # Format trajectory for judge
    trajectory_str = json.dumps(trajectory, indent=2)

    prompt = TRAJECTORY_JUDGE_PROMPT.format(
        task_description=task_description,
        trajectory_json=trajectory_str
    )

    # Call LLM for evaluation
    response = await model.generate_content(prompt)
    return json.loads(response.text)
```

### 1.5 Deterministic Replay for Debugging

Based on [Trustworthy AI Agents: Deterministic Replay](https://www.sakurasky.com/blog/missing-primitives-for-trustworthy-ai-part-8/):

```python
class TraceReplayRunner:
    """Replay agent sessions deterministically for debugging."""

    def __init__(self, trace_store: BigQueryTraceStore):
        self.trace_store = trace_store

    async def replay_session(
        self,
        session_id: str,
        replay_mode: str = "full"  # "full" | "step" | "tool_only"
    ) -> ReplayResult:
        """
        Replay a recorded session step by step.

        Modes:
        - full: Replay all events including LLM responses
        - step: Pause at each step for inspection
        - tool_only: Only replay tool calls with recorded responses
        """
        trace = await self.trace_store.get_session_trace(session_id)

        replay_context = ReplayContext()
        for event in trace:
            if event['event_type'] == 'LLM_RESPONSE':
                # Substitute recorded LLM response
                replay_context.inject_llm_response(event['content'])
            elif event['event_type'] == 'TOOL_COMPLETED':
                # Substitute recorded tool output
                replay_context.inject_tool_response(
                    tool_name=event['attributes']['tool_name'],
                    response=event['content']
                )

        return await self._execute_replay(replay_context)
```

---

## Part 2: Long-Horizon Agent Memory from BigQuery Traces

### 2.1 Background

Long-horizon agents face significant challenges with context management. According to [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564):

> "Memory serves as the cornerstone of foundation model-based agents, underpinning their ability to perform long-horizon reasoning, adapt continually, and interact effectively with complex environments."

The [CORAL framework](https://openreview.net/forum?id=NBGlItueYE) demonstrates that:
> "LLM agents often falter on long-horizon tasks due to cognitive overload, as their working memory becomes cluttered with expanding and irrelevant information."

### 2.2 Memory Architecture Using BigQuery Traces

```
┌─────────────────────────────────────────────────────────────────────┐
│              Long-Horizon Agent Memory Architecture                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                      ┌─────────────────────────┐  │
│  │   Agent     │◀────────────────────▶│   Working Memory        │  │
│  │   Runtime   │                      │   (Context Window)      │  │
│  └─────────────┘                      └─────────────────────────┘  │
│         │                                        ▲                 │
│         │                                        │                 │
│         ▼                                        │ Retrieve        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                 Memory Retrieval Layer                       │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │  │
│  │  │  Semantic    │ │   Temporal   │ │   Graph-Based        │ │  │
│  │  │  Search      │ │   Recency    │ │   Retrieval          │ │  │
│  │  │  (Embeddings)│ │   Weighting  │ │   (Relationships)    │ │  │
│  │  └──────────────┘ └──────────────┘ └──────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    BigQuery Trace Store                      │  │
│  │                                                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │  Session    │  │   User      │  │   Embeddings        │  │  │
│  │  │  Traces     │  │   Profiles  │  │   (AI.EMBED)        │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │  │
│  │                                                               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Memory Types from Trace Data

#### 2.3.1 Session Memory (Short-Term)

```python
class BigQuerySessionMemory(BaseSessionService):
    """
    Session memory backed by BigQuery traces.
    Enables cross-session context for the same user.
    """

    async def get_recent_context(
        self,
        user_id: str,
        session_id: str,
        lookback_sessions: int = 5,
        max_events: int = 50
    ) -> list[dict]:
        """Retrieve recent context from past sessions."""
        query = f"""
        WITH recent_sessions AS (
          SELECT DISTINCT session_id, MIN(timestamp) as start_time
          FROM `{self.table_ref}`
          WHERE user_id = @user_id
            AND session_id != @current_session
          GROUP BY session_id
          ORDER BY start_time DESC
          LIMIT @lookback_sessions
        )
        SELECT
          e.session_id,
          e.event_type,
          e.timestamp,
          JSON_EXTRACT_SCALAR(e.content, '$.summary') as content
        FROM `{self.table_ref}` e
        JOIN recent_sessions rs ON e.session_id = rs.session_id
        WHERE e.event_type IN ('USER_MESSAGE_RECEIVED', 'AGENT_COMPLETED')
        ORDER BY e.timestamp DESC
        LIMIT @max_events
        """
        return await self._execute_query(query, {
            'user_id': user_id,
            'current_session': session_id,
            'lookback_sessions': lookback_sessions,
            'max_events': max_events
        })
```

#### 2.3.2 Episodic Memory (Past Interactions)

```python
class BigQueryEpisodicMemory:
    """
    Episodic memory retrieves relevant past interactions
    based on semantic similarity.
    """

    async def retrieve_similar_episodes(
        self,
        query: str,
        user_id: str,
        top_k: int = 5
    ) -> list[Episode]:
        """Find past interactions similar to current query."""
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)

        # Search using BigQuery vector similarity
        sql = f"""
        SELECT
          session_id,
          content,
          timestamp,
          ML.DISTANCE(embedding, @query_embedding, 'COSINE') as similarity
        FROM `{self.table_ref}`
        WHERE user_id = @user_id
          AND embedding IS NOT NULL
        ORDER BY similarity ASC
        LIMIT @top_k
        """

        results = await self._execute_query(sql, {
            'user_id': user_id,
            'query_embedding': query_embedding,
            'top_k': top_k
        })

        return [Episode.from_row(r) for r in results]
```

#### 2.3.3 Semantic Memory (Learned Knowledge)

```python
class BigQuerySemanticMemory:
    """
    Semantic memory extracts and stores learned facts
    from agent interactions.
    """

    async def extract_and_store_knowledge(
        self,
        session_id: str
    ) -> list[KnowledgeFact]:
        """
        Use LLM to extract knowledge facts from session traces.
        Store as structured data in BigQuery.
        """
        # Get session trace
        trace = await self._get_session_trace(session_id)

        # Use AI.GENERATE to extract facts
        extraction_sql = f"""
        SELECT AI.GENERATE(
          'Extract key facts and user preferences from this conversation.
           Output as JSON array of facts.',
          @conversation_text
        ) as extracted_facts
        """

        facts = await self._execute_query(extraction_sql, {
            'conversation_text': self._format_trace(trace)
        })

        # Store facts with embeddings for retrieval
        await self._store_facts(facts, session_id)

        return facts
```

### 2.4 Context Management Strategies

Based on [JetBrains Research on Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/):

#### 2.4.1 Observation Masking

```python
class ContextManager:
    """Manage agent context to prevent cognitive overload."""

    def __init__(self, max_context_tokens: int = 32000):
        self.max_tokens = max_context_tokens

    def select_relevant_context(
        self,
        current_task: str,
        available_memories: list[Memory],
        current_context: list[Message]
    ) -> list[Memory]:
        """
        Select most relevant memories for current task.
        Implements observation masking to reduce noise.
        """
        # Score memories by relevance
        scored_memories = []
        for memory in available_memories:
            relevance = self._compute_relevance(memory, current_task)
            recency = self._compute_recency_weight(memory.timestamp)
            score = relevance * 0.7 + recency * 0.3
            scored_memories.append((memory, score))

        # Select top memories within token budget
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        selected = []
        token_count = self._count_tokens(current_context)

        for memory, score in scored_memories:
            memory_tokens = self._count_tokens([memory])
            if token_count + memory_tokens < self.max_tokens:
                selected.append(memory)
                token_count += memory_tokens

        return selected
```

#### 2.4.2 Progressive Summarization

```python
async def summarize_old_context(
    self,
    context: list[Message],
    preserve_recent: int = 10
) -> list[Message]:
    """
    Summarize older context to save tokens while preserving information.
    """
    if len(context) <= preserve_recent:
        return context

    old_context = context[:-preserve_recent]
    recent_context = context[-preserve_recent:]

    # Use BigQuery AI.GENERATE for summarization
    summary_sql = f"""
    SELECT AI.GENERATE(
      'Summarize the key points from this conversation history,
       preserving important facts, user preferences, and decisions made.',
      @conversation_history
    ) as summary
    """

    summary = await self._execute_query(summary_sql, {
        'conversation_history': self._format_messages(old_context)
    })

    summary_message = Message(
        role="system",
        content=f"Summary of previous conversation: {summary}"
    )

    return [summary_message] + recent_context
```

### 2.5 User Profile Building from Traces

```python
class UserProfileBuilder:
    """Build and maintain user profiles from trace data."""

    async def build_profile(self, user_id: str) -> UserProfile:
        """
        Analyze all user traces to build a profile.
        """
        sql = f"""
        WITH user_interactions AS (
          SELECT
            session_id,
            timestamp,
            JSON_EXTRACT_SCALAR(content, '$.summary') as content,
            event_type
          FROM `{self.table_ref}`
          WHERE user_id = @user_id
            AND event_type = 'USER_MESSAGE_RECEIVED'
        ),
        -- Use AI to extract preferences
        preference_extraction AS (
          SELECT AI.GENERATE(
            'Analyze these user messages and extract:
             1. Topics of interest
             2. Communication style preferences
             3. Common requests/patterns
             Output as JSON.',
            STRING_AGG(content, ' | ')
          ) as preferences
          FROM user_interactions
        )
        SELECT * FROM preference_extraction
        """

        result = await self._execute_query(sql, {'user_id': user_id})
        return UserProfile.from_json(result['preferences'])
```

---

## Part 3: BigQuery AI/ML Integration for Agent Analytics

### 3.1 BigQuery AI Functions Overview

Google Cloud has introduced powerful AI functions in BigQuery ([BigQuery Gen AI Functions](https://cloud.google.com/blog/products/data-analytics/new-bigquery-gen-ai-functions-for-better-data-analysis/)):

| Function | Purpose | Use Case for Agent Analytics |
|----------|---------|------------------------------|
| `AI.GENERATE` | Text generation with Gemini | Trace summarization, evaluation |
| `AI.EMBED` | Generate embeddings | Semantic search over traces |
| `AI.SIMILARITY` | Compute embedding similarity | Find similar sessions |
| `ML.DETECT_ANOMALIES` | Anomaly detection | Identify unusual agent behavior |
| `ML.GENERATE_TEXT` | Text generation (100x throughput) | Batch trace analysis |
| `ML.GENERATE_EMBEDDING` | Embedding generation (30x throughput) | Index all traces |

### 3.2 Embedding-Based Trace Search

```sql
-- Create embeddings for all agent traces
CREATE OR REPLACE TABLE `{project}.{dataset}.trace_embeddings` AS
SELECT
  session_id,
  event_type,
  timestamp,
  content,
  ML.GENERATE_EMBEDDING(
    MODEL `{project}.{dataset}.embedding_model`,
    STRUCT(JSON_EXTRACT_SCALAR(content, '$.summary') AS content)
  ).ml_generate_embedding_result AS embedding
FROM `{project}.{dataset}.agent_events`
WHERE event_type IN ('USER_MESSAGE_RECEIVED', 'AGENT_COMPLETED');

-- Semantic search over traces
SELECT
  session_id,
  content,
  ML.DISTANCE(
    embedding,
    (SELECT ML.GENERATE_EMBEDDING(
      MODEL `{project}.{dataset}.embedding_model`,
      STRUCT(@query AS content)
    ).ml_generate_embedding_result),
    'COSINE'
  ) AS distance
FROM `{project}.{dataset}.trace_embeddings`
ORDER BY distance ASC
LIMIT 10;
```

### 3.3 Anomaly Detection for Agent Behavior

Based on [BigQuery ML Anomaly Detection](https://cloud.google.com/bigquery/docs/anomaly-detection-overview):

#### 3.3.1 Time Series Anomaly Detection (Latency)

```sql
-- Create ARIMA model for latency prediction
CREATE OR REPLACE MODEL `{project}.{dataset}.latency_model`
OPTIONS(
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'hour',
  time_series_data_col = 'avg_latency',
  auto_arima = TRUE,
  data_frequency = 'HOURLY'
) AS
SELECT
  TIMESTAMP_TRUNC(timestamp, HOUR) AS hour,
  AVG(latency_ms) AS avg_latency
FROM `{project}.{dataset}.agent_events`
WHERE event_type = 'LLM_RESPONSE'
GROUP BY hour;

-- Detect latency anomalies
SELECT *
FROM ML.DETECT_ANOMALIES(
  MODEL `{project}.{dataset}.latency_model`,
  STRUCT(0.95 AS anomaly_prob_threshold),
  (
    SELECT
      TIMESTAMP_TRUNC(timestamp, HOUR) AS hour,
      AVG(latency_ms) AS avg_latency
    FROM `{project}.{dataset}.agent_events`
    WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
      AND event_type = 'LLM_RESPONSE'
    GROUP BY hour
  )
)
WHERE is_anomaly = TRUE;
```

#### 3.3.2 Autoencoder Anomaly Detection (Behavior Patterns)

```sql
-- Create features for behavior analysis
CREATE OR REPLACE TABLE `{project}.{dataset}.session_features` AS
SELECT
  session_id,
  COUNT(*) AS total_events,
  COUNTIF(event_type = 'TOOL_STARTING') AS tool_calls,
  COUNTIF(event_type = 'TOOL_ERROR') AS tool_errors,
  COUNTIF(event_type = 'LLM_REQUEST') AS llm_calls,
  AVG(latency_ms) AS avg_latency,
  MAX(latency_ms) AS max_latency,
  TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), SECOND) AS session_duration
FROM `{project}.{dataset}.agent_events`
GROUP BY session_id;

-- Create autoencoder for anomaly detection
CREATE OR REPLACE MODEL `{project}.{dataset}.behavior_anomaly_model`
OPTIONS(
  model_type = 'AUTOENCODER',
  activation_fn = 'RELU',
  hidden_units = [16, 8, 16],
  l2_reg = 0.0001,
  learn_rate = 0.001
) AS
SELECT
  total_events,
  tool_calls,
  tool_errors,
  llm_calls,
  avg_latency,
  session_duration
FROM `{project}.{dataset}.session_features`;

-- Detect anomalous sessions
SELECT
  session_id,
  *
FROM ML.DETECT_ANOMALIES(
  MODEL `{project}.{dataset}.behavior_anomaly_model`,
  STRUCT(0.01 AS contamination),
  TABLE `{project}.{dataset}.session_features`
)
WHERE is_anomaly = TRUE;
```

### 3.4 BigQuery Knowledge Engine Integration

The new [BigQuery Knowledge Engine](https://cloud.google.com/blog/products/data-analytics/data-analytics-innovations-at-next25) can power intelligent agent analytics:

```python
class AgentKnowledgeEngine:
    """
    Leverage BigQuery Knowledge Engine for
    semantic understanding of agent traces.
    """

    async def semantic_search(
        self,
        natural_language_query: str
    ) -> list[dict]:
        """
        Search traces using natural language.
        Knowledge Engine translates to SQL.
        """
        # BigQuery Knowledge Engine handles NL->SQL
        sql = f"""
        -- @nl_query: {natural_language_query}
        -- Knowledge Engine interprets and executes
        SELECT *
        FROM `{self.table_ref}`
        WHERE AI.SEMANTIC_MATCH(content, @query) > 0.8
        """
        return await self._execute_with_knowledge_engine(
            natural_language_query
        )

    async def get_data_insights(
        self,
        question: str
    ) -> str:
        """
        Get AI-powered insights about agent behavior.
        Uses Knowledge Engine's data insights feature.
        """
        sql = f"""
        SELECT AI.GENERATE(
          'Based on the agent trace data, answer: ' || @question,
          (SELECT STRING_AGG(content, '\\n')
           FROM `{self.table_ref}`
           LIMIT 1000)
        ) as answer
        """
        return await self._execute_query(sql, {'question': question})
```

### 3.5 Graph-Based Trace Analysis

Using BigQuery with graph capabilities ([Timbr BigQuery Graph](https://timbr.ai/timbr-posts/visualizing-and-traversing-bigquery-data-as-a-connected-graph-2/)):

```sql
-- Model agent traces as a graph
-- Nodes: Sessions, Users, Agents, Tools
-- Edges: Interactions, Tool Calls, Agent Delegations

-- Find all paths from user to successful task completion
WITH RECURSIVE agent_graph AS (
  -- Base case: user messages
  SELECT
    user_id AS source,
    session_id AS target,
    'USER_SESSION' AS edge_type,
    1 AS depth
  FROM `{project}.{dataset}.agent_events`
  WHERE event_type = 'USER_MESSAGE_RECEIVED'

  UNION ALL

  -- Agent to tool relationships
  SELECT
    agent AS source,
    JSON_EXTRACT_SCALAR(attributes, '$.tool_name') AS target,
    'AGENT_TOOL' AS edge_type,
    depth + 1
  FROM `{project}.{dataset}.agent_events`
  JOIN agent_graph ON session_id = target
  WHERE event_type = 'TOOL_STARTING'
    AND depth < 10
)
SELECT * FROM agent_graph;
```

### 3.6 Batch Evaluation Pipeline

Leveraging [BigQuery's 100x throughput improvements](https://cloud.google.com/blog/products/data-analytics/bigquery-enhancements-to-boost-gen-ai-inference):

```sql
-- Batch evaluate all sessions from the past day
CREATE OR REPLACE TABLE `{project}.{dataset}.session_evaluations` AS
WITH session_traces AS (
  SELECT
    session_id,
    STRING_AGG(
      CONCAT(event_type, ': ', JSON_EXTRACT_SCALAR(content, '$.summary')),
      '\n' ORDER BY timestamp
    ) AS trace_text
  FROM `{project}.{dataset}.agent_events`
  WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  GROUP BY session_id
)
SELECT
  session_id,
  trace_text,
  ML.GENERATE_TEXT(
    MODEL `{project}.{dataset}.eval_model`,
    STRUCT(
      CONCAT(
        'Evaluate this agent trace on a scale of 1-10 for:\n',
        '1. Task completion\n',
        '2. Efficiency\n',
        '3. Tool usage\n',
        'Trace:\n', trace_text,
        '\n\nOutput as JSON: {"task_completion": X, "efficiency": X, "tool_usage": X}'
      ) AS prompt
    ),
    STRUCT(0.1 AS temperature, 500 AS max_output_tokens)
  ).ml_generate_text_result AS evaluation
FROM session_traces;
```

---

## Part 4: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. Extend BigQuery analytics plugin schema for embeddings
2. Implement trace retrieval and formatting utilities
3. Create basic evaluation metrics (trajectory match, tool accuracy)

### Phase 2: Evaluation Harness (Weeks 3-4)
1. Build `BigQueryTraceEvaluator` class
2. Implement LLM-as-judge evaluation
3. Create evaluation dashboard integration
4. Add deterministic replay for debugging

### Phase 3: Memory System (Weeks 5-6)
1. Implement `BigQuerySessionMemory` with cross-session context
2. Build embedding-based episodic memory retrieval
3. Create semantic memory extraction pipeline
4. Implement context management (observation masking, summarization)

### Phase 4: BigQuery AI Integration (Weeks 7-8)
1. Set up embedding generation pipeline (AI.EMBED)
2. Implement anomaly detection models
3. Create batch evaluation pipeline
4. Integrate with BigQuery Knowledge Engine

### Phase 5: Production Hardening (Weeks 9-10)
1. Performance optimization and caching
2. Cost management (slot reservations, materialized views)
3. Documentation and examples
4. Integration tests and benchmarks

---

## References

### Academic Papers
- [Evaluation and Benchmarking of LLM Agents: A Survey](https://arxiv.org/html/2507.16504v1)
- [TRAJECT-Bench: A Trajectory-Aware Benchmark for Evaluating Agentic Tool Use](https://arxiv.org/html/2510.04550v1)
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564)
- [CORAL: Cognitive Resource Self-Allocation for Long-Horizon Tasks](https://openreview.net/forum?id=NBGlItueYE)

### Industry Resources
- [LangChain Trajectory Evaluations](https://docs.langchain.com/langsmith/trajectory-evals)
- [DeepEval Agent Evaluation Guide](https://deepeval.com/guides/guides-ai-agent-evaluation)
- [Braintrust Trace-Driven Evaluation](https://medium.com/@braintrustdata/evaluating-agents-with-trace-driven-insights-9ad3bfed820e)
- [JetBrains Context Management Research](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)

### BigQuery Documentation
- [BigQuery ML Anomaly Detection](https://cloud.google.com/bigquery/docs/anomaly-detection-overview)
- [BigQuery AI.GENERATE Function](https://docs.cloud.google.com/bigquery/docs/generate-text)
- [BigQuery ML.GENERATE_EMBEDDING](https://docs.cloud.google.com/bigquery/docs/generate-text-embedding)
- [BigQuery Gen AI Throughput Improvements](https://cloud.google.com/blog/products/data-analytics/bigquery-enhancements-to-boost-gen-ai-inference)
- [BigQuery Knowledge Engine](https://cloud.google.com/blog/products/data-analytics/data-analytics-innovations-at-next25)

### ADK Documentation
- `src/google/adk/evaluation/` - ADK Evaluation Framework
- `src/google/adk/memory/` - ADK Memory Services
- `src/google/adk/sessions/` - ADK Session Management
- `src/google/adk/plugins/bigquery_agent_analytics_plugin/` - BigQuery Analytics Plugin
