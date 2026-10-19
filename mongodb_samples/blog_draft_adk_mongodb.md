# Blog Submission — MongoDB Google ADK Integration

**Status:** Under review
**Your Name:** Nasir Qureshi
**Your Team:** Builder Relations and AI Frameworks
**Submission date:** Sep 29, 2026
**Requested post date:** Oct 29, 2026
**Last Updated:** Sep 29, 2026
**Wrike Ticket:** https://www.wrike.com/open.htm?id=4575480409

Please ensure edit access has been granted to Evan Alger and Kevin O'Rourke for editing and staging before submitting the post to wrike.

## Why is this post being written / what goals are we accomplishing with this post

**What it is:**
We are launching the MongoDB Google ADK Integration that gives developers the building blocks to create agents that can retrieve relevant information, preserve conversation state, and build long-term memory, all with MongoDB.

**Why it matters:**
As agents move into production, they need more than powerful models. They need secure, dependable access to data and the ability to retain context over time. With the MongoDB Google ADK Integration, developers can bring retrieval, session persistence, and memory into their agent architecture with a cohesive, ADK-native developer experience.

## Author(s)/Byline

Venkatesh Shanbhag, Ajay Tandon, Nasir Qureshi

## Imagery

adk-mongo-diagram.png

## Journey Stage

Consumption / Problem-Trigger / Onboarding / Awareness

## DACI

| Role | Name |
|---|---|
| Driver | Nasir Qureshi |
| Approver | Ajay Tandon |
| Consultants | Venkatesh Shanbhag |
| Informed | Alex Bevilacqua |

## Meta Description

The MongoDB Google ADK Integration helps developers build agents that can store sessions and long-term memory with MongoDB.

## Proposed Headline

Introducing the MongoDB Google ADK Integration: Durable Memory for Agent State

---

## Body Copy

**Authors:** Nasir Qureshi, Ajay Tandon, Venkatesh Shanbhag

### What is the Google ADK?

The Google Agent Development Kit (ADK) is an open-source agent development framework that lets you build, debug, and deploy reliable AI agents at enterprise scale. It is available in Python, TypeScript, Go, Java, and Kotlin.

ADK offers developers an extensive set of tools for creating, deploying, assessing, and scaling agents, enabling developers to build anything from personal AI assistants to mission-critical business workflows.

However, developers still need to build their own tooling for a data store that provides durable agent memory for agent state. MongoDB has built an integration that gives developers a cohesive, ADK-native experience for using MongoDB as the data platform for agents.

### Introducing MongoDB Google ADK Integration

We're excited to announce the MongoDB Google ADK Integration, which provides developers with the building blocks to create durable memory for agent state.

As agents move from prototypes to real-world applications, they need secure, dependable access to data and the ability to retain context over a long time. With this integration, developers get data retrieval, session storage, and long-term memory for their agents in a cohesive, ADK-native developer experience.

For instance, teams can prototype an agent that searches a product catalog or knowledge base, then evolve it into a production experience that preserves user context across conversations and recalls relevant prior interactions.

With MongoDB as the data layer, developers can keep the data their agents search over, the conversation state that needs to survive a restart or a scaled-out instance, and the memories agents build over time, all in the same place as the rest of their application data. That means no separate database to stand up for retrieval, no pipeline syncing app data into a different memory index, and no added latency pulling context from another system before the agent can respond.

ADK supplies the agent framework and tool interface; this integration provides the data and persistence layer beneath it, so developers can take an agent from prototype to production. These agents can maintain conversation and session memory independent of any single machine, and carry a user's history across sessions.

### How does MongoDB provide agent session and memory storage?

Great agent experiences improve with context. This integration introduces MongoDbSessionService to persist sessions, events, and application-, user-, and session-scoped state to MongoDB.

This means developers can build agents that retain the context needed for longer-running and multi-turn interactions. The session service includes optimistic concurrency control to help applications safely manage concurrent updates, while configurable collections make it easier to fit the service into existing data architectures.

The following Python code shows how you can add MongoDbSessionService to your agent.

```python
from google.adk.integrations.mongodb import MongoDbSessionService
from google.adk.runners import Runner

session_service = MongoDbSessionService(
    connection_string="mongodb+srv://...",
    database_name="my_app",
)

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
)
```

For agents that need memory beyond a single session, MongoDbMemoryService ingests session events into MongoDB collections and recalls relevant memories through keyword matching. Memory ingestion is idempotent, which prevents duplicates when a session is added again.

The following Python code shows how you can add MongoDbMemoryService to your agent.

```python
from google.adk.integrations.mongodb import MongoDbMemoryService
from google.adk.runners import Runner

memory_service = MongoDbMemoryService(
    connection_string="mongodb+srv://...",
    database_name="my_app",
)

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=session_service,
    memory_service=memory_service,
)

# Ingest a finished session's events into long-term memory (idempotent).
await memory_service.add_session_to_memory(session)

# Recall memories whose keywords match the query.
memories = await memory_service.search_memory(
    app_name="my_app", user_id="user_123", query="preferred shipping address"
)
```

Together, these services help agents become more useful over time without requiring developers to stitch together separate systems for retrieval, state, and memory.

*Figure 1: The MongoDB Google ADK Integration adds session and memory storage for single or multi-agent workflows.*

### How does MongoDB Search help agents with data retrieval?

Having memory for agent state is essential, but agents also need fast access to the operational data they search over, the same product catalogs, documents, and records your application already queries. This is where MongoDB's search capabilities come in, and they're not a separate bolt-on. MongoDB Vector Search, built on Voyage AI's embedding models, is the same engine backing MemoryService's semantic retrieval in this integration.

Developers can also combine it with MongoDB Search (full-text) using the $rankFusion or $scoreFusion aggregation stages for hybrid search, then reorder results with the $rerank stage's Voyage AI reranking model, an approach that's improved retrieval quality by up to 30% in MongoDB's own benchmarks.

This gives agents access to the right information and context at the right time, delivering better responses in production. Whether you are building product discovery, knowledge assistants, support experiences, or domain-specific copilots, agents can automatically leverage semantic and hybrid retrieval to ground their responses in your data.

The application owns the database connection, database name, and search configuration. The agent gets focused tools to call, without exposing those operational details in its context.

The following Python code shows how you can add MongoDB Search with your agents.

```python
from google.adk.agents import Agent
from google.adk.integrations.mongodb import MongoDbToolset

search_tools = MongoDbToolset(
    connection_string="mongodb+srv://...",
    database_name="products_db",
)

agent = Agent(
    model="gemini-2.5-flash",
    name="product_search_agent",
    instruction=(
        "Search the products collection with your MongoDB tools. "
        "Embed the user's query before calling the search tools."
    ),
    tools=[search_tools],
)
```

Using MongoDB as the operational data store, the memory layer for agent state, and the search engine over both, developers get one platform instead of three, on whatever deployment they run it on. There's no separate vector database to stand up, no search engine to sync against a second copy of the data, and no added latency from hopping between systems before the agent can respond.

### Which deployment versions does this integration support?

The integration supports MongoDB Atlas and self-managed MongoDB 8.0+ deployments for vector and hybrid search. Developers can provide a connection string or bring an existing PyMongo client, and configure search behavior through MongoDbToolSettings, including index names, embedding fields, limits, and candidate counts.

That flexibility makes it possible to align the integration with existing data models, deployment choices, and performance requirements.

### Get started today

Install ADK with the MongoDB extra:

```
pip install google-adk[mongodb]
```

Then connect your agent to the data and context it needs to deliver better experiences.
