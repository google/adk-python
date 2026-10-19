# MongoDB Sample Agents

Sample app that exercises the MongoDB integration
(`google.adk.integrations.mongodb`) with **Atlas Automated Embedding**: an
`autoEmbed` index over the `description` text field makes Atlas generate
embeddings with a Voyage AI model, so the tools send each query as plain text
and no Google embedding credentials are needed. Start here to try auto
embedding first, then see the twin sample
[`mongodb_samples_google`](../mongodb_samples_google/) to test the same
integration with Google embedding models (Vertex AI `text-embedding-005`).

- `agent.py` — an ADK agent using `MongoDbToolset`'s `mongodb_vector_search`
  and `mongodb_hybrid_search` tools in auto-embedding mode.
- `test_integration.py` — a live, LLM-free integration test covering the
  search tools plus `MongoDbSessionService` and `MongoDbMemoryService`.

## Setup

1. Install the optional dependency:

   ```bash
   pip install google-adk[mongodb]
   ```

2. Export your MongoDB configuration:

   ```bash
   export MONGODB_URI="mongodb+srv://<user>:<pass>@cluster.mongodb.net/"
   export MONGODB_DATABASE="my_db"
   export MONGODB_COLLECTION="my_collection"
   ```

   The collection needs a vector search index for `mongodb_vector_search`,
   plus a full-text search index for `mongodb_hybrid_search`
   (MongoDB Atlas or MongoDB 8.0+).

   Use a least-privilege database user, not an admin: the agent only reads.
   See "Required permissions (least-privilege setup)" in
   `src/google/adk/integrations/mongodb/README.md` — the model chooses the
   collection at run time, so scope the user's `find` privilege to the
   collections the agent may search.

   To create a ready-to-use sample database with documents, pre-computed
   embeddings, and both search indexes, run the seed script from the
   repository root:

   ```bash
   set -a && source .env && set +a   # or export the vars above manually
   uv run python setup_mongodb_sample.py
   ```

3. Set a Gemini API key for the agent's model (only needed to run the agent,
   not the integration test):

   ```bash
   export GOOGLE_API_KEY="..."
   ```

## Test the integration

`test_integration.py` exercises every component of the integration against
your live MongoDB deployment — no LLM required. It checks the toolset's tool
exposure, runs vector and hybrid searches against the seeded products
collection, and walks `MongoDbSessionService` and `MongoDbMemoryService`
through create/append/recall/delete flows (cleaning up after itself):

```bash
set -a && source .env && set +a
uv run python mongodb_samples/test_integration.py
```

Exit code is `0` when all checks pass, `1` otherwise.

## Run the agent

From the repository root:

```bash
adk run mongodb_samples    # CLI
adk web                    # then pick "mongodb_samples" in the UI
```

## Deploy to Agent Engine

Agent Engine packages the app object with cloudpickle, so the MongoDB
toolset is built for it: it drops its live MongoDB client at pickle time and
rebuilds it on the runtime from `MONGODB_URI` (see "Deploying to Agent
Engine" in `src/google/adk/integrations/mongodb/README.md`). Two extra
environment variables are needed at deploy time:

```bash
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"                 # optional, this is the default
export GOOGLE_CLOUD_STAGING_BUCKET="gs://my-staging-bucket"
```

Then, from the repository root:

```bash
pip install "google-cloud-aiplatform[adk,agent_engines]"
set -a && source .env && set +a
uv run python mongodb_samples/deploy_agent_engine.py
```

The `requirements.txt` in this folder covers the runtime dependencies
(`google-adk[mongodb]` for pymongo, `dnspython` for `mongodb+srv` URIs), so
the container-based `adk deploy agent_engine mongodb_samples` flow works too
— put a `.env` with the `MONGODB_*` variables inside this folder first.

On Agent Engine, sessions and memory default to the managed Vertex AI
services; `MongoDbSessionService` and `MongoDbMemoryService` remain the
choice for self-hosted runners (`adk run`, `adk web`, Cloud Run) — they are
picklable as well when constructed with `connection_string`, should a custom
deployment wire them in. Support for wiring the MongoDB session and memory
services into Agent Engine deployments directly will be added in an upcoming
release.
