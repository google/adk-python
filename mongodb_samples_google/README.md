# MongoDB Sample Agents — Google Embedding Models

Sample app that exercises the MongoDB integration
(`google.adk.integrations.mongodb`) in the **default embedding mode**: the
search tools embed each query with the Google embedding model (Vertex AI
`text-embedding-005`) through the genai client and send the vector to
`$vectorSearch` as `queryVector`.

This is the Google-embedding twin of [`mongodb_samples`](../mongodb_samples/),
which tests Atlas Automated Embedding instead (Atlas embeds queries with a
Voyage AI model on an `autoEmbed` index, so no Google credentials are needed
there). Run `mongodb_samples` first to try auto embedding, then this sample
to compare against Google embedding models.

- `agent.py` — an ADK agent using `MongoDbToolset`'s `mongodb_vector_search`
  and `mongodb_hybrid_search` tools with a Vertex AI genai client.
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
   export MONGODB_GOOGLE_COLLECTION="products_google"   # optional, this is the default
   ```

   The collection needs documents with an `embedding` field of
   `text-embedding-005` vectors, a vector search index over that field for
   `mongodb_vector_search`, plus a full-text search index for
   `mongodb_hybrid_search` (MongoDB Atlas or MongoDB 8.0+).

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

3. Export the Google Cloud configuration used to embed queries (and to seed
   the embeddings above):

   ```bash
   export GOOGLE_CLOUD_PROJECT="my-gcp-project"
   export GOOGLE_CLOUD_LOCATION="us-central1"   # optional, this is the default
   ```

4. Set a Gemini API key for the agent's model (only needed to run the agent,
   not the integration test):

   ```bash
   export GOOGLE_API_KEY="..."
   ```

## Test the integration

`test_integration.py` exercises every component of the integration against
your live MongoDB deployment — no LLM required. It checks the toolset's tool
exposure, runs vector and hybrid searches in Google embedding mode against
the seeded collection, and walks `MongoDbSessionService` and
`MongoDbMemoryService` through create/append/recall/delete flows (cleaning up
after itself):

```bash
set -a && source .env && set +a
uv run python mongodb_samples_google/test_integration.py
```

Exit code is `0` when all checks pass, `1` otherwise.

## Run the agent

From the repository root:

```bash
adk run mongodb_samples_google    # CLI
adk web                           # then pick "mongodb_samples_google" in the UI
```

## Deploy to Agent Engine

Agent Engine packages the app object with cloudpickle, so the MongoDB
toolset is built for it: it drops its live MongoDB client at pickle time and
rebuilds it on the runtime from `MONGODB_URI` (see "Deploying to Agent
Engine" in `src/google/adk/integrations/mongodb/README.md`). The
query-embedding genai client is likewise rebuilt lazily from the ambient
environment — the deploy script therefore sets `GOOGLE_GENAI_USE_VERTEXAI`
so the ambient client resolves to Vertex AI, where `text-embedding-005`
lives. Two extra environment variables are needed at deploy time:

```bash
export GOOGLE_CLOUD_PROJECT="my-gcp-project"
export GOOGLE_CLOUD_LOCATION="us-central1"                 # optional, this is the default
export GOOGLE_CLOUD_STAGING_BUCKET="gs://my-staging-bucket"
```

Then, from the repository root:

```bash
pip install "google-cloud-aiplatform[adk,agent_engines]"
set -a && source .env && set +a
uv run python mongodb_samples_google/deploy_agent_engine.py
```

The `requirements.txt` in this folder covers the runtime dependencies
(`google-adk[mongodb]` for pymongo, `dnspython` for `mongodb+srv` URIs), so
the container-based `adk deploy agent_engine mongodb_samples_google` flow
works too — put a `.env` with the `MONGODB_*` variables inside this folder
first.

On Agent Engine, sessions and memory default to the managed Vertex AI
services; `MongoDbSessionService` and `MongoDbMemoryService` remain the
choice for self-hosted runners (`adk run`, `adk web`, Cloud Run) — they are
picklable as well when constructed with `connection_string`, should a custom
deployment wire them in. Support for wiring the MongoDB session and memory
services into Agent Engine deployments directly will be added in an upcoming
release.
