# Agent Engine Deployment — Verification Report

**Date:** 2026-10-14
**Code under test:** `theshanbhag/adk-python@mongodb-hardening` — the union of
split PRs #7509 (auto-embedding), #7510 (picklable toolset), #7511 (session
service), #7512 (memory service), #7533 (hardening: `connection_string_env_var`),
plus the gemini-code-assist review fixes.
**GCP project:** `gcp-pov` (787220387490), region `us-central1`
**Atlas:** MongoDB 8.0.34, database `mongodb_sample_db` (collections
`products` autoEmbed index, `products_google` text-embedding-005 vectors)

## Verdict: PASS — both applications deploy and answer correctly end to end.

## What was deployed

Two ADK applications, each an `AdkApp` wrapping the sample agent with the
`MongoDbToolset`:

| App | Engine resource | Embedding mode |
| :--- | :--- | :--- |
| `mongodb_samples` | `reasoningEngines/3509280750924988416` | Atlas Automated Embedding (`autoEmbed`, Voyage AI — no Google creds) |
| `mongodb_samples_google` | `reasoningEngines/404048807853031424` | Google `text-embedding-005` via Vertex AI on the runtime |

Deployment specifics:

- The runtime installs the integration branch directly from the fork:
  `google-adk[mongodb] @ git+...@mongodb-hardening` (PyPI's google-adk does
  not yet carry pickling / `connection_string_env_var` / the services).
- The sample agents were switched to `connection_string_env_var="MONGODB_URI"`,
  so the pickled package stores only the variable *name*.
- `MONGODB_URI` is injected at runtime from Secret Manager
  (`projects/787220387490/secrets/mongodb-uri`, version `latest`) via
  `SecretRef` in `env_vars`.
- `google-cloud-aiplatform[adk,agent_engines]` + `dnspython` in requirements;
  `GOOGLE_GENAI_USE_VERTEXAI=TRUE` on the Google-embedding engine.

## Security verification

`spec.deployment_spec` of both engines, read back after deployment:

```
env (plaintext):  MONGODB_DATABASE, MONGODB_COLLECTION / GOOGLE_GENAI_USE_VERTEXAI, telemetry flag
secret_env:       MONGODB_URI -> SecretRef(secret="mongodb-uri", version="latest")
```

The database password appears nowhere in the deployment spec, and (per the
unit test in #7533) not in the pickled package either — the toolset resolves
the env var on the runtime. This closes the review finding "the database
password is saved in the deployment package".

## Functional verification (live queries against the deployed engines)

**Auto-embedding engine** — query: *"What robot vacuums do you have for pet hair?"*

1. Model called `mongodb_hybrid_search(collection_name="products", query=…, text_search_field="description")`.
2. Tool returned `status: SUCCESS`, top row `Robot Vacuum Pro` (Atlas embedded the query server-side — the runtime made no embedding call).
3. Final answer correctly summarized the hit with its `search_score`.

**Google-embedding engine** — query: *"Which espresso machine comes with a milk frother?"*

1. Model called `mongodb_hybrid_search(collection_name="products_google", …)`.
2. Runtime rebuilt the genai client from the ambient environment, embedded the query with Vertex AI `text-embedding-005`, `$vectorSearch` returned `Espresso Maker Barista` first (`status: SUCCESS`).
3. Final answer: "The Espresso Maker Barista … comes with a milk frother." ✓

**Session continuity (managed Vertex AI sessions):** on the auto engine,
`async_create_session` + two turns — turn 2 ("What did I just ask about?")
answered "You just asked about vacuums for pet hair." ✓ Sessions on Agent
Engine use the managed Vertex AI session service, as documented (the MongoDB
session/memory services are not yet wireable there — follow-up).

## Issues encountered and resolved

| Issue | Resolution |
| :--- | :--- |
| `gcloud` CLI user auth expired | Used Application Default Credentials via Python clients for all setup (bucket reuse, Secret Manager, IAM). |
| Reasoning Engine service agent `service-787220387490@gcp-svc-aiplatform-re…` did not exist before the first engine, so the pre-grant of `secretmanager.secretAccessor` failed | The runtime resolved the secret anyway (this POC project's default AI Platform identity already has Secret Manager access). **For production:** create the engine once, then grant `roles/secretmanager.secretAccessor` on the secret to the Reasoning Engine service agent, per the README guidance. |

## Review data battery (2026-10-14, later run)

`mongodb_samples/generate_review_data.py` drove both deployed engines with a
10-question product battery (20 queries total), mirrored every interaction
into MongoDB (sessions/events/state, deterministic ids so re-runs upsert),
and ingested all sessions into both memory modes. **Nothing was dropped** —
everything is reviewable in Atlas:

| Collection | Docs | Content |
| :--- | ---: | :--- |
| `sessions` | 29 | incl. 20 `review-user-*` sessions mirroring AE runs |
| `events` | 114 | user message → tool call → tool response → answer per run |
| `user_states` | 27 | per-user `last_query` state |
| `memories` | 58 | keyword-mode recall docs |
| `memories_vector` | 43 | vector-mode docs (768-dim embeddings) |
| `products` / `products_google` | 10 + 10 | unchanged catalogs |

**Result: 19/20 queries surfaced the expected product** (per-app tool choice
was correct throughout — hybrid for most, vector for the two "vibe" queries).
The single miss: *"Do you have kitchen appliances for frying without oil?"*
on the Google engine — the tool returned `SUCCESS` but ranked Steam Mop Elite
above Air Fryer Max and the answer didn't mention the fryer. The
auto-embedding engine ranked the same top hit yet its answer did mention the
fryer. This is small-catalog relevance tuning (rank-fusion weights on a
10-product store), not a connector failure.

Per-query detail: `mongodb_samples/review_data_report.json`.

## Vector memory retrieval — durable proof in Atlas

Earlier E2E runs of the vector memory service (PR #7573) cleaned up their
scratch collections, so nothing was visible afterwards. A persistent proof
collection now exists in Atlas:

- **Collection:** `mongodb_sample_db.memories_vector` — 3 documents, each
  with a `text` field and a 768-dimension `embedding` (Vertex AI
  `text-embedding-005`), written by `MongoDbMemoryService(retrieval_mode="vector")`.
- **Index:** `memory_vector_index` (vectorSearch, cosine, filter fields
  `app_name`/`user_id`), status READY — created by the service itself via
  `ensure_indexes`.
- **Semantic recall proof:** query "coffee maker fixing" returned "My
  espresso machine stopped working and I need a repair service." first —
  zero keyword overlap, pure vector similarity.

Check in Atlas UI: Database → cluster → Collections → `mongodb_sample_db` →
`memories_vector` (embedding field on each doc) → Search Indexes tab →
`memory_vector_index`.

## Latency notes

Both engines answered in a few seconds per turn. Google-embedding mode pays
the known Vertex AI embed round trip per search (see `load_test_report.md` —
that endpoint is currently showing elevated tail latency).

## Cost / cleanup

Both engines were left running (min 1 instance each). To tear down:

```bash
uv run python -c "
from vertexai import agent_engines
import vertexai
vertexai.init(project='gcp-pov', location='us-central1')
for r in ['3509280750924988416', '404048807853031424']:
    agent_engines.delete(f'projects/787220387490/locations/us-central1/reasoningEngines/{r}', force=True)
"
```
