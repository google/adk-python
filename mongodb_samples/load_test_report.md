# MongoDB Integration — Load Test Report

- **Date:** 2026-10-14 16:04:31 IST
- **Host:** venkatesh.shanbhag@M-N3343HCRGH (Darwin 25.6.0)
- **Python:** 3.13.2
- **MongoDB server:** 8.0.34
- **Network baseline (median ping RTT):** 280.5 ms — every operation sits on top of this floor
- **Concurrency levels:** [1, 5, 10, 20]

## Method

Each scenario runs `level x ops-per-worker` operations with at most
`level` in flight (asyncio semaphore). Latency is per-operation wall
time from this client, so it includes the network round trip to Atlas
and, where noted, server-side Voyage AI embedding (auto-embedding
mode) or a client-side Vertex AI embedding call (Google mode). The
session round trip bundles four operations (create + append + get +
delete); the memory scenario bundles ingest (idempotent upsert over a
pre-created pool) + keyword recall. All state writes used a dedicated
app name and were cleaned up after the run.

## Results

### vector_search (Atlas auto-embedding)

| Concurrency | Ops | Errors | Ops/sec | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0 | 2.7 | 364.1 | 362.6 | 376.1 | 376.1 | 376.1 |
| 5 | 50 | 0 | 9.2 | 529.8 | 378.4 | 1651.4 | 2335.0 | 2335.0 |
| 10 | 100 | 0 | 19.7 | 496.4 | 388.2 | 868.2 | 1881.5 | 1930.2 |
| 20 | 200 | 0 | 32.7 | 582.7 | 542.0 | 712.4 | 1733.4 | 2124.1 |

### hybrid_search (Atlas auto-embedding)

| Concurrency | Ops | Errors | Ops/sec | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 0 | 2.5 | 397.7 | 396.5 | 419.5 | 419.5 | 419.5 |
| 5 | 50 | 0 | 12.5 | 387.1 | 386.8 | 414.3 | 435.1 | 435.1 |
| 10 | 100 | 0 | 24.9 | 387.0 | 384.9 | 424.1 | 496.9 | 512.8 |
| 20 | 200 | 0 | 35.2 | 542.6 | 524.9 | 644.5 | 820.8 | 847.0 |

### vector_search (Google embedding + Vertex round trip)

| Concurrency | Ops | Errors | Ops/sec | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0 | 0.5 | 1851.8 | 1330.6 | 2312.0 | 2312.0 | 2312.0 |
| 5 | 25 | 0 | 2.5 | 1895.2 | 2227.8 | 2453.5 | 2468.4 | 2468.4 |
| 10 | 50 | 0 | 3.8 | 2051.3 | 2166.4 | 5868.1 | 6075.5 | 6075.5 |
| 20 | 100 | 0 | 7.1 | 2494.7 | 2235.5 | 6025.3 | 7011.0 | 7027.6 |

### session_service round trip

| Concurrency | Ops | Errors | Ops/sec | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | 0 | 0.2 | 5121.6 | 4526.3 | 7393.3 | 7393.3 | 7393.3 |
| 5 | 25 | 0 | 1.0 | 4678.3 | 4333.2 | 5984.0 | 6603.5 | 6603.5 |
| 10 | 50 | 0 | 2.0 | 4578.2 | 4374.0 | 5561.2 | 6965.5 | 6965.5 |
| 20 | 100 | 0 | 2.9 | 6446.5 | 6382.6 | 7192.5 | 7618.8 | 8972.5 |

### memory_service ingest + recall

| Concurrency | Ops | Errors | Ops/sec | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Max (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0 | 0.4 | 2329.8 | 1843.8 | 3328.9 | 3328.9 | 3328.9 |
| 5 | 15 | 0 | 2.1 | 2162.0 | 1894.1 | 3048.6 | 3718.1 | 3718.1 |
| 10 | 30 | 0 | 4.1 | 2019.6 | 1828.4 | 3029.5 | 3297.6 | 3297.6 |
| 20 | 60 | 0 | 6.7 | 2724.6 | 2598.2 | 4231.0 | 4275.2 | 4307.3 |

## Observations

- **vector_search (Atlas auto-embedding):** p50 362.6 ms at concurrency 1 grows to 542.0 ms at concurrency 20; throughput scales 12.1x (2.7 -> 32.7 ops/s); 0 errors across 360 ops.
- **hybrid_search (Atlas auto-embedding):** p50 396.5 ms at concurrency 1 grows to 524.9 ms at concurrency 20; throughput scales 14.1x (2.5 -> 35.2 ops/s); 0 errors across 360 ops.
- **vector_search (Google embedding + Vertex round trip):** p50 1330.6 ms at concurrency 1 grows to 2235.5 ms at concurrency 20; throughput scales 14.2x (0.5 -> 7.1 ops/s); 0 errors across 180 ops.
- **session_service round trip:** p50 4526.3 ms at concurrency 1 grows to 6382.6 ms at concurrency 20; throughput scales 14.5x (0.2 -> 2.9 ops/s); 0 errors across 180 ops.
- **memory_service ingest + recall:** p50 1843.8 ms at concurrency 1 grows to 2598.2 ms at concurrency 20; throughput scales 16.8x (0.4 -> 6.7 ops/s); 0 errors across 108 ops.

Throughput scaling below the concurrency increase indicates a
saturation point — client CPU, the asyncio executor, the Atlas tier,
or (Google mode) the embedding API round trip. Compare p50 against
the network baseline to separate server work from transit time.

## Conclusions (run under the combined split PRs #7509–#7512 + #7533)

1. **Zero failures:** 1,188 operations across all five scenarios, 0 errors —
   with the new code paths active: `bulk_write` memory ingestion, batched
   user-state reads, default-on index creation, and ranked/capped recall.
2. **Auto-embedding search is healthy and slightly faster** than the previous
   run (p50 363 ms vs 393 ms at c=1; 542 ms vs 587 ms at c=20); hybrid p95 at
   c=20 improved 714 ms -> 645 ms. Server-side work stays ~90–130 ms per call
   on top of the 281 ms network baseline.
3. **Session round trip unchanged** (p50 ~4.5 s at c=1; 2.9 rt/s at c=20) —
   the four sequential RTTs dominate; the new batched `list_sessions` read is
   not on this path, and default-on index creation was a no-op (indexes
   already present, idempotent).
4. **Memory ingest + recall unchanged to slightly better** (p50 ~1.8–1.9 s at
   low concurrency) — `bulk_write` ingestion works against real Atlas with
   zero errors.
5. **Google embedding mode shows a tail-latency regression (external).**
   p95 at c=10/20 rose from ~2.5 s to ~5.9–6.0 s. Isolation check: the raw
   Vertex AI `embed_content` call alone measured p50 878 ms / max 3.4 s, and
   a focused c=10 re-run reproduced p95 10.9 s with 0 errors. The
   Google-embedding code path (`_embed_query`, the `queryVector` branch) is
   byte-identical to the previous run — this is Vertex AI endpoint variance,
   not a code regression. Watch it; if it persists, consider batching embed
   calls or retry-with-backoff guidance for that mode.

**Caveats:** single client machine, high-RTT path to the Atlas region, and a
small (10-document) dataset — these numbers characterize the connector's
latency profile and concurrency behavior, not Atlas's absolute search
capacity.
