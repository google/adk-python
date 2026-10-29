---
type: Reference
resource: https://cloud.google.com/blog/topics/public-datasets/bitcoin-in-bigquery-blockchain-analytics-on-public-data
title: Duplicate Transactions Metric
description: An anomaly detection metric to find historical duplicate transactions
  across different blocks.
tags:
- metric
- anomaly-detection
- bitcoin
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:15:47+00:00'
sources:
- id: gcp-blog
  title: 'Bitcoin in BigQuery: blockchain analytics on public data'
  resource: https://cloud.google.com/blog/topics/public-datasets/bitcoin-in-bigquery-blockchain-analytics-on-public-data
---

The anomaly query pattern identifies transactions that appear in multiple blocks. Historically, in the Bitcoin blockchain, transactions could be duplicated due to a behavior in the original BerkeleyDB database engine that allowed non-unique keys. This was later addressed by implementing Bitcoin Improvement Proposal [BIP-0030](https://github.com/bitcoin/bips/blob/master/bip-0030.mediawiki) and transitioning to LevelDB.

### standardSQL
```sql
SELECT
  transaction_id,
  COUNT(transaction_id) AS dup_transaction_count
FROM (
  SELECT
    hash AS transaction_id
  FROM
    `bigquery-public-data.crypto_bitcoin.transactions`
)
GROUP BY
  transaction_id
HAVING
  dup_transaction_count > 1;
```
