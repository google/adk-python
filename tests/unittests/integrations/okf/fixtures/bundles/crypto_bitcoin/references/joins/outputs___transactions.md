---
type: Reference
resource: https://github.com/blockchain-etl/bitcoin-etl
title: Transactions to Outputs Join Path
description: Join path between transactions and outputs to audit target recipient
  distribution.
tags:
- join
- bitcoin
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:15:59+00:00'
sources:
- title: Bitcoin ETL Parser
  resource: https://github.com/blockchain-etl/bitcoin-etl
  id: bitcoin-etl
---

This join path relates a transaction to its generated outputs. Joining `transactions` with `outputs` is useful for tracking how funds are distributed (split or forwarded) from a parent transaction into target addresses.

```sql
SELECT
  t.hash AS transaction_hash,
  t.block_timestamp AS transaction_timestamp,
  o.index AS output_index,
  o.addresses,
  o.value AS output_value_satoshis
FROM
  `bigquery-public-data.crypto_bitcoin.transactions` AS t
JOIN
  `bigquery-public-data.crypto_bitcoin.outputs` AS o
ON
  t.hash = o.transaction_hash;
```
