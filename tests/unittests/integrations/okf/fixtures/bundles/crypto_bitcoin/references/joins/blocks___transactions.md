---
type: Reference
resource: https://github.com/blockchain-etl/bitcoin-etl
title: Blocks to Transactions Join Path
description: Join relationship linking blocks to their corresponding transactions
  by block height / block number.
tags:
- join
- bitcoin
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:15:51+00:00'
sources:
- id: bitcoin-etl
  resource: https://github.com/blockchain-etl/bitcoin-etl
  title: Bitcoin ETL Parser
---

This join path represents the link between a block and all the transactions included in that block. This is useful for analyzing block density, mining fee shares, and validating transaction confirmation times relative to block production.

```sql
SELECT
  b.number AS block_height,
  b.hash AS block_hash,
  b.timestamp AS block_timestamp,
  t.hash AS transaction_hash,
  t.fee AS transaction_fee
FROM
  `bigquery-public-data.crypto_bitcoin.blocks` AS b
JOIN
  `bigquery-public-data.crypto_bitcoin.transactions` AS t
ON
  b.number = t.block_number;
```
