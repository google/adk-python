---
type: Reference
resource: https://github.com/blockchain-etl/bitcoin-etl
title: Transactions to Inputs Join Path
description: Join path between transactions and inputs to trace fund consumption details.
tags:
- join
- bitcoin
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:15:56+00:00'
sources:
- id: bitcoin-etl
  title: Bitcoin ETL Parser
  resource: https://github.com/blockchain-etl/bitcoin-etl
---

This join path connects transactions to their inputs. In the Unspent Transaction Output (UTXO) database structure, joining the main `transactions` table with the flat `inputs` table lets analysts audit the historical origin of funds being consumed in a transaction.

```sql
SELECT
  t.hash AS transaction_hash,
  t.block_timestamp AS transaction_timestamp,
  i.index AS input_index,
  i.spent_transaction_hash,
  i.spent_output_index,
  i.value AS input_value_satoshis
FROM
  `bigquery-public-data.crypto_bitcoin.transactions` AS t
JOIN
  `bigquery-public-data.crypto_bitcoin.inputs` AS i
ON
  t.hash = i.transaction_hash;
```
