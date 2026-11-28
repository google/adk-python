---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/crypto_bitcoin/tables/transactions
title: Bitcoin Transactions Table
description: All Bitcoin transactions containing inputs, outputs, block metadata,
  and fee structures.
tags:
- bitcoin
- crypto
- blockchain
- transactions
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:16:14+00:00'
sources:
- title: Bitcoin ETL Parser
  resource: https://github.com/blockchain-etl/bitcoin-etl
  id: bitcoin-etl
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/crypto_bitcoin/tables/transactions
  id: bq-metadata
  title: BigQuery transactions Table Schema
- resource: https://cloud.google.com/blog/topics/public-datasets/bitcoin-in-bigquery-blockchain-analytics-on-public-data
  id: gcp-blog
  title: 'Bitcoin in BigQuery: blockchain analytics on public data'
---

This table contains all Bitcoin transactions since the genesis block in January 2009. The data is exported from the Bitcoin blockchain using the open-source `bitcoin-etl` utility.[^bitcoin-etl] The grain of this table is one row per transaction.

Each transaction contains structural data such as its hash, size, coinbase indicator, block metadata (like block number, hash, and timestamp), fees, and nested records representing its spending inputs and resulting outputs. To perform cost-effective queries, the table is partitioned on the `block_timestamp_month` column.

This table links directly to several sibling tables in the [crypto_bitcoin](../datasets/crypto_bitcoin.md) dataset, such as [blocks](blocks.md). While inputs and outputs are nested as repeated records here, they are also flattened into dedicated sibling tables: [inputs](inputs.md) and [outputs](outputs.md).

# Schema

| Field Name | Type | Mode | Description |
|---|---|---|---|
| **hash** | STRING | REQUIRED | The unique SHA-256 hash of this transaction |
| **size** | INTEGER | NULLABLE | The size of this transaction in bytes |
| **virtual_size** | INTEGER | NULLABLE | The virtual transaction size (differs from size for SegWit/witness transactions) |
| **version** | INTEGER | NULLABLE | Protocol version specified in the block which contained this transaction |
| **lock_time** | INTEGER | NULLABLE | Earliest time/block height that miners can include the transaction |
| **block_hash** | STRING | REQUIRED | Hash of the block which contains this transaction |
| **block_number** | INTEGER | REQUIRED | Number of the block which contains this transaction |
| **block_timestamp** | TIMESTAMP | REQUIRED | Timestamp of the block which contains this transaction |
| **block_timestamp_month** | DATE | REQUIRED | Partitioning column; month of the block which contains this transaction |
| **input_count** | INTEGER | NULLABLE | The number of inputs in the transaction |
| **output_count** | INTEGER | NULLABLE | The number of outputs in the transaction |
| **input_value** | NUMERIC | NULLABLE | Total value of inputs in the transaction |
| **output_value** | NUMERIC | NULLABLE | Total value of outputs in the transaction |
| **is_coinbase** | BOOLEAN | NULLABLE | True if this transaction is a coinbase transaction (mined block reward) |
| **fee** | NUMERIC | NULLABLE | The transaction fee paid to miners (input_value - output_value) |
| **inputs** | RECORD | REPEATED | Nested array of transaction inputs |
| *inputs.***index** | INTEGER | REQUIRED | 0-indexed number of an input within a transaction |
| *inputs.***spent_transaction_hash** | STRING | NULLABLE | The hash of the transaction containing the output that this input spends |
| *inputs.***spent_output_index** | INTEGER | NULLABLE | The index of the output this input spends |
| *inputs.***script_asm** | STRING | NULLABLE | Symbolic representation of the script signature |
| *inputs.***script_hex** | STRING | NULLABLE | Hexadecimal representation of the script signature |
| *inputs.***sequence** | INTEGER | NULLABLE | Sequence number for locktime modifications |
| *inputs.***required_signatures** | INTEGER | NULLABLE | The number of signatures required to authorize the spent output |
| *inputs.***type** | STRING | NULLABLE | The address type of the spent output (e.g. pubkeyhash, scripthash) |
| *inputs.***addresses** | STRING | REPEATED | Array of addresses which own the spent output |
| *inputs.***value** | NUMERIC | NULLABLE | The value in base currency (satoshis) attached to the spent output |
| **outputs** | RECORD | REPEATED | Nested array of transaction outputs |
| *outputs.***index** | INTEGER | REQUIRED | 0-indexed number of the output used to reference this specific output later |
| *outputs.***script_asm** | STRING | NULLABLE | Symbolic representation of the script pubkey |
| *outputs.***script_hex** | STRING | NULLABLE | Hexadecimal representation of the script pubkey |
| *outputs.***required_signatures** | INTEGER | NULLABLE | The number of signatures required to authorize spending of this output |
| *outputs.***type** | STRING | NULLABLE | The address type of the output |
| *outputs.***addresses** | STRING | REPEATED | Array of addresses which own this output |
| *outputs.***value** | NUMERIC | NULLABLE | The value in base currency (satoshis) attached to this output |

# Common query patterns

### 1. Calculate average transaction fees and size over a month
The following query aggregates daily transaction volumes, average fees, and average sizes for a specific partitioned month.

```sql
SELECT
  DATE(block_timestamp) AS transaction_date,
  COUNT(1) AS transaction_count,
  AVG(fee) AS avg_fee_satoshis,
  AVG(size) AS avg_size_bytes
FROM
  `bigquery-public-data.crypto_bitcoin.transactions`
WHERE
  block_timestamp_month = '2023-10-01'
GROUP BY
  transaction_date
ORDER BY
  transaction_date;
```

### 2. Identify the highest-value transactions in a given month
This query retrieves the largest transactions by output value, excluding coinbase transactions (block rewards).

```sql
SELECT
  hash,
  block_number,
  block_timestamp,
  output_count,
  output_value
FROM
  `bigquery-public-data.crypto_bitcoin.transactions`
WHERE
  block_timestamp_month = '2023-10-01'
  AND is_coinbase = FALSE
ORDER BY
  output_value DESC
LIMIT 10;
```

### 3. Analyze output types and values (unnesting repeated records)
To analyze the distribution of different Bitcoin address types (such as `scripthash` or `witness_v0_keyhash`), you must unnest the `outputs` repeated record.

```sql
SELECT
  out.type AS address_type,
  COUNT(1) AS output_count,
  SUM(out.value) AS total_value
FROM
  `bigquery-public-data.crypto_bitcoin.transactions`,
  UNNEST(outputs) AS out
WHERE
  block_timestamp_month = '2023-10-01'
GROUP BY
  address_type
ORDER BY
  total_value DESC;
```

# Metrics

- [Duplicate transactions across blocks](../references/metrics/duplicate_transactions.md) — An anomaly detection query to spot old duplicate transaction IDs prior to the BIP-0030 implementation.

# Joins

- [blocks](../references/joins/blocks___transactions.md) — Links block metadata to find which block mined this transaction.
- [inputs](../references/joins/inputs___transactions.md) — Links a transaction to its UTXO spending sources.
- [outputs](../references/joins/outputs___transactions.md) — Links a transaction to its output receipts.

[^bitcoin-etl]: Blockchain ETL on GitHub: https://github.com/blockchain-etl/bitcoin-etl
