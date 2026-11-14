---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/crypto_bitcoin/tables/inputs
title: Bitcoin Transaction Inputs
description: Bitcoin transaction inputs detailing UTXOs spent.
tags:
- bitcoin
- crypto
- blockchain
- utxo
- inputs
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:16:22+00:00'
sources:
- resource: https://github.com/blockchain-etl/bitcoin-etl
  title: Bitcoin ETL GitHub Repository
  id: bitcoin-etl
---

The `inputs` table contains details of all transaction inputs (UTXOs spent) on the Bitcoin blockchain. Each row represents a single input that was consumed to fund a transaction[^bitcoin-etl]. Because Bitcoin uses an Unspent Transaction Output (UTXO) model, every transaction consumes existing outputs (which become "inputs" in the new transaction) and creates new outputs[^bitcoin-etl].

This table is particularly useful for tracking the flow of funds, analyzing spending behavior, and tracing transaction lineage. By linking the `spent_transaction_hash` and `spent_output_index` of an input back to the [outputs](outputs.md) table, analysts can fully reconstruct the transaction graph.

Data is exported from the blockchain using the open-source [bitcoin-etl](https://github.com/blockchain-etl/bitcoin-etl) tool[^bitcoin-etl] and is housed in the [crypto_bitcoin](../datasets/crypto_bitcoin.md) dataset.

# Schema

| Field Name | Type | Mode | Description |
| :--- | :--- | :--- | :--- |
| **transaction_hash** | STRING | NULLABLE | Hash of the transaction containing this input |
| **block_hash** | STRING | NULLABLE | Hash of the block containing this transaction |
| **block_number** | INTEGER | NULLABLE | Height of the block containing this transaction |
| **block_timestamp** | TIMESTAMP | NULLABLE | Timestamp of the block containing this transaction |
| **index** | INTEGER | NULLABLE | 0-based index of this input within the transaction |
| **spent_transaction_hash** | STRING | NULLABLE | Hash of the transaction containing the output spent by this input |
| **spent_output_index** | INTEGER | NULLABLE | Index of the output spent by this input in the original transaction |
| **script_asm** | STRING | NULLABLE | Symbolic representation of the input's script (scriptSig) |
| **script_hex** | STRING | NULLABLE | Hexadecimal representation of the input's script (scriptSig) |
| **sequence** | INTEGER | NULLABLE | Transaction input sequence number |
| **required_signatures** | INTEGER | NULLABLE | Number of signatures required to spend (if applicable) |
| **type** | STRING | NULLABLE | Type of script (e.g., `witness_v1_taproot`, `pubkeyhash`) |
| **addresses** | STRING | REPEATED | List of addresses associated with this input |
| **value** | NUMERIC | NULLABLE | Value of the spent output in Satoshis |

# Common query patterns

### 1. Identify the largest transaction inputs in a given period
This query retrieves the largest inputs consumed on a specific day, demonstrating how to find massive UTXO consolidations or large-value transfers.

```sql
SELECT
  block_timestamp,
  transaction_hash,
  value / 100000000 AS value_btc,
  addresses
FROM `bigquery-public-data.crypto_bitcoin.inputs`
WHERE block_timestamp >= '2024-04-17 00:00:00 UTC'
  AND block_timestamp < '2024-04-18 00:00:00 UTC'
ORDER BY value DESC
LIMIT 10;
```

### 2. Track input types over time
Analyze the adoption of modern Bitcoin script types (like Taproot) by counting inputs grouped by their transaction script type.

```sql
SELECT
  DATE(block_timestamp) AS block_date,
  type,
  COUNT(1) AS input_count,
  SUM(value) / 100000000 AS total_value_btc
FROM `bigquery-public-data.crypto_bitcoin.inputs`
WHERE block_timestamp >= '2024-01-01 00:00:00 UTC'
GROUP BY block_date, type
ORDER BY block_date DESC, input_count DESC;
```

### 3. Trace provenance by joining inputs and outputs
To find where funds spent in a transaction came from, you can join the inputs table to the outputs table using the spent transaction keys.

```sql
SELECT
  inp.transaction_hash AS spending_tx,
  inp.block_timestamp AS spend_time,
  out.transaction_hash AS source_tx,
  out.block_timestamp AS source_time,
  inp.value / 100000000 AS value_btc
FROM `bigquery-public-data.crypto_bitcoin.inputs` AS inp
JOIN `bigquery-public-data.crypto_bitcoin.outputs` AS out
  ON inp.spent_transaction_hash = out.transaction_hash
  AND inp.spent_output_index = out.index
WHERE inp.block_timestamp >= '2024-04-17 00:00:00 UTC'
  AND inp.block_timestamp < '2024-04-17 01:00:00 UTC'
LIMIT 10;
```

# Joins

- [transactions](../references/joins/inputs___transactions.md) — Connects this spent input to the parent transaction record which spent it.

[^bitcoin-etl]: https://github.com/blockchain-etl/bitcoin-etl
