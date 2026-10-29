---
type: BigQuery Dataset
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/crypto_bitcoin
title: Bitcoin Blockchain Dataset
description: A public Google BigQuery dataset containing the complete transaction
  ledger and block history of the Bitcoin blockchain.
tags:
- bitcoin
- blockchain
- crypto
- public-data
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:14:11+00:00'
sources:
- title: BigQuery Dataset Metadata - crypto_bitcoin
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/crypto_bitcoin
  id: bq-crypto-bitcoin-meta
---

The `crypto_bitcoin` dataset is a public Google BigQuery dataset containing the entire blockchain transaction history for Bitcoin. It is updated continuously and provides a highly structured, queryable format of block and transaction data from the genesis block onwards.

The dataset contains four primary tables:
- [blocks](../tables/blocks.md) representing Bitcoin blocks, including hashes, sizes, transaction counts, and block rewards.
- [transactions](../tables/transactions.md) containing top-level transaction details such as inputs/outputs totals, fees, and cryptographic signatures.
- [inputs](../tables/inputs.md) containing the transaction inputs (spending previous outputs) representing the source of funds.
- [outputs](../tables/outputs.md) containing the transaction outputs representing the destinations of funds (addresses and values).

This dataset is widely used for blockchain forensics, macroeconomic analysis of transaction volumes, wallet balance tracking, and research into mining activities.

# Schema

As a BigQuery Dataset, `crypto_bitcoin` acts as a namespace and container for the following tables:

| Table ID | Description |
| :--- | :--- |
| **[blocks](../tables/blocks.md)** | Blocks containing transactions that have been validated and written to the ledger. |
| **[transactions](../tables/transactions.md)** | Individual ledger entries where value is transferred between participants. |
| **[inputs](../tables/inputs.md)** | References to UTXOs (Unspent Transaction Outputs) being spent in transactions. |
| **[outputs](../tables/outputs.md)** | Outputs created by transactions that become new UTXOs. |

# Common query patterns

### 1. Count of blocks and average transaction count per block by month
This query calculates the monthly volume of blocks and the average number of transactions included per block.

```sql
SELECT
  TIMESTAMP_TRUNC(timestamp, MONTH) AS month,
  COUNT(1) AS total_blocks,
  AVG(transaction_count) AS avg_transactions_per_block
FROM
  `bigquery-public-data.crypto_bitcoin.blocks`
GROUP BY
  month
ORDER BY
  month DESC
LIMIT 12;
```

### 2. Transaction fee statistics (in Satoshis) over the last 30 days
This query explores transaction fee distributions across recent transactions.

```sql
SELECT
  MIN(fee) AS min_fee,
  MAX(fee) AS max_fee,
  AVG(fee) AS avg_fee,
  APPROX_QUANTILES(fee, 2)[OFFSET(1)] AS median_fee
FROM
  `bigquery-public-data.crypto_bitcoin.transactions`
WHERE
  block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);
```
