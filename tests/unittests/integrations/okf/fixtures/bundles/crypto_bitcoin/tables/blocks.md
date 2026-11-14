---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/crypto_bitcoin/tables/blocks
title: Bitcoin Blocks Table
description: All blocks from the Bitcoin blockchain, including block headers, transaction
  counts, sizes, and timestamps.
tags:
- bitcoin
- blockchain
- crypto
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:16:06+00:00'
sources:
- resource: https://github.com/blockchain-etl/bitcoin-etl
  title: Bitcoin ETL Export Tool
  id: bitcoin-etl
- id: bip-141
  resource: https://github.com/bitcoin/bips/blob/master/bip-0141.mediawiki
  title: BIP-141 Segregated Witness (Consensus layer)
---

The `blocks` table contains structured records for every block in the Bitcoin blockchain [^bitcoin-etl]. Each row in this table represents a single block, captured with detailed block header attributes such as hash, size, transaction count, nonce, difficulty bits, and the Merkle root of all transactions contained within that block.

The dataset is continually exported from live nodes and represents a complete historical index of Bitcoin blocks starting from the genesis block in January 2009. The table is partitioned by month using the `timestamp_month` column to optimize query performance and lower data scanning costs when filtering blocks by date.

The table can be joined with [transactions](transactions.md) to drill down into individual payments or to aggregate block-level statistics like total transaction fees, transaction densities, and witness data weights.

# Schema

| Field Name | Type | Mode | Description |
| :--- | :--- | :--- | :--- |
| **hash** | STRING | REQUIRED | Unique block hash that identifies the block. |
| **size** | INTEGER | NULLABLE | Total size of the block data in bytes. |
| **stripped_size** | INTEGER | NULLABLE | The size of block data in bytes excluding witness data. |
| **weight** | INTEGER | NULLABLE | Three times the base size plus the total size as defined in BIP-141 [^bip-141]. |
| **number** | INTEGER | REQUIRED | The sequential height of the block. |
| **version** | INTEGER | NULLABLE | Protocol version specified in the block header. |
| **merkle_root** | STRING | NULLABLE | The root node of a Merkle tree, where leaves are transaction hashes. |
| **timestamp** | TIMESTAMP | REQUIRED | Block creation timestamp specified in the block header. |
| **timestamp_month** | DATE | REQUIRED | Month of the block creation timestamp (used as the partitioning key). |
| **nonce** | STRING | NULLABLE | Difficulty solution specified in the block header. |
| **bits** | STRING | NULLABLE | Difficulty threshold specified in the block header. |
| **coinbase_param** | STRING | NULLABLE | Data specified in the coinbase transaction of this block. |
| **transaction_count** | INTEGER | NULLABLE | Number of transactions included in this block. |

# Common query patterns

### 1. Daily block counts and average transactions per block
Find out how many blocks are mined each day and the average number of transactions per block for a specific month.

```sql
SELECT
  DATE(timestamp) AS block_date,
  COUNT(1) AS blocks_mined,
  AVG(transaction_count) AS avg_transactions_per_block,
  SUM(transaction_count) AS total_transactions
FROM
  `bigquery-public-data.crypto_bitcoin.blocks`
WHERE
  timestamp_month = '2023-10-01'
GROUP BY
  block_date
ORDER BY
  block_date ASC;
```

### 2. Retrieve details for a specific block height
Lookup a single block's metadata and structure using its height number.

```sql
SELECT
  number,
  hash,
  timestamp,
  size,
  transaction_count,
  version,
  coinbase_param
FROM
  `bigquery-public-data.crypto_bitcoin.blocks`
WHERE
  number = 800000;
```

### 3. Calculate monthly average block size and weight
Analyze the adoption and impact of SegWit over time by analyzing the trends in block size, stripped size, and SegWit weight [^bip-141].

```sql
SELECT
  timestamp_month,
  COUNT(1) AS blocks_mined,
  AVG(size) AS avg_block_size_bytes,
  AVG(stripped_size) AS avg_stripped_size_bytes,
  AVG(weight) AS avg_weight
FROM
  `bigquery-public-data.crypto_bitcoin.blocks`
WHERE
  timestamp_month >= '2020-01-01'
GROUP BY
  timestamp_month
ORDER BY
  timestamp_month DESC;
```

# Joins

- [transactions](../references/joins/blocks___transactions.md) — Connects blocks to all included transactions to trace block validation times, miner fee revenue, or transaction densities.

[^bitcoin-etl]: https://github.com/blockchain-etl/bitcoin-etl
[^bip-141]: https://github.com/bitcoin/bips/blob/master/bip-0141.mediawiki
