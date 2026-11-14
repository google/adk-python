---
type: BigQuery Dataset
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/ga4_obfuscated_sample_ecommerce
title: GA4 Obfuscated Sample Ecommerce Dataset
description: Obfuscated Google Analytics 4 dataset emulating a web ecommerce implementation
  of the Google Merchandise Store.
tags:
- ga4
- ecommerce
- obfuscated
- analytics
- sample-data
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:14:56+00:00'
sources:
- title: BigQuery Dataset Metadata for ga4_obfuscated_sample_ecommerce
  id: ga4-metadata
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/ga4_obfuscated_sample_ecommerce
- resource: https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset
  title: Google Analytics 4 eCommerce Demo Dataset Documentation
  id: ga4-demo-docs
---

The `ga4_obfuscated_sample_ecommerce` dataset is an obfuscated, publicly accessible export of Google Analytics 4 (GA4) event data representing a real-world web ecommerce implementation (specifically, from the Google Merchandise Store)[^ga4-demo-docs]. It spans three months of historical activity from November 1, 2020, to January 1, 2021[^ga4-demo-docs], and is designed to allow developers, analysts, and students to experiment with high-volume, granular GA4 event data in BigQuery without provisioning a proprietary dataset.

This dataset contains a single sharded table family, [events_](../tables/events_.md), which holds daily export tables containing individual session interactions, user properties, and ecommerce transaction details. Analysts can leverage this dataset to learn how to query GA4 nested schemas, build user acquisition models, reconstruct user journeys, and analyze purchase funnels.

# Schema

As a BigQuery Dataset, this resource acts as a namespace containing tables and does not have a flat column schema of its own. It hosts the following tables:

*   [events_](../tables/events_.md): A partitioned, sharded table containing daily Google Analytics 4 event export records.

# Common query patterns

### 1. Count total events and distinct users across the entire dataset

This query demonstrates how to query over all sharded tables in the dataset using a wildcard suffix pattern.

```sql
SELECT
  COUNT(*) AS total_events,
  COUNT(DISTINCT user_pseudo_id) AS total_users
FROM
  `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
```

### 2. Locate tables and confirm data availability

This query retrieves metadata about the individual tables contained within the dataset namespace.

```sql
SELECT
  table_id,
  creation_time,
  row_count,
  size_bytes
FROM
  `bigquery-public-data.ga4_obfuscated_sample_ecommerce.__TABLES__`
ORDER BY
  table_id DESC
```

[^ga4-demo-docs]: [Google Analytics 4 eCommerce Demo Dataset documentation](https://developers.google.com/analytics/bigquery/web-ecommerce-demo-dataset)
