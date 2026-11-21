---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/ga4_obfuscated_sample_ecommerce/tables/events_*
title: GA4 Events Export
description: Google Analytics 4 event-level daily sharded export tables containing
  user interaction logs.
tags:
- analytics
- e-commerce
- ga4
- sharded-tables
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:15:20+00:00'
sources:
- title: 'Google Analytics Help: BigQuery Export Schema'
  id: ga4-export-docs
  resource: https://support.google.com/analytics/answer/7029846
- title: BigQuery Table Metadata
  id: metadata
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/ga4_obfuscated_sample_ecommerce/tables/events_*
- title: Sample queries for audiences based on BigQuery data - Analytics Help
  resource: https://support.google.com/analytics/answer/9037342
  id: sample_queries
---

The `events_` table family contains obfuscated Google Analytics 4 (GA4) event-level export data from the Google Merchandise Store.[^ga4-export-docs] It is structured as a series of daily sharded tables, starting from `events_20201101` through `events_20210131`.[^metadata] Each row represents a single event (e.g., `page_view`, `scroll`, `session_start`, `view_item`, `purchase`) triggered by a user's interaction with the online storefront.

The data is useful for behavioral analysis, funnel conversion mapping, and e-commerce tracking. Key user attributes such as geographic location, device platform, and acquisition traffic source are nested within top-level records. Additionally, parameters associated with specific events or products are stored in repeated record fields (`event_params` and `items`), requiring flatting or unnesting operations during querying.

# Schema

Below is the flattened schema representation for the sharded daily events table.

| Field Name | Type | Mode | Description |
| :--- | :--- | :--- | :--- |
| **event_date** | STRING | NULLABLE | The date on which the event was logged (formatted as `YYYYMMDD`). |
| **event_timestamp** | INTEGER | NULLABLE | The POSIX timestamp (in microseconds) when the event was registered. |
| **event_name** | STRING | NULLABLE | The name of the event (e.g., `page_view`, `purchase`, `session_start`). |
| **event_params** | RECORD | REPEATED | Key-value parameters associated with the event. |
| *event_params.key* | STRING | NULLABLE | The name of the parameter. |
| *event_params.value* | RECORD | NULLABLE | The value of the parameter, nested by data type. |
| *event_params.value.string_value* | STRING | NULLABLE | Parameter value if it is a string. |
| *event_params.value.int_value* | INTEGER | NULLABLE | Parameter value if it is an integer. |
| *event_params.value.float_value* | FLOAT | NULLABLE | Parameter value if it is a float. |
| *event_params.value.double_value* | FLOAT | NULLABLE | Parameter value if it is a double. |
| **event_previous_timestamp** | INTEGER | NULLABLE | The timestamp of the previous event (in microseconds). |
| **event_value_in_usd** | FLOAT | NULLABLE | The monetary value of the event, converted to USD. |
| **event_bundle_sequence_id** | INTEGER | NULLABLE | The sequence ID of the upload bundle. |
| **event_server_timestamp_offset** | INTEGER | NULLABLE | Difference between server time and device logging time. |
| **user_id** | STRING | NULLABLE | The unique identifier of the user (when signed in). |
| **user_pseudo_id** | STRING | NULLABLE | The pseudonymous consumer device identifier (e.g. GA client ID). |
| **privacy_info** | RECORD | NULLABLE | Consent and privacy settings. |
| *privacy_info.analytics_storage* | INTEGER | NULLABLE | Status of analytics storage consent. |
| *privacy_info.ads_storage* | INTEGER | NULLABLE | Status of ads storage consent. |
| *privacy_info.uses_transient_token* | STRING | NULLABLE | Whether a transient token is used. |
| **user_properties** | RECORD | REPEATED | Custom user properties. |
| *user_properties.key* | INTEGER | NULLABLE | Property name/key. |
| *user_properties.value* | RECORD | NULLABLE | Nested custom value and update timestamp. |
| **user_first_touch_timestamp** | INTEGER | NULLABLE | The time (in microseconds) when the user first interacted with the site. |
| **user_ltv** | RECORD | NULLABLE | User lifetime value details. |
| *user_ltv.revenue* | FLOAT | NULLABLE | Total revenue attributed to the user over time. |
| *user_ltv.currency* | STRING | NULLABLE | The currency of the lifetime revenue value. |
| **device** | RECORD | NULLABLE | Device information of the visitor. |
| *device.category* | STRING | NULLABLE | Device category (e.g., `mobile`, `desktop`, `tablet`). |
| *device.mobile_brand_name* | STRING | NULLABLE | Mobile device brand name (e.g., `Apple`, `Samsung`). |
| *device.mobile_model_name* | STRING | NULLABLE | Mobile model name. |
| *device.mobile_marketing_name* | STRING | NULLABLE | Device marketing name. |
| *device.operating_system* | STRING | NULLABLE | Operating system name (e.g., `iOS`, `Android`, `Web`). |
| *device.operating_system_version* | STRING | NULLABLE | OS version. |
| *device.language* | STRING | NULLABLE | Browser/device language code. |
| *device.web_info.browser* | STRING | NULLABLE | Web browser name. |
| *device.web_info.browser_version* | STRING | NULLABLE | Web browser version. |
| **geo** | RECORD | NULLABLE | Geographical information derived from IP addresses. |
| *geo.continent* | STRING | NULLABLE | Continent name. |
| *geo.sub_continent* | STRING | NULLABLE | Sub-continent name. |
| *geo.country* | STRING | NULLABLE | Country name. |
| *geo.region* | STRING | NULLABLE | Region or state name. |
| *geo.city* | STRING | NULLABLE | City name. |
| *geo.metro* | STRING | NULLABLE | Metro area name. |
| **app_info** | RECORD | NULLABLE | Application specific information. |
| **traffic_source** | RECORD | NULLABLE | User acquisition source. |
| *traffic_source.medium* | STRING | NULLABLE | The medium (e.g., `organic`, `referral`, `cpc`). |
| *traffic_source.name* | STRING | NULLABLE | The campaign name. |
| *traffic_source.source* | STRING | NULLABLE | The source (e.g., `google`, `direct`). |
| **stream_id** | INTEGER | NULLABLE | Data stream ID. |
| **platform** | STRING | NULLABLE | The collection platform (e.g., `WEB`, `IOS`, `ANDROID`). |
| **event_dimensions** | RECORD | NULLABLE | Event-level metadata dimensions. |
| *event_dimensions.hostname* | STRING | NULLABLE | Target hostname where the event occurred. |
| **ecommerce** | RECORD | NULLABLE | Order level transaction details. |
| *ecommerce.total_item_quantity* | INTEGER | NULLABLE | Total items in the transaction. |
| *ecommerce.purchase_revenue_in_usd* | FLOAT | NULLABLE | Revenue of the transaction converted to USD. |
| *ecommerce.transaction_id* | STRING | NULLABLE | Transaction identifier. |
| **items** | RECORD | REPEATED | Product-level attributes for the items involved in the event. |
| *items.item_id* | STRING | NULLABLE | Product ID or SKU. |
| *items.item_name* | STRING | NULLABLE | Name of the product. |
| *items.item_brand* | STRING | NULLABLE | Brand of the product. |
| *items.price_in_usd* | FLOAT | NULLABLE | Unit price in USD. |
| *items.quantity* | INTEGER | NULLABLE | Quantity of items. |

# Common query patterns

### 1. Count events and active users by event name
This query counts the total events logged and counts distinct users (`user_pseudo_id`) for each event type over the full range of tables.

```sql
SELECT
  event_name,
  COUNT(1) AS event_count,
  COUNT(DISTINCT user_pseudo_id) AS unique_users
FROM
  `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
GROUP BY
  1
ORDER BY
  event_count DESC;
```

### 2. Extract nested page_location from event_params
Since `event_params` is a repeated record (ARRAY), you must unnest it or filter using a subquery to extract a specific parameter like `page_location` for page views.

```sql
SELECT
  event_date,
  (SELECT value.string_value FROM UNNEST(event_params) WHERE key = 'page_location') AS page_path,
  COUNT(1) AS page_views
FROM
  `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE
  event_name = 'page_view'
  AND _TABLE_SUFFIX BETWEEN '20210101' AND '20210115'
GROUP BY
  1, 2
ORDER BY
  page_views DESC;
```

### 3. Compute top selling products from items array
To analyze product sales, we unnest the repeated `items` record structure on purchase events and aggregate quantities.

```sql
SELECT
  item.item_id,
  item.item_name,
  SUM(item.quantity) AS units_sold,
  ROUND(SUM(item.item_revenue_in_usd), 2) AS total_revenue_usd
FROM
  `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`,
  UNNEST(items) AS item
WHERE
  event_name = 'purchase'
  AND _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
GROUP BY
  1, 2
ORDER BY
  total_revenue_usd DESC
LIMIT 10;
```

# Metrics
The following predefined and custom audience cohort metrics can be derived from the events log table:
* [Purchasers](../references/metrics/purchasers.md) — Users who have logged either `in_app_purchase` or `purchase`.
* [N-Day Active Users](../references/metrics/n_day_active_users.md) — Users who have logged at least one event with `engagement_time_msec > 0` in the last N days.
* [N-Day Inactive Users](../references/metrics/n_day_inactive_users.md) — Active users from the last M days who have not logged any event with `engagement_time_msec > 0` in the last N days (M > N).
* [Frequently Active Users](../references/metrics/frequently_active_users.md) — Users who have logged at least one event with `engagement_time_msec > 0` on N of the last M days.
* [Highly Active Users](../references/metrics/highly_active_users.md) — Users who have been active/engaged for more than N minutes in the last M days.
* [Acquired Users](../references/metrics/acquired_users.md) — Users acquired via a specific campaign source, medium, and name.
* [Google Acquired Cohorts](../references/metrics/google_acquired_cohorts.md) — Users acquired in a specific weekly cohort filtered by Google campaign source.

[^ga4-export-docs]: [Google Analytics Help: BigQuery Export Schema](https://support.google.com/analytics/answer/7029846)
[^metadata]: Source dataset `ga4_obfuscated_sample_ecommerce` table list metadata.
