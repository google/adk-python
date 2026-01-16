# BigQuery Data Formats Reference

Complete guide to supported data formats for loading and external tables.

## Format Comparison

| Format | Compression | Schema | Nested Data | Best For |
|--------|-------------|--------|-------------|----------|
| Parquet | Excellent | Embedded | Full support | Analytics |
| Avro | Good | Embedded | Full support | Streaming, CDC |
| ORC | Excellent | Embedded | Full support | Hive migration |
| CSV | Moderate | Required | None | Simple flat data |
| JSON | Moderate | Optional | Full support | Semi-structured |

## Parquet

### Overview

Apache Parquet is a columnar storage format optimized for analytics workloads.

**Pros:**
- Best query performance (columnar)
- Excellent compression
- Schema embedded in file
- Predicate pushdown support

**Cons:**
- Not human-readable
- Write overhead

### Load Options

```sql
LOAD DATA INTO `project.dataset.table`
FROM FILES (
  format = 'PARQUET',
  uris = ['gs://bucket/data/*.parquet'],
  -- Optional: read specific columns only
  projection_fields = ['column1', 'column2', 'column3'],
  -- Optional: for BigLake tables
  metadata_cache_mode = 'AUTOMATIC'
);
```

### Supported Compression

- Snappy (default, recommended)
- GZIP
- LZO
- BROTLI
- LZ4
- ZSTD

### Type Mapping

| Parquet Type | BigQuery Type |
|--------------|---------------|
| BOOLEAN | BOOL |
| INT32/INT64 | INT64 |
| FLOAT/DOUBLE | FLOAT64 |
| BYTE_ARRAY (UTF8) | STRING |
| BYTE_ARRAY | BYTES |
| INT96 | TIMESTAMP |
| DATE | DATE |
| DECIMAL | NUMERIC/BIGNUMERIC |
| LIST | ARRAY |
| MAP | STRUCT (key/value) |
| STRUCT | STRUCT |

## Avro

### Overview

Apache Avro is a row-based format with strong schema support.

**Pros:**
- Schema evolution support
- Compact binary format
- Good for streaming
- Self-describing

**Cons:**
- Less efficient for analytics
- Larger than Parquet for analytics

### Load Options

```sql
LOAD DATA INTO `project.dataset.table`
FROM FILES (
  format = 'AVRO',
  uris = ['gs://bucket/data/*.avro'],
  -- Use Avro logical types
  use_avro_logical_types = TRUE,
  -- Enable schema inference for missing tables
  enable_list_inference = TRUE
);
```

### Type Mapping

| Avro Type | BigQuery Type |
|-----------|---------------|
| boolean | BOOL |
| int | INT64 |
| long | INT64 |
| float | FLOAT64 |
| double | FLOAT64 |
| bytes | BYTES |
| string | STRING |
| record | STRUCT |
| array | ARRAY |
| map | ARRAY<STRUCT<key, value>> |
| enum | STRING |
| fixed | BYTES |
| union | Nullable type |

### Logical Types

| Avro Logical Type | BigQuery Type |
|-------------------|---------------|
| date | DATE |
| time-millis | TIME |
| time-micros | TIME |
| timestamp-millis | TIMESTAMP |
| timestamp-micros | TIMESTAMP |
| decimal | NUMERIC/BIGNUMERIC |

## ORC

### Overview

Optimized Row Columnar format from Hive ecosystem.

**Pros:**
- Excellent compression
- Good Hive compatibility
- Predicate pushdown

**Cons:**
- Less common outside Hive
- Limited tooling

### Load Options

```sql
LOAD DATA INTO `project.dataset.table`
FROM FILES (
  format = 'ORC',
  uris = ['gs://bucket/data/*.orc']
);
```

### Supported Compression

- ZLIB (default)
- Snappy
- LZO
- LZ4
- ZSTD

## CSV

### Overview

Comma-separated values, the most universal format.

**Pros:**
- Human-readable
- Universal compatibility
- Easy to generate

**Cons:**
- No schema
- Poor compression
- Type inference needed
- No nested data

### Load Options

```sql
LOAD DATA INTO `project.dataset.table`
FROM FILES (
  format = 'CSV',
  uris = ['gs://bucket/data/*.csv'],
  -- Schema handling
  skip_leading_rows = 1,
  autodetect = TRUE,  -- or provide explicit schema
  -- Delimiters
  field_delimiter = ',',
  -- Quoting
  quote = '"',
  allow_quoted_newlines = TRUE,
  -- Null handling
  null_marker = '',
  -- Error handling
  allow_jagged_rows = FALSE,
  max_bad_records = 0,
  -- Encoding
  encoding = 'UTF-8',
  -- Compression
  compression = 'GZIP'  -- if files are compressed
);
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Wrong delimiter | Set `field_delimiter` |
| Quotes in values | Set `quote` and `allow_quoted_newlines` |
| Header row | Set `skip_leading_rows = 1` |
| Empty values | Set `null_marker` |
| Encoding errors | Set `encoding = 'UTF-8'` |
| Extra columns | Set `allow_jagged_rows = TRUE` |

### Type Inference

When using `autodetect = TRUE`:
- Numbers → INT64 or FLOAT64
- Dates → DATE (if recognized)
- Everything else → STRING

Recommendation: Provide explicit schema for production.

## JSON / NEWLINE_DELIMITED_JSON

### Overview

JSON format for semi-structured data.

**Pros:**
- Human-readable
- Flexible schema
- Nested data support

**Cons:**
- Verbose
- Poor compression
- Parsing overhead

### Load Options

```sql
LOAD DATA INTO `project.dataset.table`
FROM FILES (
  format = 'NEWLINE_DELIMITED_JSON',  -- One JSON object per line
  uris = ['gs://bucket/data/*.jsonl'],
  -- Schema handling
  autodetect = TRUE,
  -- Error handling
  max_bad_records = 10,
  ignore_unknown_values = TRUE,
  -- Encoding
  encoding = 'UTF-8'
);
```

### JSON Array Format

```sql
-- For JSON arrays (not newline-delimited)
LOAD DATA INTO `project.dataset.table`
FROM FILES (
  format = 'JSON',
  uris = ['gs://bucket/data/*.json'],
  json_extension = 'GEOJSON'  -- For GeoJSON files
);
```

### Nested Data

```json
// Input JSON
{
  "user_id": "123",
  "profile": {
    "name": "John",
    "age": 30
  },
  "events": [
    {"type": "click", "timestamp": "2024-01-15T10:00:00Z"},
    {"type": "view", "timestamp": "2024-01-15T10:01:00Z"}
  ]
}
```

```sql
-- Resulting schema
CREATE TABLE example (
  user_id STRING,
  profile STRUCT<name STRING, age INT64>,
  events ARRAY<STRUCT<type STRING, timestamp TIMESTAMP>>
);
```

## Google Sheets

### External Table from Sheets

```sql
CREATE EXTERNAL TABLE `project.dataset.sheet_data`
OPTIONS (
  format = 'GOOGLE_SHEETS',
  uris = ['https://docs.google.com/spreadsheets/d/SHEET_ID/edit'],
  skip_leading_rows = 1,
  range = 'Sheet1!A1:Z1000'
);
```

### Limitations

- Maximum 100,000 rows
- Read-only
- Performance varies
- Authentication required

## Compression

### Supported Compression by Format

| Format | GZIP | SNAPPY | LZ4 | ZSTD | BROTLI |
|--------|------|--------|-----|------|--------|
| Parquet | Yes | Yes (default) | Yes | Yes | Yes |
| Avro | Yes | Yes (default) | - | - | - |
| ORC | Yes | Yes | Yes | Yes | - |
| CSV | Yes | - | - | - | - |
| JSON | Yes | - | - | - | - |

### Compression Recommendations

1. **Parquet**: Use Snappy for speed, ZSTD for size
2. **Avro**: Use Snappy (default)
3. **CSV/JSON**: Use GZIP for storage, uncompressed for speed
4. **ORC**: Use ZLIB for size, Snappy for speed

## Best Practices

### Format Selection

1. **Analytics workloads**: Parquet
2. **Streaming/CDC**: Avro
3. **Hive migration**: ORC
4. **Quick exports**: CSV
5. **APIs/events**: JSON

### Schema Management

1. **Provide explicit schemas** for production loads
2. **Use schema files** for version control
3. **Test schema changes** before deployment
4. **Document transformations** for auditing

### Performance Tips

1. **File size**: Target 100MB-1GB per file
2. **Avoid many small files**: Combine before loading
3. **Use columnar formats**: Parquet/ORC for analytics
4. **Enable predicate pushdown**: Filter at source
