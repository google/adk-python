---
name: bigquery-integration
description: Integrate BigQuery with external systems - client libraries, REST APIs, JDBC/ODBC drivers, Data Transfer Service, Dataflow, and third-party tools. Use when connecting BigQuery to applications, pipelines, or BI tools.
license: Apache-2.0
compatibility: BigQuery, Python, Java, Node.js, JDBC, ODBC
metadata:
  author: Google Cloud
  version: "1.0"
  category: integration
adk:
  config:
    timeout_seconds: 300
    max_parallel_calls: 10
  allowed_callers:
    - bigquery_agent
    - integration_agent
    - developer_agent
---

# BigQuery Integration Skill

Integrate BigQuery with external systems including client libraries, REST APIs, JDBC/ODBC drivers, Data Transfer Service, and third-party tools.

## When to Use This Skill

Use this skill when you need to:
- Connect applications to BigQuery using client libraries
- Set up JDBC/ODBC connections for BI tools
- Configure Data Transfer Service for data ingestion
- Integrate with Dataflow for streaming
- Connect BigQuery to Looker, Tableau, or other tools
- Use the REST API directly

## Integration Options

| Method | Use Case | Best For |
|--------|----------|----------|
| **Python Client** | Data science, ETL | Python applications |
| **Java Client** | Enterprise apps | Java/JVM applications |
| **Node.js Client** | Web apps, APIs | JavaScript applications |
| **REST API** | Any language | Custom integrations |
| **JDBC/ODBC** | BI tools | Tableau, Power BI |
| **Data Transfer** | Scheduled imports | External data sources |

## Quick Start

### Python Client

```python
from google.cloud import bigquery

# Initialize client
client = bigquery.Client(project='my-project')

# Run query
query = "SELECT * FROM `project.dataset.table` LIMIT 100"
df = client.query(query).to_dataframe()

# Load data
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
)
client.load_table_from_uri(
    "gs://bucket/data.csv",
    "project.dataset.table",
    job_config=job_config
).result()
```

### REST API

```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  "https://bigquery.googleapis.com/bigquery/v2/projects/PROJECT/queries" \
  -d '{
    "query": "SELECT * FROM `project.dataset.table` LIMIT 10",
    "useLegacySql": false
  }'
```

## Python Client Library

### Installation

```bash
pip install google-cloud-bigquery
pip install google-cloud-bigquery-storage  # For faster reads
pip install pandas  # For DataFrame support
pip install pyarrow  # For Arrow optimization
```

### Query Execution

```python
from google.cloud import bigquery

client = bigquery.Client()

# Simple query
query = """
    SELECT name, SUM(amount) as total
    FROM `project.dataset.sales`
    GROUP BY name
    ORDER BY total DESC
    LIMIT 10
"""

# Execute and get results
results = client.query(query).result()
for row in results:
    print(f"{row.name}: {row.total}")

# To DataFrame
df = client.query(query).to_dataframe()

# With parameters
query = """
    SELECT * FROM `project.dataset.orders`
    WHERE created_at > @start_date
    AND status = @status
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("start_date", "DATE", "2024-01-01"),
        bigquery.ScalarQueryParameter("status", "STRING", "completed"),
    ]
)
results = client.query(query, job_config=job_config).result()
```

### Loading Data

```python
# From local file
with open("data.csv", "rb") as f:
    job = client.load_table_from_file(
        f,
        "project.dataset.table",
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
        )
    )
    job.result()

# From GCS
job = client.load_table_from_uri(
    "gs://bucket/data/*.parquet",
    "project.dataset.table",
    job_config=bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
)
job.result()

# From DataFrame
import pandas as pd

df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
job = client.load_table_from_dataframe(df, "project.dataset.table")
job.result()
```

### Streaming Insert

```python
# Insert rows immediately (legacy streaming)
rows = [
    {"user_id": "123", "event": "click", "timestamp": "2024-01-15T10:00:00"},
    {"user_id": "456", "event": "view", "timestamp": "2024-01-15T10:00:01"},
]

errors = client.insert_rows_json("project.dataset.events", rows)
if errors:
    print(f"Errors: {errors}")
```

### Table Management

```python
# Create table
schema = [
    bigquery.SchemaField("id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
]

table = bigquery.Table("project.dataset.new_table", schema=schema)
table.time_partitioning = bigquery.TimePartitioning(field="created_at")
table = client.create_table(table)

# Get table info
table = client.get_table("project.dataset.table")
print(f"Rows: {table.num_rows}, Size: {table.num_bytes}")

# Delete table
client.delete_table("project.dataset.table", not_found_ok=True)
```

### BigQuery Storage API (Faster Reads)

```python
from google.cloud import bigquery_storage

# Read directly (faster for large results)
bqstorage_client = bigquery_storage.BigQueryReadClient()

df = client.query(query).to_dataframe(
    bqstorage_client=bqstorage_client
)
```

## Java Client Library

### Maven Dependency

```xml
<dependency>
  <groupId>com.google.cloud</groupId>
  <artifactId>google-cloud-bigquery</artifactId>
  <version>2.34.0</version>
</dependency>
```

### Query Execution

```java
import com.google.cloud.bigquery.*;

BigQuery bigquery = BigQueryOptions.getDefaultInstance().getService();

String query = "SELECT * FROM `project.dataset.table` LIMIT 100";
QueryJobConfiguration queryConfig = QueryJobConfiguration.newBuilder(query).build();

TableResult results = bigquery.query(queryConfig);
for (FieldValueList row : results.iterateAll()) {
    String name = row.get("name").getStringValue();
    long amount = row.get("amount").getLongValue();
    System.out.println(name + ": " + amount);
}
```

### Loading Data

```java
// From GCS
LoadJobConfiguration loadConfig = LoadJobConfiguration.newBuilder(
    TableId.of("dataset", "table"),
    "gs://bucket/data/*.csv"
)
    .setFormatOptions(CsvOptions.newBuilder().setSkipLeadingRows(1).build())
    .setAutodetect(true)
    .build();

Job job = bigquery.create(JobInfo.of(loadConfig));
job.waitFor();
```

## Node.js Client Library

### Installation

```bash
npm install @google-cloud/bigquery
```

### Query Execution

```javascript
const {BigQuery} = require('@google-cloud/bigquery');

const bigquery = new BigQuery();

async function runQuery() {
  const query = `
    SELECT name, SUM(amount) as total
    FROM \`project.dataset.sales\`
    GROUP BY name
    LIMIT 10
  `;

  const [rows] = await bigquery.query({query});
  rows.forEach(row => console.log(`${row.name}: ${row.total}`));
}

runQuery();
```

### Loading Data

```javascript
async function loadFromGCS() {
  const [job] = await bigquery
    .dataset('dataset')
    .table('table')
    .load('gs://bucket/data.csv', {
      sourceFormat: 'CSV',
      skipLeadingRows: 1,
      autodetect: true,
    });

  console.log(`Job ${job.id} completed.`);
}
```

## REST API

### Authentication

```bash
# Get access token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Or use service account
gcloud auth activate-service-account --key-file=key.json
```

### Query

```bash
curl -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  "https://bigquery.googleapis.com/bigquery/v2/projects/PROJECT/queries" \
  -d '{
    "query": "SELECT * FROM `project.dataset.table` LIMIT 10",
    "useLegacySql": false,
    "maxResults": 100
  }'
```

### Get Query Results

```bash
curl -X GET \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://bigquery.googleapis.com/bigquery/v2/projects/PROJECT/queries/JOB_ID"
```

### List Tables

```bash
curl -X GET \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://bigquery.googleapis.com/bigquery/v2/projects/PROJECT/datasets/DATASET/tables"
```

## JDBC/ODBC Drivers

### JDBC Connection

```java
// JDBC URL format
String url = "jdbc:bigquery://https://www.googleapis.com/bigquery/v2:443;"
    + "ProjectId=PROJECT_ID;"
    + "OAuthType=0;"
    + "OAuthServiceAcctEmail=service@project.iam.gserviceaccount.com;"
    + "OAuthPvtKeyPath=/path/to/key.json;";

Connection conn = DriverManager.getConnection(url);
Statement stmt = conn.createStatement();
ResultSet rs = stmt.executeQuery("SELECT * FROM dataset.table");
```

### ODBC Connection String

```
Driver={Simba ODBC Driver for Google BigQuery};
Catalog=PROJECT_ID;
OAuthMechanism=0;
Email=service@project.iam.gserviceaccount.com;
KeyFilePath=/path/to/key.json;
```

### BI Tool Connections

**Tableau:**
1. Use "Google BigQuery" connector
2. Select authentication method
3. Choose project and dataset

**Power BI:**
1. Get Data > Google BigQuery
2. Sign in with Google account
3. Select tables/views

**Looker:**
1. Admin > Connections
2. New Connection > BigQuery
3. Configure project and authentication

## Data Transfer Service

### Supported Sources

| Source | Description |
|--------|-------------|
| Google Ads | Marketing data |
| Campaign Manager | Advertising data |
| Google Play | App analytics |
| YouTube | Channel analytics |
| Cloud Storage | File imports |
| Amazon S3 | Cross-cloud transfer |
| Teradata | Migration |
| Amazon Redshift | Migration |

### Create Transfer (Python)

```python
from google.cloud import bigquery_datatransfer

client = bigquery_datatransfer.DataTransferServiceClient()

# Cloud Storage transfer
transfer_config = bigquery_datatransfer.TransferConfig(
    destination_dataset_id="my_dataset",
    display_name="GCS Daily Import",
    data_source_id="google_cloud_storage",
    schedule="every 24 hours",
    params={
        "data_path_template": "gs://bucket/data/dt=*/*.csv",
        "destination_table_name_template": "daily_data_{run_date}",
        "file_format": "CSV",
        "skip_leading_rows": "1",
    }
)

response = client.create_transfer_config(
    parent=f"projects/{project_id}/locations/US",
    transfer_config=transfer_config
)
```

### Scheduled Queries

```python
transfer_config = bigquery_datatransfer.TransferConfig(
    destination_dataset_id="analytics",
    display_name="Daily Aggregation",
    data_source_id="scheduled_query",
    schedule="every day 06:00",
    params={
        "query": """
            INSERT INTO `project.analytics.daily_summary`
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as events
            FROM `project.raw.events`
            WHERE DATE(timestamp) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
            GROUP BY 1
        """,
        "destination_table_name_template": "",
        "write_disposition": "WRITE_APPEND",
    }
)
```

## Dataflow Integration

### Read from BigQuery

```python
import apache_beam as beam
from apache_beam.io.gcp.bigquery import ReadFromBigQuery

with beam.Pipeline() as pipeline:
    rows = (
        pipeline
        | ReadFromBigQuery(
            query="SELECT * FROM `project.dataset.table`",
            use_standard_sql=True
        )
        | beam.Map(process_row)
    )
```

### Write to BigQuery

```python
from apache_beam.io.gcp.bigquery import WriteToBigQuery

with beam.Pipeline() as pipeline:
    (
        pipeline
        | beam.Create([{"name": "Alice", "score": 100}])
        | WriteToBigQuery(
            table="project:dataset.table",
            schema="name:STRING,score:INTEGER",
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
        )
    )
```

### Streaming to BigQuery

```python
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition

(
    pipeline
    | "Read PubSub" >> beam.io.ReadFromPubSub(topic="projects/p/topics/t")
    | "Parse JSON" >> beam.Map(json.loads)
    | "Write BQ" >> WriteToBigQuery(
        table="project:dataset.streaming_table",
        method=WriteToBigQuery.Method.STREAMING_INSERTS
    )
)
```

## Pub/Sub Integration

### BigQuery Subscription

```bash
# Create BigQuery subscription
gcloud pubsub subscriptions create my-bq-subscription \
  --topic=my-topic \
  --bigquery-table=PROJECT:DATASET.TABLE \
  --write-metadata
```

### Schema Requirements

```sql
-- Table schema for Pub/Sub subscription
CREATE TABLE `project.dataset.pubsub_messages`
(
  subscription_name STRING,
  message_id STRING,
  publish_time TIMESTAMP,
  data STRING,  -- or BYTES
  attributes JSON
);
```

## Connected Sheets

### Enable Connected Sheets

1. Open Google Sheets
2. Data > Data connectors > Connect to BigQuery
3. Select project and dataset
4. Write query or select table

### Scheduled Refresh

Connected Sheets can be configured to refresh automatically on a schedule.

## References

- `PYTHON_EXAMPLES.md` - Python code examples
- `API_REFERENCE.md` - REST API endpoints
- `JDBC_SETUP.md` - JDBC configuration guide

## Scripts

- `connection_test.py` - Test BigQuery connectivity
- `bulk_loader.py` - Bulk data loading utility
- `api_client.py` - REST API wrapper

## Limitations

- JDBC/ODBC: Query timeout 6 hours
- Streaming: 100,000 rows/second per table
- Data Transfer: Source-specific limits
- REST API: 10 MB response size
