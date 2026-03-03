---
name: bigquery-data-analysis
description: >
  Curated instructions for BigQuery data analysis using
  the BigQuery toolset. Guides schema exploration, SQL
  query composition, iterative analysis workflows, and
  error handling. Use when working with BigQuery data,
  SQL queries, or data analysis tasks.
license: Apache-2.0
metadata:
  author: google-adk
  version: "1.0"
---

## BigQuery Data Analysis Workflow

Follow these steps when performing data analysis with BigQuery tools.
Load reference files as needed for detailed patterns and examples.

### Step 1: Understand the Data Landscape

Before writing any queries, explore the available data:

1. Use `list_dataset_ids` to discover available datasets.
2. Use `list_table_ids` with a dataset ID to see tables within it.
3. Use `get_table_info` to inspect schema, partitioning, and row counts.
4. Load `references/schema_exploration.md` for advanced exploration patterns.

Always confirm the schema before composing queries. Do not guess column
names or types.

### Step 2: Compose SQL Queries

When using `execute_sql`, follow these best practices:

- Always use fully-qualified table names: `project.dataset.table`.
- Add `LIMIT` to exploratory queries to avoid scanning excessive data.
- Use CTEs (`WITH` clauses) for readability and modularity.
- Handle NULLs explicitly with `IFNULL`, `COALESCE`, or `IS NOT NULL`.
- Use parameterized patterns where possible.
- Load `references/sql_patterns.md` for window functions, STRUCT/ARRAY
  handling, approximate aggregations, and partitioned table patterns.

### Step 3: Iterative Analysis

Follow an iterative analysis cycle:

1. **Explore** - Run a small sample query to understand the data shape.
2. **Hypothesize** - Form a hypothesis about patterns or relationships.
3. **Execute** - Write and run a targeted query to test the hypothesis.
4. **Validate** - Check results for reasonableness (row counts, ranges,
   NULL rates).
5. **Refine** - Adjust the query or form a new hypothesis based on results.

Present intermediate findings to the user. Ask for confirmation before
proceeding to complex or costly analyses.

### Step 4: Advanced Analysis

Use specialized tools for advanced analytics:

- `forecast` - Time-series forecasting with BigQuery ML.
- `analyze_contribution` - Attribution and contribution analysis.
- `detect_anomalies` - Anomaly detection in time-series data.
- `ask_data_insights` - Natural language data insights.

These tools handle ML model creation and inference. Provide them with
the appropriate table references and column names from Step 1.

### Error Handling

When queries fail:

1. Read the error message carefully for syntax or permission issues.
2. Use `get_job_info` with the job ID to check job status and details.
3. Load `references/error_handling.md` for common error codes and
   resolution strategies.
4. For quota or timeout errors, consider reducing query scope with
   tighter filters, date ranges, or `LIMIT`.

### Safety Rules

- Never run destructive SQL (DELETE, DROP, TRUNCATE, UPDATE) unless the
  user explicitly requests it.
- Always preview data with SELECT before suggesting modifications.
- Warn the user about queries that may scan large amounts of data.
- Use `LIMIT` on exploratory queries to control costs.
