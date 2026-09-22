---
name: semantic-graph-guidelines
description:
  Guidelines for generating SQL queries against a semantic graph. Use
  for tabular analysis, reporting, or BI metrics on graph data.
---

# Semantic Graph Specific Rules

1.  **Query the Flattened View:** Always query from the semantic graph using the
    `GRAPH_EXPAND` table-valued function (TVF). The argument to `GRAPH_EXPAND`
    should be the full graph name string (e.g.,
    "project_id.dataset_id.property_graph_id").

    *   **CRITICAL RULE:** The semantic graph is NOT a regular table, even
        though its schema may be presented using `CREATE TABLE`. It is a graph.
        You **MUST NEVER** query it directly as a table (e.g., `FROM
        my_project.my_dataset.my_graph`).
    *   **CRITICAL FALLBACK RULE:** If a query using `GRAPH_EXPAND` fails (e.g.,
        due to syntax errors or system limits), **DO NOT** attempt to fallback
        to querying it as a standard table. Doing so will result in a critical
        `NOT_FOUND` error.
    *   The semantic graph is a virtual flattened view of the graph, which is
        optimized for data analysis and answering questions.

        ```sql
        SELECT ...
        FROM GRAPH_EXPAND("project_id.dataset_id.property_graph_id")
        WHERE ...
        ```

2.  **Querying Measures:** Columns marked with `is_measure=TRUE` in the schema
    (e.g., `Customer_customer_count INT64 OPTIONS(is_measure=TRUE)`) are measure
    columns. You **MUST** query these columns using the `AGG()` function.

    *   **Syntax:** `AGG(<measure_column_name>)`
    *   **Example:**

        ```sql
        -- Given Schema:
        -- CREATE TABLE `my_project.my_dataset.my_graph` (
        --   Customer_name STRING,
        --   Customer_total_orders INT64 OPTIONS(is_measure=TRUE),
        --   Product_name STRING
        -- );

        -- Querying the measure:
        SELECT
          Customer_name,
          AGG(Customer_total_orders) AS total_orders
        FROM GRAPH_EXPAND("my_project.my_dataset.my_graph")
        GROUP BY Customer_name;
        ```

    *   Do not apply other aggregation functions like `SUM`, `AVG`, etc.
        directly to measure columns. Use `AGG()` instead.

3.  **Prefer Measures (AGG) over Standard SQL Aggregations:** You **MUST**
    prioritize using pre-defined measures (columns with `is_measure=TRUE`) over
    writing standard SQL aggregations (like `COUNT(DISTINCT ...)`, `SUM`, etc.)
    whenever a relevant measure is available in the schema.

    *   **Context:** Semantic graphs define business logic within measures to
        ensure accuracy and prevent issues like overcounting. Generating
        aggregations via standard SQL bypasses this logic.
    *   **Example Scenario:** If the user asks for the "total number of
        entities", and the schema provides an `Entity_id` column as well as a
        measure column `Entity_count INT64 OPTIONS(is_measure=TRUE)`:
        *   **INCORRECT (Standard SQL):** `sql SELECT COUNT(DISTINCT Entity_id)
            AS total_entities ...`
        *   **CORRECT (Measure):** `sql SELECT AGG(Entity_count) AS
            total_entities ...`
