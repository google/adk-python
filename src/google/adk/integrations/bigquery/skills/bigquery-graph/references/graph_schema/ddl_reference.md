# BigQuery Property Graph DDL Reference

This document provides the detailed syntax and examples for defining a Property
Graph (and its semantic extensions) in BigQuery using `CREATE PROPERTY GRAPH`
DDL.

## 1. Basic Syntax

```sql
CREATE [ OR REPLACE ] PROPERTY GRAPH [ IF NOT EXISTS ] property_graph_name
  property_graph_content;

property_graph_content:
  node_tables
  [ edge_tables ]
```

*   **`OR REPLACE`**: Replaces any property graph with the same name if it
    exists. Cannot appear with `IF NOT EXISTS`.
*   **`IF NOT EXISTS`**: If a property graph with the same name exists, the
    statement has no effect. Cannot appear with `OR REPLACE`.
*   **`property_graph_name`**: The name of the property graph (can be a path
    expression e.g. `project.dataset.graph`).

--------------------------------------------------------------------------------

## 2. Node Tables (`NODE TABLES`)

Defines the node types (entities) in the graph by mapping them to existing
BigQuery tables or views.

```sql
NODE TABLES (
  node_element[, ...]
)

node_element:
  table_name [ AS node_alias ]
    [ KEY (column_name_list) ]
    [ LABEL label_name | DEFAULT LABEL ]
    [ PROPERTIES (property_element_list) | NO PROPERTIES | PROPERTIES ALL COLUMNS ]
```

*   **`table_name`**: The source BigQuery table/view.
*   **`AS node_alias`**: A unique alias for the node type. **Highly
    recommended** to avoid unsafe characters in generated column names.
*   **`KEY (column_name_list)`**: Specifies the unique identifier (Primary Key)
    for the node. Defaults to the source table's primary key if omitted.
*   **`LABEL label_name`**: Assigns a label to the node (used in queries).
    Defaults to the table name (or alias) if `DEFAULT LABEL` or omitted.
*   **`PROPERTIES`**: Specifies which columns are exposed as properties.
    *   `PROPERTIES (col1, col2)`: Exposes only specified columns. (Recommended
        for performance).
    *   `PROPERTIES ALL COLUMNS [ EXCEPT (col1, ...) ]`: Exposes all columns
        (optionally excluding some).
    *   `NO PROPERTIES`: Exposes no properties.

--------------------------------------------------------------------------------

## 3. Edge Tables (`EDGE TABLES`)

Defines the edge types (relationships) in the graph by mapping them to existing
BigQuery tables or views, linking source and destination nodes.

```sql
EDGE TABLES (
  edge_element[, ...]
)

edge_element:
  table_name [ AS edge_alias ]
    [ KEY (column_name_list) ]
    SOURCE KEY (edge_column_list) REFERENCES source_node_alias [ (node_column_list) ]
    DESTINATION KEY (edge_column_list) REFERENCES destination_node_alias [ (node_column_list) ]
    [ LABEL label_name | DEFAULT LABEL ]
    [ PROPERTIES (property_element_list) | NO PROPERTIES | PROPERTIES ALL COLUMNS ]
```

*   **`SOURCE KEY`**: Maps foreign key columns in the edge table to the `KEY`
    columns of the source node table.
*   **`DESTINATION KEY`**: Maps foreign key columns in the edge table to the
    `KEY` columns of the destination node table.
*   **`REFERENCES node_alias`**: Specifies the alias of the target node table.

--------------------------------------------------------------------------------

## 4. Semantic Extensions (Measures & Options)

BigQuery Semantic Graphs are built on top of Property Graphs by adding `MEASURE`
definitions, property options, and node/label-level metadata options.

### A. Measures (`MEASURE`)

Measures represent predefined calculations or aggregations (e.g., key business
metrics) defined inside the `PROPERTIES` list.

> [!IMPORTANT] **Dimension Property Requirement**: Any source column referenced
> inside a `MEASURE` aggregate expression **MUST** also be explicitly declared
> as a standard dimension property in the *same* `PROPERTIES(...)` block. If you
> aggregate a column that is not exposed as a dimension, the query will fail.

```sql
PROPERTIES (
  amount, -- Expose 'amount' as a dimension property (MANDATORY)
  MEASURE(SUM(amount)) AS total_order_amount -- Aggregate the dimension property
)
```

*   **Syntax**: `MEASURE(AGG_FUNC(column)) AS measure_name`
*   **Example**: `MEASURE(SUM(amount)) AS total_amount`
*   **Aggregations**: Supports standard aggregate functions like `SUM`, `COUNT`,
    `AVG`, `MIN`, `MAX`, `COUNT(DISTINCT ...)`.
*   **Limitation**: Measures must be explicitly aliased (anonymous measures are
    not allowed).

### B. Property Options (`OPTIONS`)

You can enrich properties (both dimensions and measures) with metadata like
descriptions and synonyms for better discoverability in natural language
interfaces.

```sql
property_name OPTIONS (
  [ description = "description_string" ]
  [, synonyms = ["synonym_1", "synonym_2", ...] ]
)
```

*   **`description`**: A string describing the property.
*   **`synonyms`**: An array of alternative names.

### C. Label and Node Table Options (`OPTIONS`)

You can also attach metadata options 1:1 to a node or edge element via an
`OPTIONS` clause. This is only supported when using `DEFAULT LABEL`, via the
`DEFAULT LABEL OPTIONS()` syntax. Setting `OPTIONS` directly on node/edge tables
or on explicitly named labels (e.g. `LABEL MyLabel OPTIONS(...)`) is not
supported.

```sql
-- Node/Edge Table with default label options:
NODE TABLES (
  Person
    DEFAULT LABEL OPTIONS(description = "Source table containing customer personal profiles")
    PROPERTIES(id)
)
```

--------------------------------------------------------------------------------

## 5. Complete Example: Standard Property Graph

This example shows a standard property graph mapping user profiles and their
orders, scoping only essential columns as properties and employing safe aliases.

```sql
CREATE OR REPLACE PROPERTY GRAPH `my-project.my_dataset.ecomm_standard_graph`
  NODE TABLES (
    `my-project.my_dataset.users` AS User
      KEY(user_id)
      LABEL User
      PROPERTIES (user_id, name, age),
    `my-project.my_dataset.orders` AS Order
      KEY(order_id)
      LABEL Order
      PROPERTIES (order_id, order_date, amount)
  )
  EDGE TABLES (
    `my-project.my_dataset.user_orders` AS PLACED
      KEY(order_id)
      SOURCE KEY(user_id) REFERENCES User (user_id)
      DESTINATION KEY(order_id) REFERENCES Order (order_id)
      LABEL PLACED
      NO PROPERTIES
  );
```

--------------------------------------------------------------------------------

## 6. Complete Example: Semantic Graph

This example extends the standard graph with rich semantic annotations:
node-level option descriptions, column-level descriptions/synonyms, and
pre-defined aggregate measures.

```sql
CREATE OR REPLACE PROPERTY GRAPH `my-project.my_dataset.ecomm_semantic_graph`
  NODE TABLES (
    `my-project.my_dataset.users` AS User
      KEY(user_id)
      DEFAULT LABEL
      PROPERTIES (
        user_id OPTIONS(description="Unique system identifier for each user"),
        name,
        age
      ),
    `my-project.my_dataset.orders` AS Order
      KEY(order_id)
      DEFAULT LABEL
      PROPERTIES (
        order_id,
        order_date,
        amount, -- Dimension property declaration (required for measure below)
        MEASURE(SUM(amount)) AS total_order_amount OPTIONS(
          description="Total revenue aggregated from user orders",
          synonyms=["revenue", "sales_turnover"]
        ),
        MEASURE(COUNT(*)) AS order_count OPTIONS(
          description="Count of orders placed by customers"
        )
      )
  )
  EDGE TABLES (
    `my-project.my_dataset.orders` AS PLACED
      KEY(order_id)
      SOURCE KEY(user_id) REFERENCES User (user_id)
      DESTINATION KEY(order_id) REFERENCES Order (order_id)
      DEFAULT LABEL
      NO PROPERTIES
  );
```
