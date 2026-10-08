# BigQuery Graph Schema Best Practices

This document outlines best practices for designing and defining Property Graph
and Semantic Graph schemas in BigQuery. Following these guidelines improves
graph query performance, ensures referential integrity, and avoids common
pitfalls in flattened views (`GRAPH_EXPAND`).

--------------------------------------------------------------------------------

## 1. Scope Property Definitions (Critical for Performance)

Properties are key-value pairs attached to nodes or edges. By default, or if
using `PROPERTIES ALL COLUMNS`, all columns from the source table are attached.

*   **The Pitfall**: Exposing unnecessary properties forces BigQuery to perform
    redundant column scans in graph queries, severely degrading performance.
*   **Best Practice**: **Only include properties that are actually needed for
    querying.** Use the explicit `PROPERTIES (col1, col2, ...)` syntax to
    restrict the property list.
*   **Example**:

```sql
-- POOR: Exposes all columns including large text or metadata
NODE TABLES ( my_dataset.users PROPERTIES ALL COLUMNS )

-- GOOD: Only exposes relevant querying attributes
NODE TABLES ( my_dataset.users PROPERTIES (user_id, name, age) )
```

--------------------------------------------------------------------------------

## 2. Define Key Constraints (PK / FK)

BigQuery doesn't strictly enforce Primary Key (PK) or Foreign Key (FK)
constraints at runtime, but it uses them to optimize execution plans.

*   **Optimization**: If PK/FK constraints are defined on the underlying tables,
    the query engine can leverage them to eliminate unnecessary table scans and
    prune join paths.
*   **Referential Integrity**: Ensure your application guarantees the uniqueness
    of primary keys and referential integrity of foreign keys. If they are
    violated, graph query results may be incorrect.
*   **Best Practice**: Always define PK on node tables and FK on edge tables in
    their source DDL, and reference them in `CREATE PROPERTY GRAPH`.

--------------------------------------------------------------------------------

## 3. Avoid Column Name Collisions in Flattened Schema (`GRAPH_EXPAND`)

The `GRAPH_EXPAND` TVF flattens the graph by prefixing each property with the
Node/Edge alias (e.g., `NodeAlias_propertyName`).

*   **The Danger**: If the combination of alias and property name results in
    identical column names, the query will fail with a generic internal error:
    `Error encountered during execution. Retrying may solve the problem.`
*   **Scenario**:
    *   Node `N` with property `a_b` -> Generated column: `N_a_b`
    *   Node `N_a` with property `b` -> Generated column: `N_a_b` (Collision!)
*   **Best Practice**: Design your node/edge aliases and property names
    carefully to avoid prefix-induced collisions. Renaming properties or using
    distinct aliases in the DDL resolves this.

--------------------------------------------------------------------------------

## 4. Always Use Safe Aliases (`AS alias`)

If you omit the `AS alias` clause, BigQuery defaults to using the full table
path as the alias (e.g., `project.dataset.table`).

*   **The Pitfall**: The generated column names in the flattened view will
    contain dots and hyphens (e.g., `project.dataset.table_property`). This
    violates standard SQL output schema rules, and queries like `SELECT *` will
    fail with `Invalid field name`.
*   **Best Practice**: **Always specify a simple, alphanumeric alias** using
    standard SQL naming conventions (no dots, hyphens, or special characters).
*   **Example**:

```sql
-- POOR (Omitted alias):
NODE TABLES ( `my-project.my_dataset.user_profiles` KEY(id) ... )

-- GOOD (Safe alias):
NODE TABLES ( `my-project.my_dataset.user_profiles` AS User KEY(id) ... )
```

--------------------------------------------------------------------------------

## 5. Reusing the Same Physical Table as Node and Edge Tables

In hierarchical schemas (such as employee-manager org charts or product category
trees), the same physical table often represents both the entity (Node) and the
parent-child relationship (Edge).

When modeling this in DDL, you must decide how the reused table is exposed in
`GRAPH_EXPAND`:

1.  **Explicit Edges (Special/Explicit Properties)**:
    *   **Approach**: Declare the table as an edge table and list specific
        property columns in the `PROPERTIES(...)` clause.
    *   **Result**: These property columns will be exposed in the flattened
        output view as `EdgeAlias_propertyName`. Use this when the
        self-referential relationship itself carries important metadata (e.g.,
        `assignment_date`, `relation_type`).
2.  **Logical/Structural Edges (No Properties)**:
    *   **Approach**: Declare the table as an edge table but specify `NO
        PROPERTIES`.
    *   **Result**: The edge remains "invisible" in the output columns of the
        flattened view, while still correctly representing the hierarchical
        structure for navigation and query path resolution in the backend. Use
        this to avoid cluttering the output view when only the connectivity
        matters.

*   **Example**: Self-referential organizational chart:

```sql
CREATE OR REPLACE PROPERTY GRAPH `my-project.my_dataset.org_chart`
  NODE TABLES (
    `my-project.my_dataset.employees` AS Employee
      KEY(emp_id)
      LABEL Employee
      PROPERTIES(emp_id, name, department)
  )
  EDGE TABLES (
    -- Reusing 'employees' table purely to represent the 'reports_to' edge
    `my-project.my_dataset.employees` AS ReportsTo
      KEY(emp_id)
      SOURCE KEY(emp_id) REFERENCES Employee(emp_id)
      DESTINATION KEY(manager_id) REFERENCES Employee(emp_id)
      LABEL ReportsTo
      NO PROPERTIES -- Logical edge (structural only)
  );
```

--------------------------------------------------------------------------------

## 6. Handling Special Characters in Aliases

If you absolutely must use special characters (like hyphens or spaces) in your
aliases, you must be extremely careful with quoting.

*   **DDL Quoting**: Quoting is required in the DDL:

```sql
NODE TABLES ( my_table AS `My-Node` ... )
```

*   **Querying Quoting**: You **MUST** use backticks when referencing these
    columns in queries:

```sql
SELECT `My-Node_property` FROM GRAPH_EXPAND(...)
```

*   **Pitfall**: Omitting backticks (e.g., `SELECT My-Node_property`) causes the
    query engine to interpret the hyphen as a subtraction operator (`My` minus
    `Node_property`), throwing syntax errors.
