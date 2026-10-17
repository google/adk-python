# BigQuery Graph Limitations & Feature Parity

This document outlines the current feature parity, structural requirements, and
querying limitations for BigQuery Property Graphs and Semantic Graphs (queried
via `GRAPH_EXPAND`).

--------------------------------------------------------------------------------

## 1. Property Graph vs. Semantic Graph

A **Semantic Graph** in BigQuery is not a distinct resource type. It is
physically represented and stored as a standard **Property Graph**, enriched
with semantic metadata.

| Feature               | Property Graph      | Semantic Graph            |
| :-------------------- | :------------------ | :------------------------ |
| **Core Elements**     | Nodes & Edges       | Nodes & Edges             |
| **Attributes**        | Standard Properties | Dimensions & **Measures** |
:                       : (Dimensions)        :                           :
| **Metadata**          | Name, Labels        | Descriptions, Synonyms,   |
:                       :                     : is_measure                :
| **Querying Language** | GQL (Graph Query    | GQL or SQL via            |
:                       : Language)           : **`GRAPH_EXPAND`**        :

--------------------------------------------------------------------------------

## 2. Structural Limitations of `GRAPH_EXPAND` (Critical)

The `GRAPH_EXPAND` TVF flattens a property graph into a "One-Big-Table" (OBT)
schema at query time. This imposes strict structural constraints on the graph's
topology.

*   **Tree Structure Requirement**: Currently, `GRAPH_EXPAND` requires the graph
    to form a valid **Tree** (or a set of tree-structured hierarchies).
*   **Unsupported Graph Topologies**:
    *   **Disconnected Graphs (Forests / Multiple Roots)**: Multiple independent
        subgraphs or distinct tree hierarchies (e.g., multiple root nodes with
        no shared parent) are not supported in a single `GRAPH_EXPAND` call.
        Traversal must start from a single root node.
    *   **Cyclic (Circular) Directed Graphs**: A directed cycle is not allowed.
        This is a path starting at a node that traverses edges in their defined
        direction and eventually loops back to the same node (e.g., `A -> B -> C
        -> A`).
    *   **Self-Loops**: An edge connecting a node directly to itself (e.g., `A
        -> A`) is not supported.
    *   **Convergent Paths (Diamonds)**: Multiple distinct directed paths
        connecting the same starting (ancestor) node to the same destination
        (descendant) node are not supported. *(Note: This is distinct from
        undirected cycles; if the paths have different directions, it may be
        supported, but a strict directed diamond pattern is disallowed.)*

**Workaround**: If your graph has disconnected components, query them using
separate, isolated `GRAPH_EXPAND` calls. If your graph has cycles, simplify the
schema or use standard GQL pattern matching instead of the flattened
`GRAPH_EXPAND` view.

--------------------------------------------------------------------------------

## 3. Querying Restrictions on Measures

Measures defined in the DDL (using the `MEASURE` keyword) have special behavior
under the hood (grain-locking) to prevent overcounting during joins. This
introduces specific querying rules.

### A. Mandatory `AGG()` Function

*   You **CANNOT** select a measure column directly in a standard `SELECT` list.
*   You **MUST** wrap all measure column references in the special GoogleSQL
    **`AGG()`** function.
*   **Incorrect**:

    ```sql
    SELECT Order_total_order_amount FROM GRAPH_EXPAND("my_graph")
    ```

    *Throws error*: `Returning expressions of type MEASURE is not allowed.`

*   **Correct**:

    ```sql
    SELECT AGG(Order_total_order_amount) FROM GRAPH_EXPAND("my_graph")
    ```

### B. No Direct `SELECT *`

*   Applying `SELECT *` on a `GRAPH_EXPAND` output over a graph that has
    measures will fail because it implicitly attempts to select the measure
    columns without `AGG()`.
*   **Workaround**: Explicitly list the non-measure columns, or use `SELECT *
    EXCEPT (measure_col1, ...)` to exclude them.

### C. Unsupported Correlated Subqueries

*   Using a measure column inside a complex correlated subquery (such as an
    `EXISTS` clause) that the BigQuery query optimizer cannot automatically
    de-correlate will fail.
*   **Workaround**: Rewrite the query to use explicit `JOIN`s, or perform the
    aggregation in a Common Table Expression (CTE) first.

### D. Incompatible with Native GQL Queries

*   **GQL Limitation**: Columns declared as `MEASURE` in the DDL *cannot* be
    referenced or projected in native GQL queries (e.g., `GRAPH_TABLE`). Native
    GQL executes strictly on standard property graph dimensions.
*   **Workaround**: To query measures, you must use standard SQL with
    `GRAPH_EXPAND` and wrap references in `AGG()`.
