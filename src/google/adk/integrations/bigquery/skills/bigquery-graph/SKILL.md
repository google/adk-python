---
name: bigquery-graph
license: Apache-2.0
metadata:
  author: google-adk
  version: "1.0"
description: |
  Skill for Graph Query Language (GQL) or SQL/PGQ queries against a property graph.
  Includes path finding, multi-hop traversal, topological connection, shortest path,
  node reachability, edge connectivity, and semantic graph queries.

---

# Graph Query Language (GQL) Query Generation Guidelines

You are querying a property graph consisting of nodes and edges. You **MUST
exclusively use the BigQuery GoogleSQL GQL standard**, which is the only
supported graph query language and implements the ISO GQL standard.

**You MUST NEVER, under any circumstances, generate or consider Cypher
queries.** Any deviation from the BigQuery GoogleSQL GQL standard is strictly
prohibited.

## Reference Directory

-   [Semantic Graph Guidelines](references/semantic_queries.md): Guidelines for
    generating SQL queries against a semantic graph.
-   [Graph Schema Best Practices](references/graph_schema/best_practices.md):
    Best practices for defining BigQuery property graph schemas.
-   [Graph Schema DDL Reference](references/graph_schema/ddl_reference.md): DDL
    syntax reference for property graphs.
-   [Graph Limitations & Feature Parity](references/graph_schema/feature_parity.md):
    Current limitations and feature parity for BigQuery Graph.
-   [Graph Schema DDL Advisor](references/graph_schema/graph_schema_ddl_advisor.md):
    Assists in defining and optimizing schemas.

## Pre-generation Checklist

Before generating any GQL, you MUST:

1.  **Identify Output Intent**: Determine if the user intends to **visualize a
    graph network** (requires `TO_JSON()`) or **view tabular data** (requires
    specific properties).
2.  **Verify Language Standard**: Confirm the query will use **BigQuery
    GoogleSQL GQL**. NEVER use Cypher.

## Core Directives for Agent Query Generation

When generating graph queries, you must adhere to the following global
directives:

1.  **Default Query Construction (Standalone GQL)**: Write standalone GQL
    queries using the `RETURN` statement natively. **Explicitly avoid using the
    `GRAPH_TABLE` table-valued function unless the user constraints actively
    require standard SQL relational integration or aggregation.**
2.  **Keyword Escaping**: You **MUST** enforce backticks (`) around any reserved
    SQL and GQL keywords such as 'order', 'begin' and 'path' used as identifiers
    (e.g., column names, label names, variable names).
3.  **Strictly Follow Graph Schema**: Ensure all labels (e.g., `:Person`,
    `:Account`) and properties (e.g., `n.id`, `e.amount`) used in the query
    strictly match the provided graph schema. Do NOT guess or hallucinate schema
    elements.
4.  **Result Uniqueness**: Use the `DISTINCT` keyword automatically in your
    `RETURN` or `COLUMNS` clause if the user prompt implies they want to
    retrieve unique information.
5.  **Graph Path Variables**: When a query involves "paths", "path traversal",
    "path finding", or finding relationships between nodes, you **MUST** assign
    the matched pattern to a path variable (e.g., `MATCH p = ...`).

## Basic GQL Query Construction

A linear query statement in BigQuery GQL executes clauses sequentially. The
output of one clause provides the input (the "working table") to the next.

Common sequential statements include:

-   `MATCH`: Identifies topological patterns in the graph.
-   `WITH`: Projects variables from current scope into the next scope,
    optionally sorting, limiting or grouping.
-   `LET`: Defines a new variable or alias within the query scope.
-   `FILTER`: Filters intermediate graph mappings.
-   `RETURN`: Ends a GQL query or subquery, projecting the final graph
    variables.
-   `ORDER BY`, `LIMIT`, `OFFSET`: Control sorting and pagination.

**Chaining with `NEXT`:** Multiple linear statements can be composed into a
compound query using the `NEXT` keyword. The results of the first statement pipe
into the statement following `NEXT`.

### Example 1: Sequential Statements

This query demonstrates filtering and projecting data through a single linear
statement sequence.

```sql
GRAPH <project>.<dataset>.<graph>
MATCH (src:Account)-[t:Transfers]->(dst:Account)
LET transfer_amount = t.amount
FILTER transfer_amount > 1000
WITH src, dst, transfer_amount
ORDER BY transfer_amount DESC
LIMIT 50
RETURN src.id AS source, dst.id AS destination, transfer_amount
```

### Example 2: Chaining with NEXT

This query demonstrates using `NEXT` to pipe the results of one graph pattern
match into a subsequent pattern match.

```sql
GRAPH <project>.<dataset>.<graph>
MATCH (blocked:Account WHERE blocked.is_frozen = true)
RETURN blocked.id AS frozen_id
NEXT
MATCH (a:Account)-[t:Transfers]->(b:Account)
FILTER a.id = frozen_id
RETURN a.id AS source, b.id AS destination, t.amount AS amount
```

## Graph Pattern Matching

A graph pattern matches topologies within a BigQuery property graph. Patterns
consist of vertices (nodes) and connecting edges.

### Node Patterns

Node patterns are enclosed in parentheses `()`. They identify entities in the
graph and can optionally bind to a variable or specify label and property
filters.

-   `MATCH (n)`: Matches any node and binds it to the variable `n`.
-   `MATCH (p:Person)`: Matches nodes explicitly labeled with `Person`.
-   `MATCH (p:Person|Account)`: Uses a label expression `|` (OR) to match nodes
    that have *either* the `Person` or `Account` label.
-   `MATCH (p:Person {id: 1})`: Matches nodes that satisfy a specific property
    filter.
-   `MATCH (p:Person WHERE p.age > 18)`: Matches nodes applying a `WHERE`
    condition on properties.

**Important Note on Label Expressions**: BigQuery matches a node if it possesses
*any* of the labels listed in an `OR` (`|`) expression. You cannot use `&`
directly in label expressions.

### Edge Patterns

Edge patterns represent the relationships between nodes. They are enclosed in
square brackets `[]` and connected using arrows (`-`, `->`, `<-`) to denote
directionality.

-   `MATCH (a)-[e]->(b)`: Matches any directed edge from `a` to `b`, binding the
    edge to `e`.
-   `MATCH (a)-[e:Transfers]->(b)`: Directed edge specifically labeled
    `Transfers`.
-   `MATCH (a)-[e:Transfers {amount: 50}]->(b)`: Edge with a specific property
    filter applied.
-   `MATCH (a)-[e:Transfers]-(b)`: Matches an undirected (any direction) edge
    between `a` and `b`. Use preferred explicit direction when possible for
    better performance.

### Pattern Joins and Commas

A complex graph pattern consists of one or more path patterns separated by
commas `,`. When multiple comma-separated patterns are used:

-   If they do not share any variables, they result in a **cross join**.
-   If they share a common variable, BigQuery automatically performs an
    **equijoin** on that variable.

```sql
-- Equijoin example where 'interim' connects the two paths
GRAPH <project>.<dataset>.<graph>
MATCH (src:Account)-[t1:Transfers]->(interim:Account),
      (interim)<-[:Owns]-(p:Person)
RETURN src.id AS account_id, p.name AS owner_name
```

### Variable-Length Paths and Quantifiers

You can find multi-hop connections by appending a quantifier to an edge pattern,
defining variable-length paths. - `{m, n}`: Specifies that the edge pattern must
be repeated between `m` and `n` times (e.g., `{1,3}`).

**Group Variables**: When an edge variable is quantified (e.g.,
`[e:Transfers]->{1,3}`), the variable `e` becomes a "group variable." This
represents an array of the matched edges in the path. You must use array
functions to interact with it, such as `ARRAY_LENGTH(e)` or horizontal
aggregation like `SUM(e.amount)`.

### Path Search Prefixes

Variable-length paths can result in exponential combinations and repeating
paths. You can constrain the search between source and destination pairs using
search prefixes placed immediately before the path pattern:

-   `ANY`: Returns exactly one arbitrary matching path between each unique pair
    of source and destination nodes.
-   `ANY SHORTEST`: Returns a single path for each unique pair, specifically
    choosing from those with the minimum number of edges (hops).
-   `ANY CHEAPEST`: Returns a single path with the minimum total cost, computed
    by aggregating `COST` expressions defined on the edges.

```sql
GRAPH <project>.<dataset>.<graph>
MATCH ANY SHORTEST
  (a:Account {id: 123})-[e:Transferred]->{1,3}(b:Account {id: 456})
RETURN e
```

## GQL Functions and Operators

BigQuery property graphs support specialized native functions for interrogating
graph elements and extracting path metadata. These functions can be used
directly within `MATCH`, `WHERE`, `LET`, and `RETURN`/`COLUMNS` clauses.

### Path Extraction Functions

When an entire path pattern is bound to a variable (e.g., `MATCH p = (...)`),
you can extract specific metadata and elements from it:

-   `PATH_FIRST(p)`: Extracts and returns the starting node of path `p`.
-   `PATH_LAST(p)`: Extracts and returns the terminal (ending) node of path `p`.
-   `PATH_LENGTH(p)`: Returns an `INT64` count representing the number of edge
    hops in path `p`.
-   `NODES(p)`: Returns an array of node elements, ordered by their sequence in
    the path.
-   `EDGES(p)`: Returns an array of edge elements, ordered by their sequence in
    the path.

```sql
GRAPH <project>.<dataset>.<graph>
MATCH p = (a:Account)-[t:Transfers]->{1,3}(b:Account)
RETURN PATH_LENGTH(p) AS hops, TO_JSON(NODES(p)) AS path_nodes
```

### Element Traversal and Inspection Functions

These functions operate on individual node or edge element variables:

-   `DESTINATION_NODE_ID(e)`: Retrieves the unique internal string identifier of
    an edge `e`'s destination node.
-   `SOURCE_NODE_ID(e)`: Retrieves the unique internal string identifier of an
    edge `e`'s source node.
-   `ELEMENT_ID(x)`: Returns the unique internal identifier for the given node
    or edge `x`.
-   `LABELS(x)`: Returns an array of string labels bound to a node or edge
    element `x`.

## Output Formatting: Graph Visualization vs. Tabular Data

When constructing the `RETURN` clause, strictly distinguish between **graph
visualization** intent and **tabular data** intent based on the user's
objective.

### Path Variables

You can assign an entire matched pattern sequence to a path variable using the
assignment operator `=`. This allows you to reference the entire topological
sequence later in the query.

```sql
MATCH p = (a:Person)-[e:Knows]->(b:Person)
```

In this example, `p` represents the full path, encapsulating the nodes `a` and
`b` and the edge `e`.

### 1. Graph Visualization Intent

Use this when the user wants to see relationships, paths, topology, networks,
connectivity or entire entities (nodes/edges) as a whole.

-   **Trigger & Keywords**: "visualize", "show the graph", "network",
    "connections", "find the path", "relationship between X and Y".
-   **Default JSON Serialization (`TO_JSON`)**: Unless specific properties
    (e.g., `n.name`) or path metrics (e.g., `PATH_LENGTH(p)`) are explicitly
    requested, you **MUST** wrap all graph topology outputs (nodes, edges, and
    path variables) in the standard `TO_JSON()` function. This ensures
    compatibility with graphing UI components that expect full JSON objects.
-   **Example**: `RETURN TO_JSON(src) AS source, TO_JSON(p) AS full_path`
-   **Limit**: Always append `LIMIT 500` to the query to prevent overwhelming
    the UI with too many nodes/edges, unless the user explicitly requests a
    different number.

```sql
GRAPH <project>.<dataset>.<graph>
MATCH p = (src:Person)-[e:Knows]->(dst:Person)
RETURN
    TO_JSON(src) AS source_node,
    TO_JSON(e) AS relationship,
    TO_JSON(dst) AS destination_node,
    TO_JSON(p) AS full_path
LIMIT 500
```

### 2. Tabular or Chart Intent

Use this when the user focuses on specific attributes, statistics, or metrics.

-   **Trigger & Keywords**: "what is the name", "list", "how many", "count",
    "average", "top 10", "aggregate".
-   **Action**: Return ONLY the specific required properties or aggregates. **Do
    NOT** use `TO_JSON()`.
-   **Example**: `RETURN account.id, SUM(t.amount) AS total_transfer`

## GRAPH_TABLE Syntax and SQL Integration

The `GRAPH_TABLE` table-valued function is the primary mechanism for integrating
property graph queries with standard SQL operations in BigQuery.

### When to Use GRAPH_TABLE

You **SHOULD** use `GRAPH_TABLE()` only when your query requires integration
with SQL capabilities beyond basic graph pattern matching. Use it for:

-   **SQL Aggregations & Analysis**: Mixing graph pattern matching with standard
    SQL aggregations (e.g., `SUM`, `COUNT`, `GROUP BY`).
-   **Relational Joins**: Joining graph query results with relational tables or
    other `GRAPH_TABLE` calls.
-   **Advanced SQL Operations**: Utilizing advanced SQL filtering, reporting, or
    pagination on the graph results.

### Basic Syntax

The basic structure of a `GRAPH_TABLE` query involves specifying the graph name,
the GQL statements, and a `COLUMNS` clause to define the output relational
schema.

```sql
SELECT
  src_account_id,
  COUNT(*) AS transfer_count,
  SUM(amount) AS total_transfer_volume
FROM GRAPH_TABLE(
    <project>.<dataset>.<graph>
    MATCH (src:Account)-[t:Transfers]->(dst:Account)
    WHERE src.is_blocked = true
    COLUMNS (src.id AS src_account_id, t.amount AS amount)
)
GROUP BY src_account_id
HAVING total_transfer_volume > 10000
ORDER BY total_transfer_volume DESC
```

### The COLUMNS Clause

The `COLUMNS` clause is mandatory if you want to explicitly define the returned
table's schema.

-   **Explicit Projection**: It limits the output to only the specified
    expressions from the graph query scope.
-   **Anonymous Columns**: You *must* alias any expressions in the `COLUMNS`
    clause if they generate an anonymous column (e.g., `COLUMNS (t.amount * 2 AS
    doubled_amount)`).
-   **Default Behavior**: If the `COLUMNS` clause is entirely omitted,
    `GRAPH_TABLE` returns all graph pattern variables present in the query
    scope.
-   **Aggregations**: You can include standard SQL aggregate functions directly
    within the `COLUMNS` clause to perform grouping and aggregation across the
    rows of the resulting graph matches.

### Joins with Relational Tables

You can join the result of `GRAPH_TABLE` with other standard BigQuery tables or
even other `GRAPH_TABLE` results using standard SQL semantics (e.g., `JOIN`,
`LEFT JOIN`).

To make a `GRAPH_TABLE` aware of variables from an earlier table in the `FROM`
clause, you can use parameterized `GRAPH_TABLE`. In the example below, `a.id`
from the `Accounts` table is passed into the `GRAPH_TABLE` scope:

```sql
SELECT
  a.name,
  g.total_amount
FROM Accounts AS a
JOIN GRAPH_TABLE(
    <project>.<dataset>.<graph>
    MATCH (src:Account {id: a.id})-[t:Transfers]->(dst:Account)
    COLUMNS (SUM(t.amount) AS total_amount)
) AS g
```

## Subquery Limitations

A subquery in BigQuery GQL is enclosed in braces `{}` and evaluates nested
operations within a linear query statement. While BigQuery Graph supports
subqueries, there are critical limitations and syntax differences compared to
standard GoogleSQL that you **MUST** adhere to.

### Mandatory Graph Name Specification

In BigQuery Graph, unlike standard GoogleSQL, you **MUST** specify the graph
name within the subquery block. If the outer query uses `GRAPH
<project>.<dataset>.<graph>`, the internal subquery must also explicitly
redeclare it.

```sql
MATCH (n1)
WHERE EXISTS {
  -- REQUIRED: You must re-specify the graph name here
  GRAPH <project>.<dataset>.<graph>
  MATCH (n2)
  WHERE n1 = n2
  RETURN 1 as one
}
```

Failure to include the graph name in the subquery will result in a job-server
error.

### The WHERE vs. FILTER Rule

Certain types of subqueries **throw errors when used inside a `WHERE` clause**
because BigQuery's query planner cannot decorrelate them if they act as join
predicates.

For the following subquery types, you **CANNOT** use the `WHERE` clause. You
**MUST** use the `FILTER` clause instead: `EXISTS`, `IN`, `LIKE`, and `LIKE
ANY/SOME/ALL` subqueries.

**INCORRECT (will throw error):**

```sql
MATCH (p:Person) WHERE EXISTS { GRAPH
<project>.<dataset>.<graph> MATCH (p)-[:Owns]->(:Account) }
```

**CORRECT:**

```sql
MATCH (p:Person) FILTER EXISTS { GRAPH
<project>.<dataset>.<graph> MATCH (p)-[:Owns]->(:Account) }
```

### Supported Subquery Types and Correlations

-   **`ARRAY` Subquery**: Fully Supported. Evaluates the query block and returns
    an array of the results.
-   **`VALUE` Subquery**: Partially Supported. Evaluates the internal query and
    returns a single scalar value. **Limitation**: `VALUE` subqueries throw
    errors when correlated variables from the outer block are referenced inside
    the `VALUE` subquery.
-   **`EXISTS`, `IN`, `LIKE` Subqueries**: Partially Supported. **Limitation**:
    Throw errors when correlated variables are used. Throw errors when used in
    `WHERE` filter (Must use `FILTER`).

## Query Optimization and Best Practices

Performance is a key consideration for highly connected BigQuery graphs. Adhere
to these principles whenever writing GQL statements to ensure optimal execution.

### 1. Start Traversals From Low-Cardinality Nodes

Always write your path traversals so they originate from the lowest cardinality
nodes (the most specific entities). This drastically reduces the intermediate
result set sizes and speeds up execution, especially for variable-length
traversals.

-   **Example**: Instead of starting from a highly active `Account` node and
    traversing backwards to find the owner, start with the specific `Person`
    node and traverse forward.
-   **Filter Early**: Push specific properties (e.g., `Account {id: 7}`) as
    early as possible in your `MATCH` clause to prune the search space
    immediately.

### 2. Specify Labels Explicitly

You must explicitly provide node and edge labels when they are known (e.g.,
`(a:Account)-[:Transfers]->(b:Account)`).

While BigQuery attempts to infer labels from query usage, if inference fails or
labels are omitted, the engine is forced to perform full table scans over
multiple distinct underlying node/edge tables.

### 3. Avoid Bi-directional Graph Traversals

BigQuery Graph schema physical implementations are directional. You should
always specify a source and destination node for an edge (using `->` or `<-`).

Although query pattern syntax allows for bidirectional or undirected path
traversal (`(node)-[edge]-(node)`), doing so incurs a severe implicit
performance penalty.

If you need to find an edge between two specific nodes regardless of direction,
**DO NOT** use a bidirectional pattern. Instead, use explicit directional
traversals combined with `UNION ALL`:

**GOOD:**

```sql
GRAPH <project>.<dataset>.<graph> MATCH (a1:Account
{id:10})-[t:Transfer]->(a2:Account {id: 20}) RETURN t UNION ALL MATCH
(a2:Account{id: 20})-[t:Transfer]->(a1:Account {id: 10}) RETURN t
```

### 4. Prefer Single MATCH Statements

When possible without sacrificing readability or violating logic intent, prefer
composing a single comprehensive `MATCH` statement over chaining multiple
individual `MATCH` statements. A single statement allows the query optimizer a
wider global view of the graph pattern, often leading to better execution plans.
