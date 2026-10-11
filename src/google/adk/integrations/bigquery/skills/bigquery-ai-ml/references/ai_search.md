# BigQuery AI.Search

`AI.SEARCH` is a table-valued function for semantic search on tables that have
autonomous embedding generation enabled. If your table has a column that has
generated_expression metadata with format "AI.EMBED(source_column)", then
AI.SEARCH uses it to optimize the search.

You can use AI.SEARCH to help with the following tasks:

-   Semantic search: search entities ranked by semantic similarity.
-   Recommendation: return entities with attributes similar to a given entity.
-   Classification: return the class of entities whose attributes are similar to
    the given entity.
-   Clustering: cluster entities whose attributes are similar to a given entity.
-   Outlier detection: return entities whose attributes are least related to the
    given entity.

## Syntax Reference

```sql
AI.SEARCH(
  { TABLE base_table | base_table_query },
  column_to_search,
  query_value
  [, top_k => top_k_value ]
  [, distance_type => distance_type_value ]
  [, options => options_value]
)
```

**IMPORTANT:** Do not add "column_to_search =>" prefixes to the column_to_search
argument because column_to_search is a positional argument.

### Input Arguments

Argument               | Requirement  | Type     | Description
:--------------------- | :----------- | :------- | :----------
**`base_table`**       | **One Of**   | Table    | The table to search for nearest neighbor embeddings. The table must have autonomous embedding generation enabled.
**`base_table_query`** | **One Of**   | Subquery | The query that you can use to pre-filter the base table. Only SELECT, FROM, and WHERE clauses are allowed in this query. Don't apply any filters to the embedding column.
**`column_to_search`** | **Required** | STRING   | A STRING literal that contains the name of the string column to search
**`query_value`**      | **Required** | STRING   | A string literal that represents the search query.
**`top_k`**            | Optional     | INT64    | A named argument with an INT64 value, specifies the number of nearest neighbors to return. The default is 10.
**`distance_type`**    | Optional     | STRING   | A named argument with a STRING value. distance_type_value specifies the type of metric to use to compute the distance between two vectors. Supported distance types are EUCLIDEAN, COSINE, and DOT_PRODUCT. The default is EUCLIDEAN. Recommend to use COSINE.
**`options`**          | Optional     | STRING   | A named argument with a JSON-formatted STRING value that specifies the following search options: `fraction_lists_to_search` or `use_brute_force`

### Output Schema

Column Name    | Type    | Description
:------------- | :------ | :----------------------------------------------------
**`base`**     | STRUCT  | A struct containing all columns from the input table.
**`distance`** | FLOAT64 | The distance score between the query and the result.

## Examples

```sql
# Create a table with a product column, a description column, and an autonomous
# embedding generation column over the description column.
CREATE TABLE mydataset.products (
  name STRING,
  description STRING,
  description_embedding STRUCT<result ARRAY<FLOAT64>, status STRING>
    GENERATED ALWAYS AS (AI.EMBED(
      description,
      connection_id => 'us.example_connection',
      endpoint => 'text-embedding-005'
    ))
    STORED OPTIONS( asynchronous = TRUE )
);

# Get all really fun toy products.
SELECT
  base.name,
  base.description,
  distance
FROM AI.SEARCH(TABLE mydataset.products, 'description', "A really fun toy", distance_type => "COSINE");

# Get top 5 fun toy products.
SELECT DISTINCT(base.description)
FROM AI.SEARCH(TABLE `mydataset.products`, 'description', "A really fun toy", distance_type => "COSINE", top_k => 100)
LIMIT 5;
```
