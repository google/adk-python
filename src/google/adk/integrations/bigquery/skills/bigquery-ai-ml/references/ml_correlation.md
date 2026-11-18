# BigQuery ML.CORRELATION

`ML.CORRELATION` is a Table-Valued Function (TVF) that calculates the
statistical correlation between a target column and one or more target
correlation columns, optionally sliced by various dimensions.

## Syntax Reference

```sql
ML.CORRELATION(
  { TABLE TABLE_NAME | (QUERY_STATEMENT) },
  target_col => TARGET_COL,
  target_correlation_cols => TARGET_CORRELATION_COLS
  [, dimension_cols => DIMENSION_COLS ]
  [, method => METHOD ]
)
```

### Input Arguments

Argument                      | Requirement  | Type                        | Description
:---------------------------- | :----------- | :-------------------------- | :----------
**`input_data (positional)`** | **Required** |                             | The source table or query containing the data to analyze.
**`target_col`**              | **Required** | `STRING`                    | The name of the *numerical* column to be correlated.
**`target_correlation_cols`** | **Required** | `STRING` or `ARRAY<STRING>` | The name of the numerical columns to correlate against the `target_col`.
**`dimension_cols`**          | Optional     | `STRING` or `ARRAY<STRING>` | A value that contains the names of columns to slice the data by. The function calculates correlations for every combination of these dimensions (Max: 12 columns). Supported Types: Any Groupable data type. Unsupported: ARRAY, STRUCT, JSON, GEOGRAPHY.
**`method`**                  | Optional     | `STRING`                    | The correlation method: `PEARSON` (default), `SPEARMAN`, or `KENDALL`. The `KENDALL` method has O(N^2) performance implications on massive tables. For large datasets, it is highly recommended to use `PEARSON` or `SPEARMAN`.

**Note on Reserved Column Names:** The column names used in arguments must not
overlap with the reserved output column names: `segment`, `target_col`,
`corr_col`, `correlation`, `segment_size`, `segment_proportion`.

### Output Schema

| Column                   | Type                        | Description         |
| :----------------------- | :-------------------------- | :------------------ |
| **`segment`**            | `ARRAY<STRUCT<dimension_col | Identifies the      |
:                          : STRING, dimension_value     : specific            :
:                          : JSON>>`                     : combination of      :
:                          :                             : dimension names and :
:                          :                             : values used to      :
:                          :                             : group the data for  :
:                          :                             : this row.           :
| *`<dimension_cols>`*     | Varies                      | Columns             |
:                          :                             : corresponding to    :
:                          :                             : each dimension in   :
:                          :                             : `dimension_cols`.   :
| **`target_col`**         | `STRING`                    | The name of the     |
:                          :                             : target column.      :
| **`corr_col`**           | `STRING`                    | The name of the     |
:                          :                             : column being        :
:                          :                             : correlated against  :
:                          :                             : the target.         :
| **`correlation`**        | `FLOAT64`                   | The calculated      |
:                          :                             : correlation         :
:                          :                             : coefficient.        :
| **`segment_size`**       | `INT64`                     | The number of rows  |
:                          :                             : in the segment.     :
| **`segment_proportion`** | `FLOAT64`                   | A value that        |
:                          :                             : contains the        :
:                          :                             : fraction of total   :
:                          :                             : rows in the input   :
:                          :                             : table that belong   :
:                          :                             : to this segment.    :

**Note:** Results are sorted by `segment_size` (descending) and then `corr_col`
(ascending) by default.

## Examples

The following examples show how to use the `ML.CORRELATION` function and use the
`my_dataset.marketing_sample` table:

```sql
CREATE OR REPLACE TABLE my_dataset.marketing_sample AS (
  -- New York data
  SELECT 'USA' AS country, 'New York' AS city, 'Electronics' AS product_category, 100 AS ad_spend, 150 AS budget, 1000 AS revenue UNION ALL
  SELECT 'USA', 'New York', 'Electronics', 150, 200, 1500 UNION ALL
  SELECT 'USA', 'New York', 'Apparel',     200, 250, 2000 UNION ALL

  -- Seattle data
  SELECT 'USA', 'Seattle',  'Apparel',     200, 250, 2000 UNION ALL
  SELECT 'USA', 'Seattle',  'Apparel',     300, 350, 3000 UNION ALL

  -- London data (Genuine NULL country)
  SELECT NULL,  'London',   'Electronics', 100, 120, 500  UNION ALL
  SELECT NULL,  'London',   'Electronics', 200, 220, 900  UNION ALL

  -- Missing city data (Genuine NULL city)
  SELECT NULL,  NULL,       'Apparel',     200, 200, 1000 UNION ALL
  SELECT NULL,  NULL,       'Apparel',     250, 250, 1200
);


/*---------+----------+------------------+----------+---------+---------+
 | country | city     | product_category | ad_spend | budget  | revenue |
 +---------+----------+------------------+----------+---------+---------+
 | USA     | New York | Electronics      | 100      | 150     | 1000    |
 | USA     | New York | Electronics      | 150      | 200     | 1500    |
 | USA     | New York | Apparel          | 200      | 250     | 2000    |
 | USA     | Seattle  | Apparel          | 200      | 250     | 2000    |
 | USA     | Seattle  | Apparel          | 300      | 350     | 3000    |
 | null    | London   | Electronics      | 100      | 120     | 500     |
 | null    | London   | Electronics      | 200      | 220     | 900     |
 | null    | null     | Apparel          | 200      | 200     | 1000    |
 | null    | null     | Apparel          | 250      | 250     | 1200    |
 +---------+----------+------------------+----------+---------+---------*/
```

### Calculate Pearson correlation

The following example calculates the Pearson correlation between `revenue` and
`ad_spend` from the table `my_dataset.marketing_sample` and uses `country` as a
dimension column:

```sql
SELECT
  country,
  segment,
  correlation,
  segment_size
FROM ML.CORRELATION(
  TABLE my_dataset.marketing_sample,
  target_col => 'revenue',
  target_correlation_cols => 'ad_spend',
  dimension_cols => ['country']
);

/*---------+---------------------------------------------------------------+-------------+--------------+
 | country | segment                                                       | correlation | segment_size |
 +---------+---------------------------------------------------------------+-------------+--------------+
 | 'USA'   | [{dimension_col: 'country', dimension_value: 'USA'}] | 1.0         | 2            |
 | NULL    | [{dimension_col: 'country', dimension_value: null}]  | 1.0         | 2            |
 | NULL    | []                                                            | 0.688       | 4            |
 +---------+---------------------------------------------------------------+-------------+--------------*/
```

The second row of the result corresponds to a genuine `NULL` in the input data
for country because the `dimension_value` field is `NULL`. The third row of the
result contains `NULL` for country because it corresponds to aggregation over
all countries.

### Calculate correlation for multiple columns

The following example calculates the correlation between `revenue` and
`ad_spend` and `budget`, sliced by `city` and `product_category`, from the table
`my_dataset.marketing_sample`:

```sql
SELECT *
FROM
ML.CORRELATION(
 (SELECT * FROM my_dataset.marketing_sample WHERE country = 'USA'),
 target_col => 'revenue',
 target_correlation_cols => ['ad_spend', 'budget'],
 dimension_cols => ['city', 'product_category']
);
```

### Distinguish between global aggregates and missing data

The following example shows how to use the `segment` column to label your rows
clearly in a report:

```sql
SELECT
  -- Create a clean label for reporting
  CASE
    -- If 'city' is NULL and not in the segment array, it's a total
    WHEN city IS NULL AND NOT EXISTS(SELECT 1 FROM UNNEST(segment) s WHERE s.dimension_col = 'city')
      THEN 'ALL CITIES (Global)'
    -- If 'city' is NULL and in the segment array, it's missing data
    WHEN city IS NULL
      THEN 'UNKNOWN CITY'
    ELSE city
  END AS city_label,
  correlation,
  segment_size
FROM ML.CORRELATION(
  TABLE my_dataset.marketing_sample,
  target_col => 'revenue',
  target_correlation_cols => 'ad_spend',
  dimension_cols => ['city']
);
```
