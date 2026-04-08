# BigQuery AI.Score

The `AI.SCORE` function is commonly used with the ORDER BY clause and works well
when you want to rank items. The following are common use cases:

-   Retail: Find the top 5 most negative customer reviews about a product.
-   Hiring: Find the top 10 resumes that appear most qualified for a job post.
-   Customer success: Find the top 20 best customer support interactions.

## Syntax Reference

```sql
AI.SCORE(
  [ prompt => ] 'PROMPT'
  [, connection_id => 'CONNECTION_ID' ]
  [, endpoint => 'ENDPOINT' ]
)
```

### Input Arguments

| Argument            | Requirement  | Type          | Description            |
| :------------------ | :----------- | :------------ | :--------------------- |
| **`prompt`**        | **Required** | String/Struct | The prompt text or a   |
:                     :              :               : struct/tuple of        :
:                     :              :               : `(data, instruction)`. :
| **`connection_id`** | Optional     | String        | The connection ID to   |
:                     :              :               : use for the LLM.       :
| **`endpoint`**      | Optional     | String        | The model endpoint     |
:                     :              :               : (e.g.                  :
:                     :              :               : `'gemini-2.5-flash'`). :

### Output Schema

| Column Name         | Type      | Description                                |
| :------------------ | :-------- | :----------------------------------------- |
| **(Scalar Result)** | `FLOAT64` | A numerical score representing the degree  |
:                     :           : to which the data matches the instruction. :

## Examples

### Rank rows by semantic relevance

```sql
SELECT *
FROM `dataset.table`
ORDER BY AI.SCORE(
  (content_column, 'relevance to sports'),
  connection_id => 'my-project.us.my-connection'
) DESC
LIMIT 10;
```
