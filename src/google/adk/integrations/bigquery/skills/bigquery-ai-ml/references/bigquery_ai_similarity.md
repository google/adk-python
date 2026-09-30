# BigQuery AI.Similarity

`AI.SIMILARITY` computes the cosine similarity between two inputs

## Syntax Reference

```sql
AI.SIMILARITY(
  content1 => 'CONTENT1',
  content2 => 'CONTENT2'
  endpoint => 'ENDPOINT'
  [, model_params => 'MODEL_PARAMS']
  [, connection_id => 'CONNECTION_ID']
)
```

### Input Arguments

| Argument            | Requirement  | Type   | Description                   |
| :------------------ | :----------- | :----- | :---------------------------- |
| **`content1`**      | **Required** | String | The first text content.       |
| **`content2`**      | **Required** | String | The second text content to    |
:                     :              :        : compare against.              :
| **`connection_id`** | Optional     | String | The connection ID to use for  |
:                     :              :        : the LLM.                      :
| **`endpoint`**      | Optional     | String | The model endpoint (e.g.      |
:                     :              :        : `'multimodalembedding@001'`). :
| **`model_params`**  | Optional     | JSON   | JSON object for model         |
:                     :              :        : parameters (e.g.,             :
:                     :              :        : `temperature`,                :
:                     :              :        : `max_output_tokens`).         :

### Output Schema

| Column Name         | Type      | Description                         |
| :------------------ | :-------- | :---------------------------------- |
| **(Scalar Result)** | `FLOAT64` | A similarity score (e.g., cosine    |
:                     :           : similarity). Returns null if error. :

## Examples

```sql
SELECT AI.SIMILARITY(
  content1 => 'The cat sat on the mat',
  content2 => 'A feline is resting on the rug',
  endpoint => 'text-embedding-005'
) as similarity_score;
```
