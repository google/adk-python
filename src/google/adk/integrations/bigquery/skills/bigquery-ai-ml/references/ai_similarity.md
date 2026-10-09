# BigQuery AI.Similarity

`AI.SIMILARITY` computes the semantic similarity between two inputs.

Use cases include the following:

-   Semantic search: Search for text or images based off a description, without
    having to match specific keywords.
-   Recommendation: Return entities with attributes similar to a given entity.

## Syntax Reference

```sql
AI.SIMILARITY(
  content1 => 'CONTENT1',
  content2 => 'CONTENT2'
  [, endpoint => 'ENDPOINT']
  [, model_params => 'MODEL_PARAMS']
  [, connection_id => 'CONNECTION_ID']
)
```

### Input Arguments

| Argument            | Requirement  | Type      | Description              |
| :------------------ | :----------- | :-------- | :----------------------- |
| **`content1`**      | **Required** | String or | The first text content   |
:                     :              : ObjectRef : or image context.        :
| **`content2`**      | **Required** | String or | The second text content  |
:                     :              : ObjectRef : or image to compare      :
:                     :              :           : against.                 :
| **`connection_id`** | Optional     | String    | The connection ID to use |
:                     :              :           : for the LLM.             :
| **`endpoint`**      | Optional     | String    | The model endpoint (e.g. |
:                     :              :           : `'text-embedding-005'`). :
| **`model_params`**  | Optional     | JSON      | JSON object for model    |
:                     :              :           : parameters (e.g.,        :
:                     :              :           : `temperature`,           :
:                     :              :           : `max_output_tokens`).    :

### Output Schema

| Column Name         | Type      | Description                         |
| :------------------ | :-------- | :---------------------------------- |
| **(Scalar Result)** | `FLOAT64` | A similarity score (e.g., cosine    |
:                     :           : similarity). Returns null if error. :

## Examples

### Compute semantic similarity between two text inputs

```sql
SELECT AI.SIMILARITY(
  content1 => 'The cat sat on the mat',
  content2 => 'A feline is resting on the rug'
) as similarity_score;
```

### Compute semantic similarity between text and image

```sql
CREATE SCHEMA IF NOT EXISTS cymbal_pets;

CREATE OR REPLACE EXTERNAL TABLE cymbal_pets.product_images
WITH CONNECTION DEFAULT
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/images/*.png']
);

SELECT
  uri,
  OBJ.GET_READ_URL(ref) AS signed_url,
  ai.similarity(
    "aquarium device",
    ref,
    endpoint => 'multimodalembedding@001') AS similarity_score
FROM cymbal_pets.product_images
ORDER BY similarity_score DESC
LIMIT 3;
```
