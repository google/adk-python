# BigQuery AI.Generate

`AI.GENERATE` is a general-purpose function text and content generation.

## Syntax Reference

```sql
AI.GENERATE(
  [ prompt => ] 'PROMPT',
  [, endpoint => 'ENDPOINT']
  [, model_params => 'MODEL_PARAMS']
  [, output_schema => 'OUTPUT_SCHEMA']
  [, connection_id => 'CONNECTION_ID']
  [, request_type => 'REQUEST_TYPE']
)
```

### Input Arguments

| Argument            | Requirement  | Type   | Description           |
| :------------------ | :----------- | :----- | :-------------------- |
| **`prompt`**        | **Required** | String | The prompt text or    |
:                     :              :        : instruction for the   :
:                     :              :        : model.                :
| **`connection_id`** | Optional     | String | The connection ID.    |
:                     :              :        : Optional if           :
:                     :              :        : configured via other  :
:                     :              :        : means or testing.     :
| **`endpoint`**      | Optional     | String | The model name, e.g., |
:                     :              :        : `'gemini-2.5-flash'`. :
| **`output_schema`** | Optional     | String | Schema definition for |
:                     :              :        : structured output,    :
:                     :              :        : e.g., `'answer BOOL,  :
:                     :              :        : reason STRING'`.      :
| **`request_type`**  | Optional     | String | `'DEDICATED'` or      |
:                     :              :        : `'SHARED'`.           :
| **`model_params`**  | Optional     | JSON   | JSON object for model |
:                     :              :        : parameters (e.g.,     :
:                     :              :        : `temperature`,        :
:                     :              :        : `max_output_tokens`). :

### Output Schema

Returns a `STRUCT` with the following fields:

| Column Name         | Type                 | Description                    |
| :------------------ | :------------------- | :----------------------------- |
| **`result`**        | `STRING` (or Custom) | The generated content. If      |
:                     :                      : `output_schema` is used, this  :
:                     :                      : field is replaced by the       :
:                     :                      : schema's fields.               :
| **`status`**        | `STRING`             | API response status (empty on  |
:                     :                      : success).                      :
| **`full_response`** | `JSON`               | The complete raw JSON response |
:                     :                      : from the model (including      :
:                     :                      : safety ratings, usage          :
:                     :                      : metadata).                     :

## Examples

### Basic Text Generation

```sql
SELECT
  AI.GENERATE(
    'Summarize this article: ' || article_content,
    connection_id => 'my-project.us.my-connection',
    endpoint => 'gemini-2.5-flash'
  ) as summary
FROM `dataset.articles`
LIMIT 5;
```

### Structured Output Generation

```sql
SELECT
  AI.GENERATE(
    'Extract the date and amount from this invoice: ' || invoice_text,
    output_schema => 'date DATE, amount FLOAT64'
  ) as extracted_data
FROM `dataset.invoices`;
```

### Process images in a Cloud Storage bucket

```
CREATE SCHEMA IF NOT EXISTS bqml_tutorial;

CREATE OR REPLACE EXTERNAL TABLE bqml_tutorial.product_images
  WITH CONNECTION DEFAULT OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://cloud-samples-data/bigquery/tutorials/cymbal-pets/images/*.png']);

SELECT
  uri,
  STRING(OBJ.GET_ACCESS_URL(ref,'r').access_urls.read_url) AS signed_url,
  AI.GENERATE(
    ("What is this: ", OBJ.GET_ACCESS_URL(ref, 'r')),
    output_schema =>
      "image_description STRING, entities_in_the_image ARRAY<STRING>").*
FROM bqml_tutorial.product_images
WHERE uri LIKE "%aquarium%";
```

### Using Grounding

```
SELECT
  name,
  AI.GENERATE(
    ('Please check the weather of ', name, ' for today.'),
    model_params => JSON '{"tools": [{"googleSearch": {}}]}'
  )
FROM UNNEST(['Seattle', 'NYC', 'Austin']) AS name;
```
