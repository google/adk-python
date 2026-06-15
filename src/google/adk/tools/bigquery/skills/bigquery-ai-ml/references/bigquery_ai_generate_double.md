# BigQuery AI.Generate_Double

`AI.GENERATE_DOUBLE` generates a floating-point number based on the prompt.

## Syntax Reference

```sql
AI.GENERATE_DOUBLE(
  [ prompt => ] 'PROMPT'
  [, connection_id => 'CONNECTION_ID' ]
  [, model_params => 'MODEL_PARAMS']
  [, endpoint => 'ENDPOINT' ]
  [, request_type => 'REQUEST_TYPE']
)
```

### Input Arguments

| Argument            | Requirement  | Type   | Description            |
| :------------------ | :----------- | :----- | :--------------------- |
| **`prompt`**        | **Required** | String | The prompt text or     |
:                     :              :        : instruction.           :
| **`connection_id`** | Optional     | String | The connection ID to   |
:                     :              :        : use for the LLM.       :
| **`endpoint`**      | Optional     | String | The model endpoint     |
:                     :              :        : (e.g.                  :
:                     :              :        : `'gemini-2.5-flash'`). :
| **`model_params`**  | Optional     | JSON   | JSON object for model  |
:                     :              :        : parameters (e.g.,      :
:                     :              :        : `temperature`,         :
:                     :              :        : `max_output_tokens`).  :
| **`request_type`**  | Optional     | String | `'DEDICATED'` or       |
:                     :              :        : `'SHARED'`.            :

### Output Schema

Column Name         | Type      | Description
:------------------ | :-------- | :--------------------------------------
**`result`**        | `FLOAT64` | The generated floating-point value.
**`status`**        | `STRING`  | API response status (empty on success).
**`full_response`** | `JSON`    | The complete raw JSON response.

## Examples

```sql
SELECT AI.GENERATE_DOUBLE(
  'What is the total price mentioned in this text? ' || text_content
) as total_price
FROM `dataset.receipts`;
```
