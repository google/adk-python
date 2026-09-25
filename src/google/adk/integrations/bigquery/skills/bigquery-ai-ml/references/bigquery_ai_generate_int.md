# BigQuery AI.Generate_Int

`AI.GENERATE_INT` generates an integer value based on the prompt.

## Syntax Reference

```sql
AI.GENERATE_INT(
  [ prompt => ] 'PROMPT'
  [, connection_id => 'CONNECTION_ID' ]
  [, endpoint => 'ENDPOINT' ]
  [, request_type => 'REQUEST_TYPE']
  [, model_params => 'MODEL_PARAMS']
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

Column Name         | Type     | Description
:------------------ | :------- | :--------------------------------------
**`result`**        | `INT64`  | The generated integer value.
**`status`**        | `STRING` | API response status (empty on success).
**`full_response`** | `JSON`   | The complete raw JSON response.

## Examples

```sql
SELECT AI.GENERATE_INT(
  'How many items are in this list? ' || list_content
) as item_count
FROM `dataset.inventory`;
```
