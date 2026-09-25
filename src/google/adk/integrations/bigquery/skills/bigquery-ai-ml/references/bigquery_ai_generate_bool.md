# BigQuery AI.Generate_Bool

`AI.GENERATE_BOOL` generates a boolean value (`TRUE` or `FALSE`) based on the
prompt.

## Syntax Reference

```sql
AI.GENERATE_BOOL(
  [ prompt => ] 'PROMPT'
  [, connection_id => 'CONNECTION_ID' ]
  [, endpoint => 'ENDPOINT' ]
  [, model_params => 'MODEL_PARAMS']
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

Column Name         | Type     | Description
:------------------ | :------- | :--------------------------------------
**`result`**        | `BOOL`   | The generated boolean value.
**`status`**        | `STRING` | API response status (empty on success).
**`full_response`** | `JSON`   | The complete raw JSON response.

## Examples

```sql
SELECT AI.GENERATE_BOOL(
  'Is this a valid email address? ' || email_address
) as is_valid
FROM `dataset.users`;
```
