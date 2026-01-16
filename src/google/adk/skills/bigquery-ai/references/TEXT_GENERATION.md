# Text Generation in BigQuery

Complete guide to generating text using AI.GENERATE_TEXT and ML.GENERATE_TEXT functions.

## Function Overview

BigQuery provides two primary functions for text generation:

| Function | Description | Use Case |
|----------|-------------|----------|
| `AI.GENERATE_TEXT` | Table function with full parameter control | Complex generation tasks |
| `ML.GENERATE_TEXT` | Scalar function for simpler use | Single-row or inline generation |

## AI.GENERATE_TEXT Syntax

```sql
SELECT *
FROM AI.GENERATE_TEXT(
  MODEL `project.dataset.model_name`,
  { TABLE source_table | (SELECT query) },
  STRUCT(
    max_output_tokens AS max_output_tokens,
    temperature AS temperature,
    top_p AS top_p,
    top_k AS top_k,
    stop_sequences AS stop_sequences,
    ground_with_google_search AS ground_with_google_search,
    safety_settings AS safety_settings,
    request_type AS request_type
  )
);
```

## Parameters Reference

### Generation Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `max_output_tokens` | INT64 | 1-8192 | 128 | Maximum tokens in response |
| `temperature` | FLOAT64 | 0.0-2.0 | 0.0 | Randomness (0=deterministic, higher=creative) |
| `top_p` | FLOAT64 | 0.0-1.0 | 0.95 | Nucleus sampling probability |
| `top_k` | INT64 | 1-40 | 40 | Top-k token selection |
| `stop_sequences` | ARRAY<STRING> | - | [] | Strings that stop generation |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ground_with_google_search` | BOOL | FALSE | Enable web grounding for factual responses |
| `request_type` | STRING | UNSPECIFIED | Quota type: DEDICATED, SHARED, UNSPECIFIED |
| `safety_settings` | ARRAY<STRUCT> | - | Content filtering configuration |

### Safety Settings

Configure content filtering with category and threshold pairs:

```sql
STRUCT(
  [STRUCT('HARM_CATEGORY_HATE_SPEECH' AS category, 'BLOCK_LOW_AND_ABOVE' AS threshold),
   STRUCT('HARM_CATEGORY_DANGEROUS_CONTENT' AS category, 'BLOCK_MEDIUM_AND_ABOVE' AS threshold)]
  AS safety_settings
)
```

**Harm Categories:**
- `HARM_CATEGORY_HATE_SPEECH`
- `HARM_CATEGORY_DANGEROUS_CONTENT`
- `HARM_CATEGORY_HARASSMENT`
- `HARM_CATEGORY_SEXUALLY_EXPLICIT`

**Thresholds:**
- `BLOCK_NONE` (requires allowlisting)
- `BLOCK_LOW_AND_ABOVE`
- `BLOCK_MEDIUM_AND_ABOVE` (default)
- `BLOCK_ONLY_HIGH`

## Output Schema

The function returns these columns:

| Column | Type | Description |
|--------|------|-------------|
| `ml_generate_text_result` | JSON | Generated text and metadata |
| `ml_generate_text_status` | STRING | Error message (empty if success) |
| Original columns | Various | All columns from input table |

### Parsing Results

```sql
SELECT
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS generated_text,
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].safetyAttributes.blocked') AS was_blocked,
  CAST(JSON_VALUE(ml_generate_text_result, '$.tokenMetadata.outputTokenCount.totalTokens') AS INT64) AS output_tokens
FROM ML.GENERATE_TEXT(...);
```

## Examples

### Basic Text Generation

```sql
SELECT
  id,
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS summary
FROM ML.GENERATE_TEXT(
  MODEL `myproject.mydataset.gemini_model`,
  (SELECT id, CONCAT('Summarize in 2 sentences: ', article_text) AS prompt
   FROM `myproject.mydataset.articles`
   WHERE date = CURRENT_DATE()),
  STRUCT(150 AS max_output_tokens, 0.3 AS temperature)
);
```

### Creative Writing with Higher Temperature

```sql
SELECT *
FROM AI.GENERATE_TEXT(
  MODEL `myproject.mydataset.gemini_pro`,
  (SELECT CONCAT('Write a creative tagline for: ', product_name) AS prompt
   FROM products),
  STRUCT(
    50 AS max_output_tokens,
    0.9 AS temperature,
    0.95 AS top_p
  )
);
```

### Factual Q&A with Web Grounding

```sql
SELECT
  question,
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS answer
FROM ML.GENERATE_TEXT(
  MODEL `myproject.mydataset.gemini_model`,
  (SELECT question, CONCAT('Answer factually: ', question) AS prompt
   FROM questions),
  STRUCT(
    256 AS max_output_tokens,
    0.0 AS temperature,
    TRUE AS ground_with_google_search
  )
);
```

### Multi-turn Conversation

```sql
SELECT *
FROM AI.GENERATE_TEXT(
  MODEL `myproject.mydataset.gemini_model`,
  (SELECT
    CONCAT(
      'Previous conversation:\n',
      conversation_history,
      '\n\nUser: ', user_message,
      '\n\nAssistant:'
    ) AS prompt
   FROM conversations),
  STRUCT(512 AS max_output_tokens, 0.7 AS temperature)
);
```

### Structured Output (JSON)

```sql
SELECT
  id,
  JSON_VALUE(ml_generate_text_result, '$.predictions[0].content') AS extracted_json
FROM ML.GENERATE_TEXT(
  MODEL `myproject.mydataset.gemini_model`,
  (SELECT id,
    CONCAT(
      'Extract entities as JSON with keys: name, date, amount\n\n',
      'Text: ', document_text,
      '\n\nJSON:'
    ) AS prompt
   FROM documents),
  STRUCT(
    200 AS max_output_tokens,
    0.0 AS temperature,
    ['```'] AS stop_sequences
  )
);
```

### Batch Classification

```sql
SELECT
  id,
  content,
  TRIM(JSON_VALUE(ml_generate_text_result, '$.predictions[0].content')) AS category
FROM ML.GENERATE_TEXT(
  MODEL `myproject.mydataset.gemini_flash`,
  (SELECT id, content,
    CONCAT(
      'Classify the following text into exactly one category: ',
      'Technology, Sports, Politics, Entertainment, Business\n\n',
      'Text: ', content, '\n\nCategory:'
    ) AS prompt
   FROM articles
   WHERE published_date > DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)),
  STRUCT(10 AS max_output_tokens, 0.0 AS temperature)
);
```

## Error Handling

### Check for Errors

```sql
SELECT
  id,
  CASE
    WHEN ml_generate_text_status != '' THEN CONCAT('ERROR: ', ml_generate_text_status)
    ELSE JSON_VALUE(ml_generate_text_result, '$.predictions[0].content')
  END AS result
FROM ML.GENERATE_TEXT(...);
```

### Filter Successful Results

```sql
SELECT *
FROM ML.GENERATE_TEXT(...)
WHERE ml_generate_text_status = '';
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `RESOURCE_EXHAUSTED` | Rate limit exceeded | Reduce batch size, add delays |
| `INVALID_ARGUMENT` | Bad prompt or parameters | Check prompt format, parameter ranges |
| `PERMISSION_DENIED` | Missing IAM roles | Grant `aiplatform.user` role |
| `BLOCKED` | Safety filter triggered | Adjust safety settings or modify prompt |

## Performance Optimization

### Batch Processing

```sql
-- Process in batches of 1000
SELECT * FROM ML.GENERATE_TEXT(
  MODEL `myproject.mydataset.gemini_flash`,
  (SELECT * FROM source_table LIMIT 1000 OFFSET @batch_offset),
  STRUCT(100 AS max_output_tokens)
);
```

### Use Appropriate Model

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| Gemini Flash | Fast | Good | Low | High-volume, simple tasks |
| Gemini Pro | Medium | High | Medium | Complex reasoning |
| Gemini Ultra | Slow | Highest | High | Research, critical tasks |

### Minimize Token Usage

1. Keep prompts concise
2. Set appropriate `max_output_tokens`
3. Use `stop_sequences` to cut off unnecessary output
4. Filter data before calling AI functions

## Cost Considerations

- Charged per 1000 characters (input + output)
- Different rates for different models
- Web grounding adds additional cost
- Monitor usage in Cloud Console > BigQuery > Quotas

## Supported Models

### Google Models
- `gemini-2.0-flash` - Latest, fastest
- `gemini-1.5-pro` - High quality
- `gemini-1.5-flash` - Balanced
- `gemini-1.0-pro` - Legacy

### Partner Models
- `claude-3-5-sonnet` - Anthropic
- `claude-3-opus` - Anthropic (highest quality)
- `llama-3.1-405b` - Meta
- `mistral-large` - Mistral AI

See `REMOTE_MODELS.md` for model connection setup.
