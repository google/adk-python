# BigQuery AI Functions Reference

## ML.GENERATE_TEXT

Generate text using an LLM remote model.

### Syntax
```sql
ML.GENERATE_TEXT(
  MODEL model_name,
  { TABLE table_name | (query_statement) },
  STRUCT(
    [temperature AS temperature],
    [max_output_tokens AS max_output_tokens],
    [top_k AS top_k],
    [top_p AS top_p],
    [flatten_json_output AS flatten_json_output],
    [stop_sequences AS stop_sequences]
  )
)
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| temperature | FLOAT64 | 0.0 | Randomness (0.0-1.0) |
| max_output_tokens | INT64 | 1024 | Maximum response length |
| top_k | INT64 | 40 | Top-k sampling |
| top_p | FLOAT64 | 0.95 | Nucleus sampling |
| flatten_json_output | BOOL | FALSE | Flatten output structure |
| stop_sequences | ARRAY<STRING> | [] | Stop generation sequences |

### Example
```sql
SELECT
  prompt,
  ml_generate_text_result['candidates'][0]['content']['parts'][0]['text'] AS response,
  ml_generate_text_status AS status
FROM ML.GENERATE_TEXT(
  MODEL `project.dataset.gemini_model`,
  (SELECT CONCAT('Classify sentiment: ', review_text) AS prompt FROM reviews),
  STRUCT(0.1 AS temperature, 100 AS max_output_tokens)
)
```

## ML.GENERATE_EMBEDDING

Generate vector embeddings for text or multimodal content.

### Syntax
```sql
ML.GENERATE_EMBEDDING(
  MODEL model_name,
  { TABLE table_name | (query_statement) },
  STRUCT(
    [flatten_json_output AS flatten_json_output],
    [output_dimensionality AS output_dimensionality]
  )
)
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| flatten_json_output | BOOL | FALSE | Return as flat array |
| output_dimensionality | INT64 | Model default | Truncate embedding dimensions |

### Example
```sql
SELECT
  text_content,
  ml_generate_embedding_result['embeddings'][0]['values'] AS embedding
FROM ML.GENERATE_EMBEDDING(
  MODEL `project.dataset.embedding_model`,
  (SELECT content AS text_content FROM documents),
  STRUCT(TRUE AS flatten_json_output)
)
```

## ML.UNDERSTAND_TEXT

Analyze text for entities, sentiment, and syntax.

### Syntax
```sql
ML.UNDERSTAND_TEXT(
  MODEL model_name,
  { TABLE table_name | (query_statement) },
  STRUCT(
    [classify_content_types AS classify_content_types],
    [extract_entities AS extract_entities],
    [analyze_sentiment AS analyze_sentiment],
    [analyze_syntax AS analyze_syntax]
  )
)
```

### Example
```sql
SELECT
  text,
  ml_understand_text_result['entities'] AS entities,
  ml_understand_text_result['document_sentiment']['score'] AS sentiment_score
FROM ML.UNDERSTAND_TEXT(
  MODEL `project.dataset.nlp_model`,
  (SELECT comment AS text FROM feedback),
  STRUCT(TRUE AS extract_entities, TRUE AS analyze_sentiment)
)
```

## ML.TRANSLATE

Translate text between languages.

### Syntax
```sql
ML.TRANSLATE(
  MODEL model_name,
  { TABLE table_name | (query_statement) },
  STRUCT(
    source_language_code AS source_lang,
    target_language_code AS target_lang
  )
)
```

### Example
```sql
SELECT
  original_text,
  ml_translate_result['translatedText'] AS translated
FROM ML.TRANSLATE(
  MODEL `project.dataset.translate_model`,
  (SELECT content AS text FROM articles WHERE lang = 'es'),
  STRUCT('es' AS source_lang, 'en' AS target_lang)
)
```

## Vector Search Functions

### VECTOR_SEARCH
```sql
SELECT query.id, base.id, distance
FROM VECTOR_SEARCH(
  (SELECT id, embedding FROM base_table),
  'embedding',
  (SELECT id, embedding FROM query_table),
  'embedding',
  top_k => 10,
  distance_type => 'COSINE'
)
```

### ML.DISTANCE
```sql
SELECT
  ML.DISTANCE(embedding1, embedding2, 'COSINE') AS cosine_distance,
  ML.DISTANCE(embedding1, embedding2, 'EUCLIDEAN') AS euclidean_distance
FROM embeddings_table
```

## Output Structure

### generate_text Output
```json
{
  "candidates": [{
    "content": {
      "parts": [{"text": "Generated response..."}],
      "role": "model"
    },
    "finishReason": "STOP",
    "safetyRatings": [...]
  }],
  "usageMetadata": {
    "promptTokenCount": 10,
    "candidatesTokenCount": 50
  }
}
```

### generate_embedding Output
```json
{
  "embeddings": [{
    "values": [0.123, -0.456, ...],
    "statistics": {
      "truncated": false,
      "token_count": 5
    }
  }]
}
```
