# Cloud AI Services in BigQuery

Guide to using Google Cloud AI services directly from BigQuery SQL.

## Overview

BigQuery integrates with Cloud AI services for:
- Translation
- Natural Language Processing
- Document Understanding
- Speech Transcription
- Computer Vision

## Translation (Cloud Translation API)

### Create Translation Model

```sql
CREATE OR REPLACE MODEL `project.dataset.translator`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (ENDPOINT = 'cloud_ai_translate_v3');
```

### ML.TRANSLATE Function

```sql
SELECT *
FROM ML.TRANSLATE(
  MODEL `project.dataset.translator`,
  (SELECT text AS text_to_translate FROM documents),
  STRUCT(
    'es' AS target_language_code,
    'en' AS source_language_code  -- Optional, auto-detected if omitted
  )
);
```

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `translated_text` | STRING | Translated content |
| `detected_language_code` | STRING | Detected source language |
| Original columns | Various | All input columns |

### Examples

```sql
-- Translate to multiple languages
SELECT
  original_text,
  es.translated_text AS spanish,
  fr.translated_text AS french,
  de.translated_text AS german
FROM (SELECT content AS text_to_translate, content AS original_text FROM articles) src
LEFT JOIN ML.TRANSLATE(MODEL `project.dataset.translator`, src,
  STRUCT('es' AS target_language_code)) es ON TRUE
LEFT JOIN ML.TRANSLATE(MODEL `project.dataset.translator`, src,
  STRUCT('fr' AS target_language_code)) fr ON TRUE
LEFT JOIN ML.TRANSLATE(MODEL `project.dataset.translator`, src,
  STRUCT('de' AS target_language_code)) de ON TRUE;
```

```sql
-- Detect language
SELECT
  text,
  detected_language_code AS language
FROM ML.TRANSLATE(
  MODEL `project.dataset.translator`,
  (SELECT text AS text_to_translate FROM unknown_language_texts),
  STRUCT('en' AS target_language_code)
);
```

## Natural Language (Cloud Natural Language API)

### Create NLP Model

```sql
CREATE OR REPLACE MODEL `project.dataset.nlp_analyzer`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (ENDPOINT = 'cloud_ai_natural_language_v1');
```

### ML.UNDERSTAND_TEXT Function

```sql
SELECT *
FROM ML.UNDERSTAND_TEXT(
  MODEL `project.dataset.nlp_analyzer`,
  (SELECT text AS text_content FROM reviews),
  STRUCT('ANALYZE_SENTIMENT' AS nlp_task)
);
```

### NLP Tasks

| Task | Description | Output |
|------|-------------|--------|
| `ANALYZE_SENTIMENT` | Sentiment analysis | Score, magnitude |
| `ANALYZE_ENTITIES` | Entity extraction | Entities with types |
| `ANALYZE_SYNTAX` | Syntax/grammar analysis | Tokens, POS tags |
| `CLASSIFY_TEXT` | Text classification | Categories |
| `ANALYZE_ENTITY_SENTIMENT` | Entity-level sentiment | Entities with sentiment |

### Examples

#### Sentiment Analysis

```sql
SELECT
  review_id,
  text,
  ml_understand_text_result.document_sentiment.score AS sentiment_score,
  ml_understand_text_result.document_sentiment.magnitude AS sentiment_magnitude,
  CASE
    WHEN ml_understand_text_result.document_sentiment.score > 0.25 THEN 'positive'
    WHEN ml_understand_text_result.document_sentiment.score < -0.25 THEN 'negative'
    ELSE 'neutral'
  END AS sentiment_label
FROM ML.UNDERSTAND_TEXT(
  MODEL `project.dataset.nlp_analyzer`,
  (SELECT review_id, review_text AS text_content FROM product_reviews),
  STRUCT('ANALYZE_SENTIMENT' AS nlp_task)
);
```

#### Entity Extraction

```sql
SELECT
  doc_id,
  entity.name AS entity_name,
  entity.type AS entity_type,
  entity.salience AS importance
FROM ML.UNDERSTAND_TEXT(
  MODEL `project.dataset.nlp_analyzer`,
  (SELECT doc_id, content AS text_content FROM documents),
  STRUCT('ANALYZE_ENTITIES' AS nlp_task)
),
UNNEST(ml_understand_text_result.entities) AS entity
WHERE entity.salience > 0.1;
```

#### Text Classification

```sql
SELECT
  article_id,
  category.name AS category,
  category.confidence
FROM ML.UNDERSTAND_TEXT(
  MODEL `project.dataset.nlp_analyzer`,
  (SELECT article_id, content AS text_content FROM articles),
  STRUCT('CLASSIFY_TEXT' AS nlp_task)
),
UNNEST(ml_understand_text_result.categories) AS category
ORDER BY category.confidence DESC;
```

## Document AI (Document Understanding)

### Create Document AI Model

```sql
CREATE OR REPLACE MODEL `project.dataset.doc_processor`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (
    ENDPOINT = 'cloud_ai_document_v1',
    DOCUMENT_PROCESSOR = 'projects/my-project/locations/us/processors/abc123'
  );
```

### ML.PROCESS_DOCUMENT Function

```sql
SELECT *
FROM ML.PROCESS_DOCUMENT(
  MODEL `project.dataset.doc_processor`,
  (SELECT gcs_uri AS uri FROM pdf_files)
);
```

### Processor Types

| Processor | Description | Use Case |
|-----------|-------------|----------|
| Form Parser | Extract form fields | Surveys, applications |
| Invoice Parser | Extract invoice data | Accounting |
| Receipt Parser | Extract receipt info | Expense tracking |
| ID Parser | Extract ID information | KYC |
| OCR | General text extraction | Digitization |

### Examples

#### Process Invoices

```sql
SELECT
  invoice_uri,
  JSON_VALUE(ml_process_document_result, '$.entities[?(@.type=="invoice_id")].mentionText') AS invoice_id,
  JSON_VALUE(ml_process_document_result, '$.entities[?(@.type=="total_amount")].mentionText') AS total,
  JSON_VALUE(ml_process_document_result, '$.entities[?(@.type=="invoice_date")].mentionText') AS date
FROM ML.PROCESS_DOCUMENT(
  MODEL `project.dataset.invoice_processor`,
  (SELECT uri AS uri FROM `project.dataset.invoice_pdfs`)
);
```

#### Extract Text from PDFs

```sql
SELECT
  pdf_uri,
  ml_process_document_result.text AS extracted_text,
  ARRAY_LENGTH(ml_process_document_result.pages) AS page_count
FROM ML.PROCESS_DOCUMENT(
  MODEL `project.dataset.ocr_processor`,
  (SELECT gcs_uri AS uri FROM document_archive)
);
```

## Speech-to-Text (Cloud Speech API)

### Create Speech Model

```sql
CREATE OR REPLACE MODEL `project.dataset.speech_transcriber`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (ENDPOINT = 'cloud_ai_speech_v2');
```

### ML.TRANSCRIBE Function

```sql
SELECT *
FROM ML.TRANSCRIBE(
  MODEL `project.dataset.speech_transcriber`,
  (SELECT audio_uri AS uri FROM audio_files),
  STRUCT(
    'en-US' AS language_code,
    TRUE AS enable_automatic_punctuation
  )
);
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `language_code` | STRING | Language (e.g., 'en-US', 'es-ES') |
| `enable_automatic_punctuation` | BOOL | Add punctuation |
| `enable_word_time_offsets` | BOOL | Include timestamps |
| `model` | STRING | 'latest_long', 'latest_short' |

### Examples

```sql
-- Transcribe call recordings
SELECT
  call_id,
  ml_transcribe_result.transcript AS transcription,
  ml_transcribe_result.confidence AS confidence
FROM ML.TRANSCRIBE(
  MODEL `project.dataset.speech_transcriber`,
  (SELECT call_id, recording_uri AS uri FROM call_recordings),
  STRUCT('en-US' AS language_code, TRUE AS enable_automatic_punctuation)
);

-- With word timestamps
SELECT
  audio_id,
  word.word,
  word.start_time,
  word.end_time
FROM ML.TRANSCRIBE(
  MODEL `project.dataset.speech_transcriber`,
  (SELECT audio_id, uri FROM audio_files),
  STRUCT('en-US' AS language_code, TRUE AS enable_word_time_offsets)
),
UNNEST(ml_transcribe_result.words) AS word;
```

## Computer Vision (Cloud Vision API)

### Create Vision Model

```sql
CREATE OR REPLACE MODEL `project.dataset.image_analyzer`
  REMOTE WITH CONNECTION `project.us.connection`
  OPTIONS (ENDPOINT = 'cloud_ai_vision_v1');
```

### ML.ANNOTATE_IMAGE Function

```sql
SELECT *
FROM ML.ANNOTATE_IMAGE(
  MODEL `project.dataset.image_analyzer`,
  (SELECT image_uri AS uri FROM images),
  STRUCT(['LABEL_DETECTION', 'TEXT_DETECTION'] AS vision_features)
);
```

### Vision Features

| Feature | Description |
|---------|-------------|
| `LABEL_DETECTION` | Identify objects and concepts |
| `TEXT_DETECTION` | OCR - extract text |
| `FACE_DETECTION` | Detect faces |
| `LANDMARK_DETECTION` | Identify landmarks |
| `LOGO_DETECTION` | Detect logos |
| `SAFE_SEARCH_DETECTION` | Content moderation |
| `IMAGE_PROPERTIES` | Color analysis |
| `OBJECT_LOCALIZATION` | Locate objects with bounding boxes |

### Examples

#### Label Detection

```sql
SELECT
  image_id,
  label.description AS label,
  label.score AS confidence
FROM ML.ANNOTATE_IMAGE(
  MODEL `project.dataset.image_analyzer`,
  (SELECT image_id, gcs_uri AS uri FROM product_images),
  STRUCT(['LABEL_DETECTION'] AS vision_features)
),
UNNEST(ml_annotate_image_result.label_annotations) AS label
WHERE label.score > 0.8;
```

#### Text Extraction (OCR)

```sql
SELECT
  image_id,
  ml_annotate_image_result.full_text_annotation.text AS extracted_text
FROM ML.ANNOTATE_IMAGE(
  MODEL `project.dataset.image_analyzer`,
  (SELECT image_id, uri FROM document_images),
  STRUCT(['TEXT_DETECTION'] AS vision_features)
);
```

#### Content Moderation

```sql
SELECT
  image_id,
  ml_annotate_image_result.safe_search_annotation.adult AS adult_rating,
  ml_annotate_image_result.safe_search_annotation.violence AS violence_rating,
  CASE
    WHEN ml_annotate_image_result.safe_search_annotation.adult IN ('LIKELY', 'VERY_LIKELY')
      OR ml_annotate_image_result.safe_search_annotation.violence IN ('LIKELY', 'VERY_LIKELY')
    THEN 'FLAGGED'
    ELSE 'SAFE'
  END AS moderation_status
FROM ML.ANNOTATE_IMAGE(
  MODEL `project.dataset.image_analyzer`,
  (SELECT image_id, uri FROM user_uploads),
  STRUCT(['SAFE_SEARCH_DETECTION'] AS vision_features)
);
```

#### Object Detection with Bounding Boxes

```sql
SELECT
  image_id,
  obj.name AS object_name,
  obj.score AS confidence,
  obj.bounding_poly.normalized_vertices AS bounding_box
FROM ML.ANNOTATE_IMAGE(
  MODEL `project.dataset.image_analyzer`,
  (SELECT image_id, uri FROM images),
  STRUCT(['OBJECT_LOCALIZATION'] AS vision_features)
),
UNNEST(ml_annotate_image_result.localized_object_annotations) AS obj
WHERE obj.score > 0.7;
```

## Best Practices

### 1. Batch Processing

Process data in batches to optimize costs and performance:

```sql
-- Process in batches of 1000
SELECT * FROM ML.TRANSLATE(
  MODEL `project.dataset.translator`,
  (SELECT * FROM documents LIMIT 1000 OFFSET @batch),
  ...
);
```

### 2. Error Handling

Always check for processing errors:

```sql
SELECT *
FROM ML.PROCESS_DOCUMENT(...)
WHERE ml_process_document_status = '';  -- Empty = success
```

### 3. Cost Management

- Filter data before processing
- Use appropriate features only
- Monitor usage in Cloud Console

### 4. Regional Considerations

- Use connections in same region as data
- Some services have regional restrictions
