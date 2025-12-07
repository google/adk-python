---
name: bq_ai_operator
description: BigQuery AI Operator - Use managed AI functions (AI.CLASSIFY, AI.IF, AI.SCORE) directly in SQL for text classification, filtering, and scoring. Requires a BigQuery connection to Vertex AI.
---

# BQ AI Operator Skill (Managed AI Functions in SQL)

Use managed AI functions directly in BigQuery SQL queries for text classification, filtering, and scoring.

**IMPORTANT**: These are the NEW managed AI functions that require a `connection_id` to Vertex AI, NOT the older `ML.GENERATE_TEXT` style functions.

## Prerequisites

1. **Create a BigQuery connection to Vertex AI** (required for all AI functions):
```sql
-- Create a connection (run once)
CREATE CLOUD RESOURCE CONNECTION `us.my_ai_connection`
OPTIONS(location='us');
```

2. **Grant the connection service account access to Vertex AI**

## Available Managed AI Functions

| Function | Purpose | Return Type |
|----------|---------|-------------|
| AI.CLASSIFY | Categorize text into classes | STRING |
| AI.IF | Natural language TRUE/FALSE filtering | BOOL |
| AI.SCORE | Rate/rank by criteria (0.0 to 1.0) | FLOAT64 |

---

## AI.CLASSIFY - Categorize Text

Classify text into one of the provided categories.

### Syntax
```sql
AI.CLASSIFY(
  input,                          -- STRING: the text to classify
  categories => ['cat1', 'cat2'], -- ARRAY<STRING>: possible categories
  connection_id => 'LOCATION.CONNECTION_NAME'
)
```

### Examples

**News article classification:**
```sql
SELECT
    title,
    body,
    AI.CLASSIFY(
        body,
        categories => ['tech', 'sport', 'business', 'politics', 'entertainment', 'other'],
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS category
FROM `bigquery-public-data.bbc_news.fulltext`
LIMIT 10;
```

**Sentiment classification with descriptions:**
```sql
SELECT
    review_text,
    AI.CLASSIFY(
        review_text,
        categories => [
            ('positive', 'happy, satisfied, recommends'),
            ('negative', 'unhappy, disappointed, complaints'),
            ('neutral', 'factual, no strong emotion')
        ],
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS sentiment
FROM `project.dataset.reviews`
LIMIT 10;
```

---

## AI.IF - Natural Language Filtering

Returns TRUE or FALSE based on a natural language condition.

### Syntax
```sql
AI.IF(
  input,                    -- STRING: the text to evaluate
  condition,                -- STRING: natural language condition
  connection_id => 'LOCATION.CONNECTION_NAME'
)
```

### Examples

**Filter for eco-friendly products:**
```sql
SELECT product_name, description
FROM `project.products.catalog`
WHERE AI.IF(
    description,
    'This product is eco-friendly, sustainable, or environmentally conscious',
    connection_id => 'us.my_ai_connection'  -- Use your connection: test-project-0728-467323.us.my_ai_connection
) = TRUE
LIMIT 10;
```

**Content moderation:**
```sql
SELECT
    post_id,
    content,
    AI.IF(
        content,
        'This content is appropriate for all ages and contains no spam, harassment, or explicit material',
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS is_appropriate
FROM `project.social.user_posts`
LIMIT 10;
```

---

## AI.SCORE - Quality Scoring

Returns a score between 0.0 and 1.0 based on criteria.

### Syntax
```sql
AI.SCORE(
  input,                    -- STRING: the text to score
  criteria,                 -- STRING: scoring criteria
  connection_id => 'LOCATION.CONNECTION_NAME'
)
```

### Examples

**Review helpfulness scoring:**
```sql
SELECT
    review_id,
    review_text,
    star_rating,
    AI.SCORE(
        review_text,
        'Rate this review helpfulness based on: detail level, specific examples, balanced perspective',
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS helpfulness_score
FROM `project.reviews.product_reviews`
ORDER BY helpfulness_score DESC
LIMIT 10;
```

**Relevance scoring:**
```sql
SELECT
    document_id,
    title,
    AI.SCORE(
        content,
        'How relevant is this document to machine learning and AI topics',
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS ml_relevance
FROM `project.docs.articles`
ORDER BY ml_relevance DESC
LIMIT 10;
```

---

## Complete Pipeline Example

Combine multiple AI functions for a review intelligence pipeline:

```sql
-- Step 1: Classify and score reviews
WITH classified AS (
    SELECT
        review_id,
        review_text,
        star_rating,
        AI.CLASSIFY(
            review_text,
            categories => ['positive', 'negative', 'neutral', 'mixed'],
            connection_id => 'us.my_ai_connection'  -- Replace with your connection
        ) AS sentiment,
        AI.SCORE(
            review_text,
            'Review quality based on detail and helpfulness',
            connection_id => 'us.my_ai_connection'  -- Replace with your connection
        ) AS quality_score
    FROM `project.reviews.raw_reviews`
    LIMIT 100
)
-- Step 2: Filter appropriate content and categorize
SELECT
    review_id,
    sentiment,
    quality_score,
    CASE
        WHEN quality_score >= 0.8 THEN 'featured'
        WHEN quality_score >= 0.5 THEN 'standard'
        ELSE 'low_quality'
    END AS tier
FROM classified
WHERE AI.IF(
    review_text,
    'Content is appropriate and not spam',
    connection_id => 'us.my_ai_connection'  -- Use your connection: test-project-0728-467323.us.my_ai_connection
) = TRUE
ORDER BY quality_score DESC;
```

---

## Important Notes

1. **Connection Required**: All managed AI functions require a `connection_id` to a Vertex AI connection
2. **Preview Feature**: AI.CLASSIFY, AI.IF, and AI.SCORE are in public preview
3. **Region Support**: Works in all Gemini regions plus US/EU multi-regions
4. **Use LIMIT**: Always use LIMIT to control costs when testing
5. **String Return**: AI.CLASSIFY returns STRING, AI.IF returns BOOL, AI.SCORE returns FLOAT64

## Troubleshooting

**Error: "connection not found"**
- Ensure you've created the connection: `CREATE CLOUD RESOURCE CONNECTION`
- Use the correct format: `LOCATION.CONNECTION_NAME` (e.g., `us.my_ai_connection`)

**Error: "permission denied"**
- Grant the connection's service account access to Vertex AI API
