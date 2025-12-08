---
name: bq_ai_operator
description: BigQuery AI Operator - Use managed AI functions (AI.CLASSIFY, AI.IF, AI.SCORE) directly in SQL for text classification, filtering, and scoring. Requires a BigQuery connection to Vertex AI.
keywords:
  - ai.classify
  - ai.if
  - ai.score
  - ai function
  - ai operator
  - text classification
  - sentiment
  - categorize
  - categories
  - natural language
  - filter text
  - score text
  - score
  - rate content
  - rank content
  - rate
  - rank
  - classify
  - positive
  - negative
  - vertex ai
  - managed ai
  - list connections
  - connection_id
---

# BQ AI Operator Skill (Managed AI Functions in SQL)

Use managed AI functions directly in BigQuery SQL queries for text classification, filtering, and scoring.

**IMPORTANT**: These are the NEW managed AI functions that require a `connection_id` to Vertex AI, NOT the older `ML.GENERATE_TEXT` style functions.

## Prerequisites

1. **A BigQuery connection to Vertex AI is required** for all AI functions.

2. **Grant the connection service account access to Vertex AI**

## Connection Workflow (ALWAYS Follow This)

**CRITICAL**: AI functions require a `connection_id` to a BigQuery connection to Vertex AI.

### ⚠️ IMPORTANT: Location Matching Rule

**The connection location MUST match your dataset location!**

| Dataset Location | Connection Location | Example |
|------------------|---------------------|---------|
| `US` (multi-region) | `us` | `us.my_ai_connection` |
| `EU` (multi-region) | `eu` | `eu.my_ai_connection` |
| `us-central1` (regional) | `us-central1` | `us-central1.my_ai_connection` |

**Common Error**: Using `us-central1.my_connection` with a dataset in `US` multi-region will fail with "Dataset not found in location us-central1".

**How to check dataset location**:
```sql
SELECT option_value FROM `project.dataset.INFORMATION_SCHEMA.SCHEMATA_OPTIONS` WHERE option_name = 'location'
```

### Step 1: Determine Your Dataset Location

Before listing connections, identify where your target dataset is located:
- Most BigQuery public datasets are in `US` multi-region
- Your own datasets might be in `US`, `EU`, or a specific region like `us-central1`

### Step 2: List Connections in the SAME Location

Use the `list_connections` tool with the **same location as your dataset**:

```
# For datasets in US multi-region:
list_connections(project_id="your-project", location="us")

# For datasets in us-central1:
list_connections(project_id="your-project", location="us-central1")
```

This returns all available connections with their `connection_id` and `service_account`.

### Step 3: Use an Existing Connection If Available

If `list_connections` returns connections, **use one of them**. Pick a connection that:
- Has `connection_type: "CLOUD_RESOURCE"` (required for Vertex AI)
- Is in the **SAME location as your dataset**

Use the `connection_id` from the result, formatted as `location.connection_id`:
- Example: If connection_id is `my_ai_connection` in location `us`, use `us.my_ai_connection`

### Step 4: Only Create a New Connection If None Exist

**Only if `list_connections` returns empty or no suitable connections**, create a new one in the **same location as your dataset**:

```
# For US multi-region datasets:
create_connection(project_id="your-project", location="us", connection_id="my_ai_connection")

# For us-central1 datasets:
create_connection(project_id="your-project", location="us-central1", connection_id="my_ai_connection")
```

This automatically:
1. Creates the connection
2. Grants the Vertex AI User role to the service account (required for AI functions)

### Connection ID Formats

When using connections in SQL:
- `us.my_connection` (location.connection_name) - **Preferred for US multi-region**
- `us-central1.my_connection` - **For regional datasets**
- `project_id.us.my_connection` (fully qualified)

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

Returns a FLOAT64 score based on your scoring criteria. Commonly used with ORDER BY for ranking.

### Syntax
```sql
AI.SCORE(
  (prompt_with_criteria, column_to_score),  -- TUPLE: (STRING literal, column reference)
  connection_id => 'LOCATION.CONNECTION_NAME'
)
```

**CRITICAL**: The first argument is a **TUPLE** with parentheses containing:
1. A STRING literal describing the scoring criteria
2. A column reference to the text being scored

### Examples

**Review helpfulness scoring:**
```sql
SELECT
    review_id,
    review_text,
    star_rating,
    AI.SCORE(
        ('Rate the helpfulness of this review based on detail level and examples. Review: ', review_text),
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS helpfulness_score
FROM `project.reviews.product_reviews`
ORDER BY helpfulness_score DESC
LIMIT 10;
```

**Movie review rating (from official docs):**
```sql
SELECT
    AI.SCORE((
        'On a scale from 1 to 10, rate how much the reviewer liked the movie. Review: ',
        review),
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS ai_rating,
    reviewer_rating AS human_rating,
    review
FROM `bigquery-public-data.imdb.reviews`
WHERE title = 'The English Patient'
ORDER BY ai_rating DESC
LIMIT 10;
```

**Negativity scoring:**
```sql
SELECT
    review,
    AI.SCORE(
        ('Rate negativity from 1-10: ', review),
        connection_id => 'us.my_ai_connection'  -- Replace with your connection
    ) AS negativity_score
FROM product_reviews
ORDER BY negativity_score DESC
LIMIT 5;
```

**Relevance scoring:**
```sql
SELECT
    document_id,
    title,
    AI.SCORE(
        ('How relevant is this document to machine learning and AI topics? Document: ', content),
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
            ('Rate review quality based on detail and helpfulness. Review: ', review_text),
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
6. **Escape Single Quotes**: When using string literals with apostrophes, escape them by doubling:
   - WRONG: `'The surgeon who 'sees' inside patients'`
   - CORRECT: `'The surgeon who ''sees'' inside patients'`

## Troubleshooting

**Error: "connection not found"**
- Ensure you've created the connection: `CREATE CLOUD RESOURCE CONNECTION`
- Use the correct format: `LOCATION.CONNECTION_NAME` (e.g., `us.my_ai_connection`)

**Error: "permission denied"**
- Grant the connection's service account access to Vertex AI API
