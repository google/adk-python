# BigQuery AI.Classify

`AI.CLASSIFY` categorizes unstructured data into a predefined set of labels.

## Syntax Reference

```sql
AI.CLASSIFY(
  [ input => ] 'INPUT',
  [ categories => ] 'CATEGORIES'
  [, connection_id => 'CONNECTION_ID' ]
  [, endpoint => 'ENDPOINT' ]
  [, output_mode => 'OUTPUT_MODE' ]
)
```

### Input Arguments

| Argument            | Requirement  | Type          | Description           |
| :------------------ | :----------- | :------------ | :-------------------- |
| **`input`**         | **Required** | String        | The text content to   |
:                     :              :               : classify.             :
| **`categories`**    | **Required** | Array<String> | A list of target      |
:                     :              :               : categories/labels.    :
:                     :              :               : Can be                :
:                     :              :               : `ARRAY<STRING>` or    :
:                     :              :               : `ARRAY<STRUCT<STRING, :
:                     :              :               : STRING>>` (label,     :
:                     :              :               : description).         :
| **`connection_id`** | Optional     | String        | The connection ID to  |
:                     :              :               : use for the LLM.      :
| **`endpoint`**      | Optional     | String        | The model name, e.g., |
:                     :              :               : `'gemini-2.5-flash'`. :
| **`output_mode`**   | Optional     | String        | `'single'` (default)  |
:                     :              :               : or `'multi'`.         :
:                     :              :               : Determines the output :
:                     :              :               : type.                 :

### Output Schema

The output type depends on the `output_mode` argument:

| Output Mode      | output_mode Value | Type            | Description         |
| :--------------- | :---------------- | :-------------- | :------------------ |
| **Single Label** | `NULL` (Default)  | `STRING`        | The single category |
:                  :                   :                 : that best fits the  :
:                  :                   :                 : input.              :
| **Single Label   | `'single'`        | `ARRAY<STRING>` | An array containing |
: (Explicit)**     :                   :                 : exactly one         :
:                  :                   :                 : category string.    :
| **Multi Label**  | `'multi'`         | `ARRAY<STRING>` | An array containing |
:                  :                   :                 : zero or more        :
:                  :                   :                 : matching            :
:                  :                   :                 : categories.         :

## Examples

### Classify text into categories

```sql
SELECT
  content,
  AI.CLASSIFY(
    content,
    categories => ['Spam', 'Not Spam', 'Urgent'],
    connection_id => 'my-project.us.my-connection'
  ) as classification
FROM `dataset.emails`;
```

### Classify text into multiple topics

```
SELECT
  title,
  body,
  AI.CLASSIFY(
    body,
    categories => ['tech', 'sport', 'business', 'politics', 'entertainment', 'other'],
    output_mode => 'multi') AS categories
FROM
  `bigquery-public-data.bbc_news.fulltext`
LIMIT 100;
```

### Classify reviews by sentiment

SELECT AI.CLASSIFY( ('Classify the review by sentiment: ', review), categories
=> [('green', 'The review is positive.'), ('yellow', 'The review is neutral.'),
('red', 'The review is negative.')]) AS ai_review_rating, reviewer_rating AS
human_provided_rating, review, FROM `bigquery-public-data.imdb.reviews` WHERE
title = 'The English Patient'
