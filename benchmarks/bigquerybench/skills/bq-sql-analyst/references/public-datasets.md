# BigQuery Public Datasets Reference

Commonly used datasets in the `bigquery-public-data` project.

## usa_names

Baby name data from US Social Security Administration.

| Table | Description |
|-------|-------------|
| `usa_1910_current` | Names, gender, state, year, count from 1910 to present |

Key columns: `name` (STRING), `gender` (STRING), `state` (STRING), `year` (INT64), `number` (INT64)

## samples

Sample datasets provided by BigQuery.

| Table | Description |
|-------|-------------|
| `shakespeare` | Every word in every Shakespeare work with word count |

Key columns: `word` (STRING), `word_count` (INT64), `corpus` (STRING), `corpus_date` (INT64)

## austin_bikeshare

Austin B-cycle bikeshare trip data.

| Table | Description |
|-------|-------------|
| `bikeshare_trips` | Individual trip records with start/end stations and times |
| `bikeshare_stations` | Station locations and metadata |

Key columns (trips): `trip_id`, `start_station_name`, `end_station_name`, `duration_minutes`, `start_time`
