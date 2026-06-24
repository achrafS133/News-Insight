-- Bronze: raw ingested JSON from Airflow / feed connectors
CREATE SCHEMA IF NOT EXISTS ${catalog}.bronze;

CREATE TABLE IF NOT EXISTS ${catalog}.bronze.raw_news (
  article_id STRING NOT NULL COMMENT 'Stable hash id per feed entry',
  feed_id STRING NOT NULL,
  region STRING NOT NULL,
  source_type STRING NOT NULL,
  source_url STRING,
  title STRING,
  summary STRING,
  body STRING,
  published_at TIMESTAMP,
  language_hint STRING,
  raw_payload STRING COMMENT 'Original entry JSON',
  ingested_at TIMESTAMP NOT NULL,
  ingestion_date DATE NOT NULL
)
USING DELTA
PARTITIONED BY (ingestion_date, region)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true'
);
