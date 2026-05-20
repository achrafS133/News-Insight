-- Silver: cleaned, deduplicated articles with language tags
CREATE SCHEMA IF NOT EXISTS ${catalog}.silver;

CREATE TABLE IF NOT EXISTS ${catalog}.silver.cleaned_news (
  article_id STRING NOT NULL,
  feed_id STRING NOT NULL,
  region STRING NOT NULL,
  title STRING NOT NULL,
  content STRING NOT NULL,
  content_hash STRING NOT NULL COMMENT 'SHA-256 of normalized content for dedupe',
  language STRING NOT NULL,
  published_at TIMESTAMP,
  source_url STRING,
  metadata MAP<STRING, STRING>,
  processed_at TIMESTAMP NOT NULL,
  ingestion_date DATE NOT NULL
)
USING DELTA
PARTITIONED BY (ingestion_date, language)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true'
);
