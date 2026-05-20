-- Gold: token-optimized chunks ready for embedding / vector upsert
CREATE SCHEMA IF NOT EXISTS ${catalog}.gold;

CREATE TABLE IF NOT EXISTS ${catalog}.gold.rag_chunks (
  chunk_id STRING NOT NULL,
  article_id STRING NOT NULL,
  feed_id STRING NOT NULL,
  region STRING NOT NULL,
  language STRING NOT NULL,
  chunk_index INT NOT NULL,
  chunk_text STRING NOT NULL,
  token_count INT NOT NULL,
  published_at TIMESTAMP,
  source_url STRING,
  metadata MAP<STRING, STRING>,
  created_at TIMESTAMP NOT NULL,
  ingestion_date DATE NOT NULL,
  embedded_at TIMESTAMP COMMENT 'Set after Pinecone upsert job'
)
USING DELTA
PARTITIONED BY (ingestion_date, region)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.autoOptimize.optimizeWrite' = 'true'
);
