# Live News-to-Insight Engine

Production-oriented multi-agent, multilingual RAG platform:

| Layer | Technology |
|-------|------------|
| Orchestration | Apache Airflow (`dags/news_ingestion_dag.py`) |
| Lakehouse | Databricks Delta medallion (Bronze → Silver → Gold) |
| Embeddings | Hugging Face `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector DB | Pinecone (per-region namespaces) |
| Agents | Router → Trend Analyzer \| Procurement Planner |
| Observability | Langfuse traces on ingestion, embed, retrieval, LLM |

## Repository layout

```
news-insight-engine/
├── dags/                 # Airflow ingestion + Databricks trigger
├── databricks/
│   ├── schemas/          # Bronze / Silver / Gold DDL
│   └── src/              # PySpark medallion jobs
├── vector/               # HF embed + Pinecone upsert job
├── agents/               # Router + specialist RAG agents
├── observability/        # Langfuse helpers
├── config/               # Settings + feed registry
└── run_insight.py        # CLI for agent queries
```

## Quick start

### Run locally (no cloud keys)

```bash
cd news-insight-engine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_ingest_local.py
python scripts/run_offline_demo.py "procurement supply chain risks"
```

### 1. Install dependencies

```bash
cd news-insight-engine
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # fill secrets
```

### 2. Databricks — create schemas

Run SQL in `databricks/schemas/` against your catalog (replace `${catalog}`).

### 3. Airflow — configure

- Copy `dags/` into your Astro/Airflow `dags` folder (or set `DAGS_FOLDER`).
- Set Airflow Variables:
  - `news_insight_medallion_job_id` — Databricks job ID from bundle deploy
  - `news_insight_bronze_staging_uri` — local or cloud staging path for JSONL
- Connection `databricks_default` with host, token, job id.

### 4. Databricks bundle

```bash
cd databricks
databricks bundle validate --target dev
databricks bundle deploy --target dev
```

Set `cluster_id` in `databricks.yml` or target variables.

### 5. Pinecone

Index is auto-created on first embed job (cosine, dim **384**). Namespaces map to feed `region` (e.g. `global`, `emea`).

### 6. Run agents

```bash
python run_insight.py "What supply chain risks are emerging in EMEA electronics?"
python run_insight.py "Summarize bullish trends in global energy markets" --region global
```

Traces appear in Langfuse under spans: `airflow-fetch-feeds`, `hf-embed-batch`, `pinecone-query`, `router-handle`, `agent-llm-generation`.

## Pipeline flow

1. **Airflow** (`@hourly`): fetch RSS → JSONL staging → `DatabricksRunNowOperator` medallion job.
2. **Bronze**: append raw articles to `catalog.bronze.raw_news`.
3. **Silver**: clean, hash-dedupe, language tag → `catalog.silver.cleaned_news`.
4. **Gold**: word-window chunks → `catalog.gold.rag_chunks`.
5. **Vector job**: embed batches → Pinecone upsert → mark `embedded_at` on Gold.
6. **Agents**: router classifies intent → specialist RAG + OpenAI JSON response.

## Environment variables

See [`.env.example`](.env.example). Required for full stack:

- `DATABRICKS_*` / Airflow connection
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `OPENAI_API_KEY`

## Notes

- Router and specialists request **JSON** from the LLM (`response_format=json_object`).
- Embedding model dimension must match Pinecone index (**384** for MiniLM-L12-v2).
- For production, replace append-only Delta writes with `MERGE` in `databricks/src/common.py` and run vector upsert on a dedicated GPU cluster if batch size is large.
