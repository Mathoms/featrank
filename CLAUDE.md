# CLAUDE.md — featrank

## Project Overview

Open-source CLI tool + REST API that ingests feature requests from any source, semantically deduplicates them, and ranks clusters by business value.

**Core pipeline:**
```
Raw requests (CSV/JSON/text)
  → Clean + normalize
  → Embed (sentence-transformers)
  → Cluster (HDBSCAN)
  → Extract structure (LLM)
  → Score (frequency + MRR + GitHub + roadmap)
  → Ranked priority report
```

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-mpnet-base-v2) |
| Clustering | `hdbscan` |
| LLM extraction | `groq` (llama-3.3-70b) or `ollama` (local) |
| RAG | `faiss-cpu` + `sentence-transformers` |
| CLI | `typer` |
| API | `fastapi` + `uvicorn` |
| Data validation | `pydantic v2` |
| Config | `python-dotenv` + `pydantic-settings` |
| Testing | `pytest` + `httpx` |
| Packaging | `pyproject.toml` (hatchling) |

## Claude Code Instructions

1. **Start with Day 1 tasks only.** Don't skip ahead.
2. **Run tests after every module.** `pytest tests/` must pass before moving on.
3. **Use type hints everywhere.** Pydantic models for all data flowing between modules.
4. **Cache embeddings.** Never recompute embeddings for the same input twice.
5. **Graceful degradation.** If LLM fails → fall back to TF-IDF labels. If GitHub fails → score = 0. Never crash the pipeline.
6. **Environment check on startup.** CLI and API both check required env vars on boot and give clear error messages.
7. **Log everything.** Use `loguru` — every pipeline stage logs input count, output count, timing.
8. **No hardcoded values.** All thresholds, weights, model names → `config.py` → `.env`.

## Day-by-Day Build Plan

### Day 1 — Data + Core Pipeline ✅
- [x] Package scaffold (`pyproject.toml`)
- [x] `scripts/generate_fixtures.py` — 500 synthetic requests
- [x] `featrank/pipeline/cleaner.py`
- [x] `featrank/pipeline/embedder.py`
- [x] `notebooks/01_eda.ipynb`

### Day 2 — Clustering + Labeling ✅
- [x] `featrank/pipeline/clusterer.py`
- [x] `featrank/pipeline/labeler.py`
- [x] `notebooks/02_embedding_experiments.ipynb`
- [x] `notebooks/03_clustering_tuning.ipynb`

### Day 3 — Scoring Engine ✅
- [x] `featrank/scoring/frequency.py`
- [x] `featrank/scoring/segment_value.py`
- [x] `featrank/scoring/github_signal.py`
- [x] `featrank/scoring/roadmap_fit.py`
- [x] `featrank/scoring/ranker.py`
- [x] `notebooks/04_scoring_validation.ipynb`

### Day 4 — LLM Layer + CLI ✅
- [x] `featrank/llm/extractor.py`
- [x] `featrank/llm/narrator.py`
- [x] `featrank/cli.py`

### Day 5 — FastAPI ✅
- [x] `featrank/api/main.py` + all routes
- [x] `featrank/api/schemas.py`
- [x] `docker-compose.yml`

### Day 6 — Polish + Benchmark
- [ ] `scripts/benchmark.py` — full benchmark run ✅ (written, needs run)
- [ ] Full test suite passing (`pytest tests/`)
- [ ] Docker build passing

### Day 7 — Ship
- [ ] Publish to PyPI: `hatch build && hatch publish`
- [ ] HuggingFace model card
- [ ] GitHub repo polished
- [ ] arXiv abstract drafted
