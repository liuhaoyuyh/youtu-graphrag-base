# Project Context

## Purpose
Youtu-GraphRAG delivers a vertically unified agentic GraphRAG platform that constructs schema-guided knowledge trees, performs graph-level retrieval, and exposes both CLI and web workflows for multi-hop, knowledge-intensive reasoning. The goal is to minimize manual schema tweaking while keeping token costs low and answer quality high.

## Tech Stack
- Python 3.10 runtime with FastAPI + Uvicorn for the backend server and websocket streaming.
- Frontend is a static HTML/CSS/vanilla JS SPA that relies on Axios for HTTP calls and ECharts for graph/trace visualization.
- Graph/ML layer: PyTorch 2.6, Transformers 4.51, Sentence-Transformers, FAISS, NetworkX, spaCy, and custom multi-agent pipelines under `models/`.
- Document ingestion relies on magic-pdf, doclayout-yolo, Tika, python-docx, pypdf, etc. Configuration uses YAML/JSON via `config/`.

## Project Conventions

### Code Style
- Follow PEP 8 with explicit type hints and pydantic `BaseModel`s for request/response contracts.
- Use `utils.logger.logger` for structured logging instead of print statements.
- Keep modules small and purpose-driven (e.g., `models.constructor`, `models.retriever`, `utils.document_parser`).

### Architecture Patterns
- Backend exposes thin REST/WebSocket endpoints that orchestrate deeper pipelines: ingestion → schema selection/expansion → multi-level knowledge tree construction → agentic retrieval/reasoning.
- Config management is centralized via `config/ConfigManager`; runtime overrides flow through CLI flags or `--override` JSON.
- Long-running tasks (graph construction, retrieval) stream progress via the `ConnectionManager` to the frontend for responsive UX.
- Frontend keeps data/state in-memory and renders via cards/tabs; no heavyweight framework is introduced to ease customization.

### Testing Strategy
- No formal automated test suite yet; validation happens by running `python main.py --datasets demo` or launching `start.sh` and driving flows through the UI.
- Focus for new work: add unit tests for pure functions (parsers, config utilities) and scenario tests for constructor/retriever modules when feasible.
- Specs/proposals must pass `openspec validate <change-id> --strict` before implementation is considered stable.

### Git Workflow
- Work happens on feature branches named `feature/<short-description>` (mirrors README guidance).
- Open PRs against `main` once the branch contains a coherent feature; keep commits descriptive (`Add graph streaming progress`, etc.).
- Large features should first land as OpenSpec proposals; implementations only start after proposal approval.

## Domain Context
- Core use cases: enterprise knowledge bases, research corpora, encyclopedic datasets that demand schema-aware, multi-hop reasoning.
- Knowledge is organized as a four-level tree (attributes → relations → keywords → communities) enabling both top-down summaries and bottom-up retrieval.
- Agentic retrieval decomposes questions with schema awareness, iteratively refines sub-queries (IRCoT), and fuses retrieved triples/chunks for responses.

## Important Constraints
- Requires OpenAI-compatible LLM endpoints (e.g., DeepSeek, Azure OpenAI) configured via `.env`; missing credentials block most functionality.
- Graph construction is resource-intensive; GPU acceleration and high-memory machines are strongly recommended for large corpora.
- Keep schema files in `schemas/` synchronized with datasets; mismatches lead to construction failures.
- Sensitive/private datasets must not leave the environment—avoid logging raw documents or secrets.

## External Dependencies
- Managed LLM APIs (OpenAI, Azure OpenAI, DeepSeek) for reasoning, decomposition, and summarization.
- Hugging Face-hosted embedding models via `sentence-transformers`, FAISS for vector search, and spaCy for NLP preprocessing.
- Apache Tika/doclayout-yolo/magic-pdf for parsing varied document formats; Axios/ECharts CDN bundles for the browser UI.
