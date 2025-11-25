# Change Proposal: add-dataset-import-wizard

## Why
- Uploading documents currently assumes users already know how to format JSON payloads and where to place schema files, which causes ingestion failures and support load.
- Enterprise users have compliance requirements (naming, ownership, retention) that should be captured up front before any graph construction job runs.
- A guided experience lowers the barrier for domain experts who are not comfortable with CLI overrides but still need to extend the platform.

## What Changes
- Introduce a multi-step "Dataset Import" wizard in the frontend with tabs for metadata, schema selection/creation, document upload, and validation summary.
- Add backend endpoints/services to persist draft datasets, validate schema JSON (structure + alignment with supported node/relation types), and queue construction jobs only after validation passes.
- Extend config/schema management so each dataset stores provenance (owner, description, source type) and optional processing hints (chunk size, language, privacy tags).
- Emit structured progress + error feedback over the existing WebSocket channel so the wizard can surface actionable guidance without page reloads.

## Impact
- Touches frontend `index.html` (new wizard UI) plus supporting JS; backend `backend.py`, `utils/document_parser.py`, and schema/config helpers gain validation routes.
- Requires new tests for schema validation helpers and positive/negative ingestion flows once the testing harness exists.
- No breaking changes to existing demo flow—the wizard is additive and can fall back to the current upload tab if disabled via config.
