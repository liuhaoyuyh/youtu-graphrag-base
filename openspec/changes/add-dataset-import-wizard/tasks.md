## 1. Implementation
- [ ] 1.1 Add dataset metadata + schema validation helpers (service layer, reusable exceptions, unit tests once harness exists).
- [ ] 1.2 Extend `backend.py` with endpoints for creating drafts, uploading files, running validation, and kicking off construction jobs when ready.
- [ ] 1.3 Update the WebSocket manager to broadcast wizard-specific progress + error payloads.
- [ ] 1.4 Build the multi-step wizard UI inside `frontend/index.html`, including form state, client-side validations, and Axios calls to the new endpoints.
- [ ] 1.5 Persist wizard selections/config overrides so `main.py` and CLI workflows can reuse the stored dataset profile.

## 2. Documentation & QA
- [ ] 2.1 Update README / FULLGUIDE with wizard usage instructions and screenshots once UI is stable.
- [ ] 2.2 Add regression tests or scripted runs covering successful import and invalid schema/doc flows.
- [ ] 2.3 Run `openspec validate add-dataset-import-wizard --strict` before requesting review.
