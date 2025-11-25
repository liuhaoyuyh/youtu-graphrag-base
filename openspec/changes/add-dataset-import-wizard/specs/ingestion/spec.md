## ADDED Requirements
### Requirement: Guided Dataset Creation Wizard
A wizard MUST collect dataset metadata, schema choice, and document sources through discrete steps before a construction job starts.

#### Scenario: Minimal dataset onboarding
- **GIVEN** a user opens the Dataset Import wizard
- **WHEN** they supply dataset name, owner, and pick an existing schema or upload a new JSON schema
- **THEN** the system MUST store the draft with a unique ID and present the document upload step.

#### Scenario: Resume draft
- **GIVEN** a draft dataset exists with incomplete uploads
- **WHEN** the user reopens the wizard
- **THEN** the wizard MUST preload previously entered metadata and indicate which steps remain.

### Requirement: Schema & Document Validation Service
The backend MUST validate schema structure and uploaded documents before enabling graph construction.

#### Scenario: Invalid schema is rejected
- **GIVEN** a schema JSON missing required `Nodes` or `Relations`
- **WHEN** validation runs
- **THEN** the API MUST respond with a structured error explaining the missing fields and prevent job submission.

#### Scenario: Document ingestion succeeds
- **GIVEN** documents that meet supported formats and size limits
- **WHEN** the user completes uploads and triggers validation
- **THEN** the service MUST confirm readiness, emit a progress update over WebSocket, and enable the "Construct Graph" action.
