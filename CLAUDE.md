# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A FastAPI application deployed as a Hugging Face Space (Gradio SDK). It acts as an indexer and API server for a climbing holds dataset hosted on Hugging Face Hub. It handles:
- Webhook-triggered re-indexation when the HF dataset changes
- Vote processing for community-driven hold identification
- Anonymous hold contributions

## Running Locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Required environment variables (see `.gitignore` — use a `.env` file):
- `HF_TOKEN` — write access to the main dataset
- `WEBHOOK_SECRET` — validates incoming HF webhooks
- `HF_REVISION` — branch to read/write (defaults to `staging`)
- `HF_ANONYMOUS_TOKEN` — write access to the anonymous contributions repo
- `HF_ANONYMOUS_REPO_ID` — anonymous contributions repo ID
- `CORS_ALLOWED_ORIGINS` — comma-separated additional CORS origins

## Manual Re-indexation

```bash
# Dry run (default) — shows what would change without committing
python reindex.py

# Commit the reindex to the dataset
python reindex.py --commit

# Target a specific repo or revision
python reindex.py --repo-id owner/dataset --revision main --commit
```

## Architecture

### Data Flow: Webhook-triggered Re-indexation
1. HF dataset webhook fires → `webhooks.py` verifies secret and receives payload
2. `global_index.py` loads the current `global_index.json` from the HF dataset
3. `hf_repo.py` lists all `metadata.json` files in the dataset
4. `holds.py` normalizes/validates each hold's metadata, infers mesh presence, tracks attention items
5. A new `train.jsonl` is built from all holds
6. `hf_repo.py` commits if content hash changed (idempotent via SHA-256)

### Data Flow: Vote Processing
1. POST `/webhooks/vote` → `votes.py` validates payload (rating 1–5, ISO8601 timestamp)
2. Voter fingerprint computed from HF username or hashed IP
3. Votes stored in per-hold `votes.json`; duplicate votes rejected
4. Dominant manufacturer/model inferred from vote aggregate
5. Hold `metadata.json` updated if dominant values changed

### Data Flow: Anonymous Contributions
1. POST multipart form → `contributions.py` sanitizes filenames and deduplicates
2. Files committed to anonymous HF repo under `pending/<uuid>/`

### Key Files
- `app.py` — FastAPI app, mounts Gradio UI, wires endpoints
- `config.py` — all constants: repo IDs, file path names, mesh extensions, managed keys
- `global_index.py` — index bootstrap, reference validation (manufacturers, hold types, status), attention tracking
- `hf_repo.py` — all Hugging Face Hub I/O (load/save JSON, list files, commit)
- `holds.py` — metadata normalization, reference validation with fuzzy-match suggestions, `train.jsonl` builder
- `votes.py` — vote validation, fingerprinting, aggregation
- `webhooks.py` — webhook route handlers, CORS setup
- `contributions.py` — anonymous submission handler
- `reindex.py` — standalone CLI for manual full reindex

### Dataset Structure on HF Hub
- `global_index.json` — top-level index with allowed references and attention sets
- `<hold_id>/metadata.json` — per-hold metadata
- `<hold_id>/votes.json` — per-hold community votes
- `train.jsonl` — flat JSONL export of all normalized holds

## No Tests

There are no automated tests in this project.

## Deployment

This runs as a Hugging Face Space (see the YAML frontmatter in `README.md`). The `app_file` is `app.py`. Pushing to the Space repo triggers a redeploy.
