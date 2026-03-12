"""Webhook server and handlers for the climbing-holds dataset indexer."""

from __future__ import annotations

import os
from pathlib import PurePosixPath

from huggingface_hub import HfApi, WebhooksServer, WebhookPayload

import gradio as gr

import config
import global_index
import hf_repo
import holds


demo = gr.Blocks()
app = WebhooksServer(ui=demo, webhook_secret=os.environ.get("WEBHOOK_SECRET"))


@app.add_webhook("/index")
async def trigger_indexation(payload: WebhookPayload) -> dict:
    if payload.event.action == "ping":
        config.logger.info("Keep-alive ping received. Space is awake!")
        return {"status": "success", "message": "PONG - Space is awake"}

    # Only react to repo content updates on the target dataset
    if payload.event.action != "update":
        return {"status": "ignored", "reason": "Not an update event"}

    if not getattr(payload.event, "scope", "").startswith("repo.content"):
        return {"status": "ignored", "reason": "Not a repo content update"}

    if getattr(payload.repo, "type", None) != "dataset":
        return {"status": "ignored", "reason": "Not a dataset repo"}

    repo_id = config.DEFAULT_REPO_ID
    if repo_id != config.DEFAULT_REPO_ID:
        return {"status": "ignored", "reason": f"Repo '{repo_id}' is not the target dataset"}

    token = os.environ.get("HF_TOKEN")
    if not token:
        config.logger.error("HF_TOKEN missing in Space secrets.")
        return {"status": "error", "reason": "Missing HF_TOKEN"}

    api = HfApi(token=token)
    revision = os.environ.get("HF_REVISION")

    try:
        current_index = hf_repo.load_global_index(repo_id, token, revision)
        if not isinstance(current_index, dict):
            raise RuntimeError("global_index must be a top-level JSON object.")

        allowed_references = global_index.ensure_allowed_references(current_index)
        metadata_paths, files_by_directory = hf_repo.list_dataset_files(api, repo_id, token, revision)
        config.logger.info("Found %d metadata files in dataset '%s'.", len(metadata_paths), repo_id)

        initial_attention = global_index.prepare_initial_attention(current_index)
        holds_list, needs_attention, metadata_updates = holds.rebuild_holds(
            repo_id=repo_id,
            token=token,
            revision=revision,
            metadata_paths=metadata_paths,
            files_by_directory=files_by_directory,
            allowed_references=allowed_references,
            initial_attention=initial_attention,
        )

        updated_index = global_index.update_global_index(current_index, holds_list, needs_attention)
        train_jsonl_payload = holds.build_train_jsonl(holds_list)

        # Create missing per-hold votes.json files (empty list)
        new_votes_files: dict[str, list] = {}
        for metadata_path in metadata_paths:
            hold_dir = str(PurePosixPath(metadata_path).parent)
            votes_path = f"{hold_dir}/{config.VOTES_FILENAME}"
            if votes_path not in files_by_directory.get(hold_dir, []):
                new_votes_files[votes_path] = []

        if not global_index.has_meaningful_changes(current_index, updated_index) and not metadata_updates and not new_votes_files:
            config.logger.info("No changes detected, commit skipped to preserve history.")
            return {
                "status": "success",
                "message": "No changes",
                "holds": updated_index["stats"]["total_holds"],
                "to_identify": updated_index["stats"]["to_identify"],
            }

        hf_repo.commit_dataset_updates(
            api,
            repo_id=repo_id,
            token=token,
            global_index_payload=updated_index,
            train_jsonl_payload=train_jsonl_payload,
            metadata_updates=metadata_updates,
            new_votes_files=new_votes_files if new_votes_files else None,
        )
        config.logger.info(
            "Global index updated successfully: %d holds, %d attention entries.",
            updated_index["stats"]["total_holds"],
            updated_index["stats"]["to_identify"],
        )
        return {
            "status": "success",
            "holds": updated_index["stats"]["total_holds"],
            "to_identify": updated_index["stats"]["to_identify"],
        }

    except Exception as exc:
        config.logger.exception("Global index update failed: %s", exc)
        return {"status": "error", "reason": str(exc)}
