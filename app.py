#!/usr/bin/env python3
"""Rebuild and upload the Hugging Face dataset global index via Webhook.

Deploy this file as app.py in a Hugging Face Space (Gradio SDK).
Configure the HF webhook to point to: https://YOUR-SPACE.hf.space/webhooks/indexation
Set Space secrets: HF_TOKEN (write access to the dataset), WEBHOOK_SECRET (same as in webhook settings).
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, WebhooksServer, WebhookPayload

DEFAULT_REPO_ID = "setrsoft/climbing-holds"
GLOBAL_INDEX_PATH = "meta/global_index.json"
TRAIN_JSONL_PATH = "train.jsonl"
METADATA_FILENAME = "metadata.json"
MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl"}
MANAGED_ATTENTION_KEYS = {
    "invalid_hold_type_reference",
    "invalid_manufacturer_reference",
    "invalid_metadata",
    "invalid_status_reference",
    "missing_mesh",
    "unknown_hold_type",
    "unknown_manufacturer",
    "unknown_model",
}

logger = logging.getLogger("update_global_index")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


# =============================================================================
# Business logic (from scripts/update_global_index.py)
# =============================================================================


def bootstrap_global_index(repo_id: str) -> dict[str, Any]:
    logger.warning(
        "Remote '%s' is empty. Bootstrapping a new global index with empty allowed references.",
        GLOBAL_INDEX_PATH,
    )
    return {
        "project": repo_id.split("/")[-1],
        "allowed_references": {
            "manufacturers": [],
            "hold_types": [],
            "status": ["to_render", "to_clean", "to_identify"],
        },
        "stats": {"total_holds": 0, "to_identify": 0},
        "needs_attention": {},
    }


def load_json_file(
    repo_id: str,
    path_in_repo: str,
    token: str,
    revision: str | None,
    *,
    allow_empty: bool = False,
) -> Any:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path_in_repo,
            token=token,
            revision=revision,
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to download '{path_in_repo}' from '{repo_id}': {exc}") from exc

    try:
        with open(local_path, "r", encoding="utf-8") as handle:
            raw_content = handle.read()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"'{path_in_repo}' is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Unable to read '{path_in_repo}': {exc}") from exc

    if not raw_content.strip():
        if allow_empty:
            return bootstrap_global_index(repo_id)
        raise RuntimeError(f"'{path_in_repo}' is empty and cannot be parsed as JSON.")

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"'{path_in_repo}' is not valid JSON: {exc}") from exc


def list_dataset_files(
    api: HfApi,
    repo_id: str,
    token: str,
    revision: str | None,
) -> tuple[list[str], dict[str, list[str]]]:
    try:
        repo_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            revision=revision,
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to list files for '{repo_id}': {exc}") from exc

    files_by_directory: dict[str, list[str]] = defaultdict(list)
    metadata_paths: list[str] = []

    for repo_file in repo_files:
        repo_path = PurePosixPath(repo_file)
        directory = str(repo_path.parent)
        files_by_directory[directory].append(repo_file)
        if (
            len(repo_path.parts) == 2
            and repo_path.name == METADATA_FILENAME
            and repo_path.parent.name != "meta"
        ):
            metadata_paths.append(repo_file)

    metadata_paths.sort()
    return metadata_paths, files_by_directory


def normalize_reference_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized or None
    return str(value).strip().lower() or None


def canonical_hold_id(metadata: dict[str, Any], metadata_path: str) -> str:
    folder_name = PurePosixPath(metadata_path).parent.name
    raw_hold_id = metadata.get("hold_id")

    if raw_hold_id is None:
        return folder_name

    hold_id = str(raw_hold_id).strip()
    if not hold_id:
        return folder_name

    if hold_id != folder_name:
        logger.warning(
            "Hold ID mismatch for '%s': metadata has '%s', using folder name '%s'.",
            metadata_path,
            hold_id,
            folder_name,
        )
    return folder_name


def ensure_allowed_references(global_index: dict[str, Any]) -> dict[str, set[str]]:
    allowed_references = global_index.get("allowed_references")
    if not isinstance(allowed_references, dict):
        raise RuntimeError("global_index.json is missing the 'allowed_references' section.")

    manufacturers = allowed_references.setdefault("manufacturers", [])
    hold_types = allowed_references.setdefault("hold_types", [])
    statuses = allowed_references.setdefault("status", [])

    if not isinstance(manufacturers, list) or not isinstance(hold_types, list):
        raise RuntimeError(
            "global_index.json must define list values for 'allowed_references.manufacturers' "
            "and 'allowed_references.hold_types'."
        )
    if not isinstance(statuses, list):
        raise RuntimeError(
            "global_index.json must define a list value for 'allowed_references.status'."
        )

    return {
        "manufacturers": {value for value in map(normalize_reference_value, manufacturers) if value},
        "hold_types": {value for value in map(normalize_reference_value, hold_types) if value},
        "status": {value for value in map(normalize_reference_value, statuses) if value},
    }


def infer_mesh_presence(hold_directory: str, files_by_directory: dict[str, list[str]]) -> bool:
    for repo_file in files_by_directory.get(hold_directory, []):
        file_name = PurePosixPath(repo_file).name
        if file_name == METADATA_FILENAME:
            continue
        if PurePosixPath(repo_file).suffix.lower() in MESH_EXTENSIONS:
            return True
    return False


def warn_about_reference(
    *,
    hold_id: str,
    field_name: str,
    value: Any,
    allowed_values: set[str],
    attention_bucket: set[str],
) -> None:
    normalized_value = normalize_reference_value(value)
    attention_bucket.add(hold_id)

    if normalized_value is None:
        logger.warning("Hold '%s' has a null or empty '%s'.", hold_id, field_name)
        return

    if normalized_value == "unknown":
        logger.warning("Hold '%s' still has an unknown '%s'.", hold_id, field_name)
        return

    suggestion = difflib.get_close_matches(
        normalized_value,
        sorted(value for value in allowed_values if value != "unknown"),
        n=1,
        cutoff=0.8,
    )
    if suggestion:
        logger.warning(
            "Hold '%s' has an invalid '%s' value '%s'. Did you mean '%s'?",
            hold_id,
            field_name,
            value,
            suggestion[0],
        )
        return

    logger.warning(
        "Hold '%s' has an invalid '%s' value '%s' which is not in allowed_references.",
        hold_id,
        field_name,
        value,
    )


def validate_metadata(
    metadata: dict[str, Any],
    hold_id: str,
    allowed_references: dict[str, set[str]],
    needs_attention: dict[str, set[str]],
) -> None:
    manufacturer_value = metadata.get("manufacturer")
    manufacturer_ref = normalize_reference_value(manufacturer_value)
    if manufacturer_ref in {None, "unknown"}:
        warn_about_reference(
            hold_id=hold_id,
            field_name="manufacturer",
            value=manufacturer_value,
            allowed_values=allowed_references["manufacturers"],
            attention_bucket=needs_attention["unknown_manufacturer"],
        )
    elif manufacturer_ref not in allowed_references["manufacturers"]:
        warn_about_reference(
            hold_id=hold_id,
            field_name="manufacturer",
            value=manufacturer_value,
            allowed_values=allowed_references["manufacturers"],
            attention_bucket=needs_attention["invalid_manufacturer_reference"],
        )

    hold_type_value = metadata.get("type")
    hold_type_ref = normalize_reference_value(hold_type_value)
    if hold_type_ref in {None, "unknown"}:
        warn_about_reference(
            hold_id=hold_id,
            field_name="type",
            value=hold_type_value,
            allowed_values=allowed_references["hold_types"],
            attention_bucket=needs_attention["unknown_hold_type"],
        )
    elif hold_type_ref not in allowed_references["hold_types"]:
        warn_about_reference(
            hold_id=hold_id,
            field_name="type",
            value=hold_type_value,
            allowed_values=allowed_references["hold_types"],
            attention_bucket=needs_attention["invalid_hold_type_reference"],
        )

    model_value = metadata.get("model")
    model_ref = normalize_reference_value(model_value)
    if model_ref in {None, "unknown"}:
        needs_attention["unknown_model"].add(hold_id)
        logger.warning("Hold '%s' still has an unknown 'model'.", hold_id)

    status_value = metadata.get("status")
    status_ref = normalize_reference_value(status_value)
    if status_ref is not None and status_ref not in allowed_references["status"]:
        warn_about_reference(
            hold_id=hold_id,
            field_name="status",
            value=status_value,
            allowed_values=allowed_references["status"],
            attention_bucket=needs_attention["invalid_status_reference"],
        )


def canonical_metadata_defaults() -> list[tuple[str, Any]]:
    return [
        ("id", ""),
        ("hold_id", ""),
        ("created_at", ""),
        ("last_update", ""),
        ("timezone_offset", ""),
        ("type", ""),
        ("labels", []),
        ("color_of_scan", ""),
        ("available_colors", []),
        ("manufacturer", ""),
        ("model", ""),
        ("size", ""),
        ("note", ""),
        ("status", ""),
        ("text", ""),
    ]


def normalize_metadata(metadata: dict[str, Any], *, hold_id: str) -> tuple[dict[str, Any], bool]:
    defaults = canonical_metadata_defaults()
    normalized: dict[str, Any] = {}
    changed = False

    existing = dict(metadata)
    existing["hold_id"] = hold_id

    for key, default_value in defaults:
        if key in existing:
            normalized[key] = existing[key]
        else:
            if isinstance(default_value, list):
                normalized[key] = []
            else:
                normalized[key] = default_value
            changed = True

    for key, value in existing.items():
        if key not in normalized:
            normalized[key] = value
    return normalized, changed


def rebuild_holds(
    *,
    repo_id: str,
    token: str,
    revision: str | None,
    metadata_paths: list[str],
    files_by_directory: dict[str, list[str]],
    allowed_references: dict[str, set[str]],
    initial_attention: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[str, dict[str, Any]]]:
    needs_attention = {key: set(values) for key, values in initial_attention.items()}
    for managed_key in MANAGED_ATTENTION_KEYS:
        needs_attention[managed_key] = set()

    holds: list[dict[str, Any]] = []
    metadata_updates: dict[str, dict[str, Any]] = {}

    for metadata_path in metadata_paths:
        hold_directory = str(PurePosixPath(metadata_path).parent)
        hold_folder = PurePosixPath(metadata_path).parent.name

        try:
            metadata = load_json_file(repo_id, metadata_path, token, revision)
        except RuntimeError as exc:
            logger.warning("%s Skipping file.", exc)
            needs_attention["invalid_metadata"].add(hold_folder)
            continue

        if not isinstance(metadata, dict):
            logger.warning(
                "Metadata file '%s' does not contain a JSON object. Skipping file.",
                metadata_path,
            )
            needs_attention["invalid_metadata"].add(hold_folder)
            continue

        hold_id = canonical_hold_id(metadata, metadata_path)
        metadata_copy, did_change = normalize_metadata(metadata, hold_id=hold_id)
        if did_change:
            metadata_updates[metadata_path] = metadata_copy

        validate_metadata(metadata_copy, hold_id, allowed_references, needs_attention)

        if not infer_mesh_presence(hold_directory, files_by_directory):
            needs_attention["missing_mesh"].add(hold_id)
            logger.warning(
                "Hold '%s' does not have a mesh file next to '%s'.",
                hold_id,
                metadata_path,
            )

        holds.append(metadata_copy)

    holds.sort(key=lambda item: str(item.get("hold_id", "")))
    return holds, needs_attention, metadata_updates


def prepare_initial_attention(global_index: dict[str, Any]) -> dict[str, set[str]]:
    existing_attention = global_index.get("needs_attention", {})
    if not isinstance(existing_attention, dict):
        existing_attention = {}

    prepared: dict[str, set[str]] = {}
    for key, value in existing_attention.items():
        if isinstance(value, list):
            prepared[key] = {str(item) for item in value}
        else:
            prepared[key] = set()

    for key in MANAGED_ATTENTION_KEYS:
        prepared.setdefault(key, set())
    return prepared


def update_global_index(
    current_index: dict[str, Any],
    holds: list[dict[str, Any]],
    needs_attention: dict[str, set[str]],
) -> dict[str, Any]:
    next_index = dict(current_index)
    next_index["last_updated"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    stats = current_index.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    stats = dict(stats)
    stats["total_holds"] = len(holds)

    all_attention_ids: set[str] = set()
    serialized_attention: dict[str, list[str]] = {}
    for key in sorted(needs_attention):
        serialized_attention[key] = sorted(needs_attention[key])
        all_attention_ids.update(needs_attention[key])

    stats["to_identify"] = len(all_attention_ids)
    next_index["stats"] = stats
    next_index["needs_attention"] = serialized_attention
    return next_index


def build_comparison_payload(global_index: dict[str, Any]) -> dict[str, Any]:
    return {
        "stats": global_index.get("stats", {}),
        "needs_attention": global_index.get("needs_attention", {}),
    }


def compute_payload_hash(payload: dict[str, Any]) -> str:
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def has_meaningful_changes(current_index: dict[str, Any], updated_index: dict[str, Any]) -> bool:
    return compute_payload_hash(build_comparison_payload(current_index)) != compute_payload_hash(
        build_comparison_payload(updated_index)
    )


def build_train_jsonl(holds: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for hold in sorted(holds, key=lambda item: str(item.get("hold_id", ""))):
        lines.append(json.dumps(hold, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def commit_dataset_updates(
    api: HfApi,
    *,
    repo_id: str,
    token: str,
    global_index_payload: dict[str, Any],
    train_jsonl_payload: str,
    metadata_updates: dict[str, dict[str, Any]],
) -> None:
    operations: list[CommitOperationAdd] = []

    serialized_index = json.dumps(global_index_payload, indent=2, ensure_ascii=False) + "\n"
    operations.append(
        CommitOperationAdd(
            path_in_repo=GLOBAL_INDEX_PATH,
            path_or_fileobj=io.BytesIO(serialized_index.encode("utf-8")),
        )
    )

    operations.append(
        CommitOperationAdd(
            path_in_repo=TRAIN_JSONL_PATH,
            path_or_fileobj=io.BytesIO(train_jsonl_payload.encode("utf-8")),
        )
    )

    for metadata_path, normalized in sorted(metadata_updates.items()):
        serialized_metadata = json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"
        operations.append(
            CommitOperationAdd(
                path_in_repo=metadata_path,
                path_or_fileobj=io.BytesIO(serialized_metadata.encode("utf-8")),
            )
        )

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        operations=operations,
        commit_message=(
            "Update global index, train, and normalized metadata "
            f"({global_index_payload.get('stats', {}).get('total_holds', 0)} holds)"
        ),
    )


# =============================================================================
# Webhook server
# =============================================================================

app = WebhooksServer(webhook_secret=os.environ.get("WEBHOOK_SECRET"))

@app.add_webhook("/index")
async def trigger_indexation(payload: WebhookPayload) -> dict[str, Any]:

    if payload.event.action == "ping":
        logger.info("Keep-alive ping received. Space is awake!")
        return {"status": "success", "message": "PONG - Space is awake"}
    
    # Only react to repo content updates on the target dataset
    if payload.event.action != "update":
        return {"status": "ignored", "reason": "Not an update event"}

    if not getattr(payload.event, "scope", "").startswith("repo.content"):
        return {"status": "ignored", "reason": "Not a repo content update"}

    if getattr(payload.repo, "type", None) != "dataset":
        return {"status": "ignored", "reason": "Not a dataset repo"}

    repo_id = DEFAULT_REPO_ID
    if repo_id != DEFAULT_REPO_ID:
        return {"status": "ignored", "reason": f"Repo '{repo_id}' is not the target dataset"}

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN missing in Space secrets.")
        return {"status": "error", "reason": "Missing HF_TOKEN"}

    api = HfApi(token=token)
    revision = os.environ.get("HF_REVISION")

    try:
        current_index = load_json_file(
            repo_id,
            GLOBAL_INDEX_PATH,
            token,
            revision,
            allow_empty=True,
        )
        if not isinstance(current_index, dict):
            raise RuntimeError("global_index.json must contain a top-level JSON object.")

        allowed_references = ensure_allowed_references(current_index)
        metadata_paths, files_by_directory = list_dataset_files(api, repo_id, token, revision)
        logger.info("Found %d metadata files in dataset '%s'.", len(metadata_paths), repo_id)

        initial_attention = prepare_initial_attention(current_index)
        holds, needs_attention, metadata_updates = rebuild_holds(
            repo_id=repo_id,
            token=token,
            revision=revision,
            metadata_paths=metadata_paths,
            files_by_directory=files_by_directory,
            allowed_references=allowed_references,
            initial_attention=initial_attention,
        )

        updated_index = update_global_index(current_index, holds, needs_attention)
        train_jsonl_payload = build_train_jsonl(holds)

        if not has_meaningful_changes(current_index, updated_index) and not metadata_updates:
            logger.info("No changes detected, commit skipped to preserve history.")
            return {
                "status": "success",
                "message": "No changes",
                "holds": updated_index["stats"]["total_holds"],
                "to_identify": updated_index["stats"]["to_identify"],
            }

        commit_dataset_updates(
            api,
            repo_id=repo_id,
            token=token,
            global_index_payload=updated_index,
            train_jsonl_payload=train_jsonl_payload,
            metadata_updates=metadata_updates,
        )
        logger.info(
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
        logger.exception("Global index update failed: %s", exc)
        return {"status": "error", "reason": str(exc)}


if __name__ == "__main__":
    app.launch()
