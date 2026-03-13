"""Hugging Face API I/O: load/save JSON, list repo files, commit updates."""

from __future__ import annotations

import io
import json
import uuid
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any, BinaryIO

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

import config
import global_index


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
            return global_index.bootstrap_global_index(repo_id)
        raise RuntimeError(f"'{path_in_repo}' is empty and cannot be parsed as JSON.")

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"'{path_in_repo}' is not valid JSON: {exc}") from exc


def load_json_file_optional(
    repo_id: str,
    path_in_repo: str,
    token: str,
    revision: str | None,
    default: Any,
) -> Any:
    """Load JSON file from repo, or return default if file is missing (404) or empty."""
    try:
        return load_json_file(
            repo_id, path_in_repo, token, revision, allow_empty=False
        )
    except Exception as exc:
        err_msg = str(exc).lower()
        if "404" in str(exc) or "not found" in err_msg or "no file" in err_msg or "empty" in err_msg:
            return default
        raise


def load_global_index(
    repo_id: str,
    token: str,
    revision: str | None,
) -> dict[str, Any]:
    """Load global index from repo. Tries new path first, then legacy root path (one-time migration)."""
    last_error: Exception | None = None
    for path in (config.GLOBAL_INDEX_PATH, config.LEGACY_GLOBAL_INDEX_PATH):
        try:
            data = load_json_file(
                repo_id, path, token, revision, allow_empty=True
            )
            if not isinstance(data, dict):
                raise RuntimeError(f"'{path}' must contain a top-level JSON object.")
            if path == config.LEGACY_GLOBAL_INDEX_PATH:
                data.pop("holds", None)
                config.logger.info(
                    "Loaded index from legacy path '%s'; next commit will write to new structure.",
                    path,
                )
            return data
        except Exception as exc:
            last_error = exc
            err_msg = str(exc).lower()
            if "404" in str(exc) or "not found" in err_msg or "no file" in err_msg:
                continue
            raise
    if last_error is not None:
        err_msg = str(last_error).lower()
        if "404" in str(last_error) or "not found" in err_msg or "no file" in err_msg:
            config.logger.warning("No global index file found; bootstrapping new index.")
            return global_index.bootstrap_global_index(repo_id)
        raise RuntimeError(
            f"Could not load global index from '{repo_id}' (tried {config.GLOBAL_INDEX_PATH}, {config.LEGACY_GLOBAL_INDEX_PATH})."
        ) from last_error
    return global_index.bootstrap_global_index(repo_id)


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
            and repo_path.name == config.METADATA_FILENAME
            and repo_path.parent.name != "meta"
        ):
            metadata_paths.append(repo_file)

    metadata_paths.sort()
    return metadata_paths, files_by_directory


def commit_dataset_updates(
    api: HfApi,
    *,
    repo_id: str,
    token: str,
    global_index_payload: dict[str, Any],
    train_jsonl_payload: str,
    metadata_updates: dict[str, dict[str, Any]],
    new_votes_files: dict[str, list[Any]] | None = None,
) -> None:
    operations: list[CommitOperationAdd] = []

    serialized_index = json.dumps(global_index_payload, indent=2, ensure_ascii=False) + "\n"
    operations.append(
        CommitOperationAdd(
            path_in_repo=config.GLOBAL_INDEX_PATH,
            path_or_fileobj=io.BytesIO(serialized_index.encode("utf-8")),
        )
    )

    operations.append(
        CommitOperationAdd(
            path_in_repo=config.TRAIN_JSONL_PATH,
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

    if new_votes_files:
        for path_in_repo, votes_list in sorted(new_votes_files.items()):
            payload = votes_list if isinstance(votes_list, list) else list(votes_list)
            serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            operations.append(
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=io.BytesIO(serialized.encode("utf-8")),
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


def commit_anonymous_contribution(
    *,
    repo_id: str,
    token: str,
    revision: str | None,
    hold_id: str,
    manufacturer: str,
    model: str,
    metadata: dict[str, Any],
    files: list[tuple[str, BinaryIO]],
) -> str:
    """Commit a new anonymous contribution under pending/<uuid>/ and return the commit URL."""
    folder_uuid = str(uuid.uuid4())
    folder_path = f"pending/{folder_uuid}"

    operations: list[CommitOperationAdd] = []

    serialized_metadata = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    operations.append(
        CommitOperationAdd(
            path_in_repo=f"{folder_path}/metadata.json",
            path_or_fileobj=io.BytesIO(serialized_metadata.encode("utf-8")),
        )
    )

    for filename, fileobj in files:
        operations.append(
            CommitOperationAdd(
                path_in_repo=f"{folder_path}/{filename}",
                path_or_fileobj=fileobj,
            )
        )

    api = HfApi(token=token)
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        operations=operations,
        commit_message=f"Add hold {hold_id}",
        commit_description=f"Add climbing hold {hold_id} ({manufacturer} {model})",
        revision=revision,
    )
    return commit_info.commit_url


def commit_vote_updates(
    api: HfApi,
    *,
    repo_id: str,
    token: str,
    hold_votes: dict[str, list[Any]],
    metadata_update: tuple[str, dict[str, Any]] | None = None,
) -> None:
    """Commit per-hold votes.json files, and optionally an updated metadata.json."""
    operations: list[CommitOperationAdd] = []
    for path_in_repo, votes_list in sorted(hold_votes.items()):
        serialized = json.dumps(votes_list, indent=2, ensure_ascii=False) + "\n"
        operations.append(
            CommitOperationAdd(
                path_in_repo=path_in_repo,
                path_or_fileobj=io.BytesIO(serialized.encode("utf-8")),
            )
        )
    if metadata_update is not None:
        metadata_path, metadata_payload = metadata_update
        serialized_metadata = json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n"
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
        commit_message="Update votes (per-hold)",
    )
