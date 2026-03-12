"""Hold metadata normalization, validation, and rebuild logic."""

from __future__ import annotations

import difflib
import json
from pathlib import PurePosixPath
from typing import Any

import config
import global_index
import hf_repo


def canonical_hold_id(metadata: dict[str, Any], metadata_path: str) -> str:
    folder_name = PurePosixPath(metadata_path).parent.name
    raw_hold_id = metadata.get("hold_id")

    if raw_hold_id is None:
        return folder_name

    hold_id = str(raw_hold_id).strip()
    if not hold_id:
        return folder_name

    if hold_id != folder_name:
        config.logger.warning(
            "Hold ID mismatch for '%s': metadata has '%s', using folder name '%s'.",
            metadata_path,
            hold_id,
            folder_name,
        )
    return folder_name


def infer_mesh_presence(hold_directory: str, files_by_directory: dict[str, list[str]]) -> bool:
    for repo_file in files_by_directory.get(hold_directory, []):
        file_name = PurePosixPath(repo_file).name
        if file_name == config.METADATA_FILENAME:
            continue
        if PurePosixPath(repo_file).suffix.lower() in config.MESH_EXTENSIONS:
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
    normalized_value = global_index.normalize_reference_value(value)
    attention_bucket.add(hold_id)

    if normalized_value is None:
        config.logger.warning("Hold '%s' has a null or empty '%s'.", hold_id, field_name)
        return

    if normalized_value == "unknown":
        config.logger.warning("Hold '%s' still has an unknown '%s'.", hold_id, field_name)
        return

    suggestion = difflib.get_close_matches(
        normalized_value,
        sorted(value for value in allowed_values if value != "unknown"),
        n=1,
        cutoff=0.8,
    )
    if suggestion:
        config.logger.warning(
            "Hold '%s' has an invalid '%s' value '%s'. Did you mean '%s'?",
            hold_id,
            field_name,
            value,
            suggestion[0],
        )
        return

    config.logger.warning(
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
    manufacturer_ref = global_index.normalize_reference_value(manufacturer_value)
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
    hold_type_ref = global_index.normalize_reference_value(hold_type_value)
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
    model_ref = global_index.normalize_reference_value(model_value)
    if model_ref in {None, "unknown"}:
        needs_attention["unknown_model"].add(hold_id)
        config.logger.warning("Hold '%s' still has an unknown 'model'.", hold_id)

    status_value = metadata.get("status")
    status_ref = global_index.normalize_reference_value(status_value)
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
    for managed_key in config.MANAGED_ATTENTION_KEYS:
        needs_attention[managed_key] = set()

    holds: list[dict[str, Any]] = []
    metadata_updates: dict[str, dict[str, Any]] = {}

    for metadata_path in metadata_paths:
        hold_directory = str(PurePosixPath(metadata_path).parent)
        hold_folder = PurePosixPath(metadata_path).parent.name

        try:
            metadata = hf_repo.load_json_file(repo_id, metadata_path, token, revision)
        except RuntimeError as exc:
            config.logger.warning("%s Skipping file.", exc)
            needs_attention["invalid_metadata"].add(hold_folder)
            continue

        if not isinstance(metadata, dict):
            config.logger.warning(
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
            config.logger.warning(
                "Hold '%s' does not have a mesh file next to '%s'.",
                hold_id,
                metadata_path,
            )

        holds.append(metadata_copy)

    holds.sort(key=lambda item: str(item.get("hold_id", "")))
    return holds, needs_attention, metadata_updates


def build_train_jsonl(holds: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for hold in sorted(holds, key=lambda item: str(item.get("hold_id", ""))):
        lines.append(json.dumps(hold, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n"
