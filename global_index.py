"""Global index business logic: bootstrap, allowed references, update, comparison."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import config


def normalize_reference_value(value: Any) -> str | None:
    """Normalize a reference string for comparison (used by ensure_allowed_references)."""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized or None
    return str(value).strip().lower() or None


def bootstrap_global_index(repo_id: str) -> dict[str, Any]:
    config.logger.warning(
        "Remote '%s' is empty. Bootstrapping a new global index with empty allowed references.",
        config.GLOBAL_INDEX_PATH,
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

    for key in config.MANAGED_ATTENTION_KEYS:
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


def compute_train_jsonl_hash(train_jsonl: str) -> str:
    return hashlib.sha256(train_jsonl.encode("utf-8")).hexdigest()


def has_meaningful_changes(current_index: dict[str, Any], updated_index: dict[str, Any]) -> bool:
    return compute_payload_hash(build_comparison_payload(current_index)) != compute_payload_hash(
        build_comparison_payload(updated_index)
    )
