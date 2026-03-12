"""Vote payload validation and commit logic for the climbing-holds dataset."""

from __future__ import annotations

import os
import re
from typing import Any

from huggingface_hub import HfApi

import config
import hf_repo

# Rating allowed range (1-5)
RATING_MIN = 1
RATING_MAX = 5
ISO8601_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$"
)


def _validate_rating(value: Any) -> int:
    if not isinstance(value, (int, float)):
        raise ValueError("hold_3d_file_rating must be a number")
    r = int(value) if isinstance(value, float) else value
    if r != value or r < RATING_MIN or r > RATING_MAX:
        raise ValueError(f"hold_3d_file_rating must be an integer between {RATING_MIN} and {RATING_MAX}")
    return r


def _validate_vote_datetime(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("vote_datetime must be a non-empty string")
    s = value.strip()
    if not ISO8601_PATTERN.match(s):
        raise ValueError("vote_datetime must be ISO 8601 format (e.g. 2025-03-12T14:30:00.000Z)")
    return s


def validate_vote_payload(body: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """
    Validate vote payload and build the stored vote entry (no token).
    Returns (vote_entry, hf_token_or_none). Raises ValueError on validation error.
    """
    required = ("hold_id", "hold_manufacturer", "hold_model", "hold_3d_file_rating", "vote_datetime", "anonymous")
    for key in required:
        if key not in body:
            raise ValueError(f"Missing required field: {key}")

    hold_id = body["hold_id"]
    if not isinstance(hold_id, str) or not hold_id.strip():
        raise ValueError("hold_id must be a non-empty string")
    hold_id = hold_id.strip()

    hold_manufacturer = body["hold_manufacturer"]
    if not isinstance(hold_manufacturer, str):
        raise ValueError("hold_manufacturer must be a string")
    hold_manufacturer = hold_manufacturer.strip()

    hold_model = body["hold_model"]
    if not isinstance(hold_model, str):
        raise ValueError("hold_model must be a string")
    hold_model = hold_model.strip()

    rating = _validate_rating(body["hold_3d_file_rating"])
    vote_datetime = _validate_vote_datetime(body["vote_datetime"])
    anonymous = body["anonymous"]
    if not isinstance(anonymous, bool):
        raise ValueError("anonymous must be a boolean")

    hf_token: str | None = None
    if not anonymous:
        t = body.get("hf_token")
        if not t or not isinstance(t, str) or not t.strip():
            raise ValueError("hf_token is required when anonymous is false")
        hf_token = t.strip()

    entry = {
        "hold_id": hold_id,
        "hold_manufacturer": hold_manufacturer,
        "hold_model": hold_model,
        "hold_3d_file_rating": rating,
        "vote_datetime": vote_datetime,
        "anonymous": anonymous,
    }
    return entry, hf_token


def _resolve_commit_token(anonymous: bool, user_token: str | None) -> str:
    """Return token to use for commit: HF_TOKEN for anonymous, else user token or fallback HF_TOKEN."""
    hf_token = os.environ.get("HF_TOKEN")
    if anonymous or not user_token:
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set (required for anonymous votes or when user token is missing)")
        return hf_token
    return user_token


def process_vote(
    api: HfApi,
    repo_id: str,
    revision: str | None,
    vote_entry: dict[str, Any],
    user_token: str | None,
) -> dict[str, Any]:
    """
    Load hold votes, append entry, commit. On commit failure with user token,
    retries with HF_TOKEN. Returns a response dict for the API.
    """
    anonymous = vote_entry.get("anonymous", True)
    token = _resolve_commit_token(anonymous, user_token)
    hold_id = vote_entry["hold_id"]
    hold_votes_path = f"{hold_id}/{config.VOTES_FILENAME}"

    hold_votes: list[Any] = hf_repo.load_json_file_optional(
        repo_id, hold_votes_path, token, revision, default=[]
    )
    if not isinstance(hold_votes, list):
        hold_votes = []

    hold_votes.append(vote_entry)
    hold_votes_map = {hold_votes_path: hold_votes}

    try:
        hf_repo.commit_vote_updates(
            api,
            repo_id=repo_id,
            token=token,
            hold_votes=hold_votes_map,
        )
        return {"status": "success", "message": "Vote recorded"}
    except Exception as exc:
        if not anonymous and user_token and token == user_token:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token and hf_token != user_token:
                config.logger.warning("Commit with user token failed, retrying with HF_TOKEN: %s", exc)
                hf_repo.commit_vote_updates(
                    api,
                    repo_id=repo_id,
                    token=hf_token,
                    hold_votes=hold_votes_map,
                )
                return {"status": "success", "message": "Vote recorded"}
        raise
