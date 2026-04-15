"""Anonymous climbing hold contribution endpoint."""

from __future__ import annotations

import json
import os
import re
from typing import Annotated

from fastapi import Form, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

import config
import hf_repo


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _deduplicate_filenames(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            result.append(name)
        else:
            seen[name] += 1
            count = seen[name]
            if "." in name:
                stem, _, ext = name.rpartition(".")
                result.append(f"{stem}-{count}.{ext}")
            else:
                result.append(f"{name}-{count}")
    return result


async def handle_anonymous_contribution(
    hold_id: Annotated[str, Form()],
    id: Annotated[int, Form()],
    manufacturer: Annotated[str, Form()],
    model: Annotated[str, Form()],
    type: Annotated[str, Form()],
    size: Annotated[str, Form()],
    labels: Annotated[str, Form()],
    created_at: Annotated[int, Form()],
    last_update: Annotated[int, Form()],
    timezone_offset: Annotated[str, Form()],
    note: Annotated[str | None, Form()] = None,
    color_of_scan: Annotated[str, Form()] = "",
    available_colors: Annotated[str, Form()] = "[]",
    files: list[UploadFile] | None = None,
) -> JSONResponse:
    token = os.environ.get("HF_ANONYMOUS_TOKEN")
    if not token:
        config.logger.error("HF_ANONYMOUS_TOKEN is not set.")
        return JSONResponse({"detail": "Server misconfiguration: missing HF_ANONYMOUS_TOKEN"}, status_code=500)

    try:
        labels_list = json.loads(labels)
        if not isinstance(labels_list, list):
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        return JSONResponse({"detail": "Field 'labels' must be a JSON array string"}, status_code=422)

    try:
        available_colors_list = json.loads(available_colors)
        if not isinstance(available_colors_list, list):
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        return JSONResponse({"detail": "Field 'available_colors' must be a JSON array string"}, status_code=422)

    metadata = {
        "hold_id": hold_id,
        "id": id,
        "manufacturer": manufacturer,
        "model": model,
        "type": type,
        "size": size,
        "labels": labels_list,
        "created_at": created_at,
        "last_update": last_update,
        "timezone_offset": timezone_offset,
        "note": note or "",
        "color_of_scan": color_of_scan,
        "available_colors": available_colors_list,
    }

    uploads = files or []
    raw_names = [_sanitize_filename(f.filename or f"file_{i}") for i, f in enumerate(uploads)]
    deduped_names = _deduplicate_filenames(raw_names)

    file_pairs: list[tuple[str, bytes]] = []
    for upload_file, filename in zip(uploads, deduped_names):
        content = await upload_file.read()
        file_pairs.append((filename, content))

    repo_id = os.environ.get("HF_ANONYMOUS_REPO_ID", config.ANONYMOUS_REPO_ID_DEFAULT)
    revision = os.environ.get("HF_REVISION") or "staging"

    try:
        commit_url = await run_in_threadpool(
            hf_repo.commit_anonymous_contribution,
            repo_id=repo_id,
            token=token,
            revision=revision,
            hold_id=hold_id,
            manufacturer=manufacturer,
            model=model,
            metadata=metadata,
            files=file_pairs,
        )
    except Exception as exc:
        config.logger.exception("Anonymous contribution commit failed: %s", exc)
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return JSONResponse({"commit_url": commit_url})
