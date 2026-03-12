#!/usr/bin/env python3
"""Rebuild and upload the Hugging Face dataset global index via Webhook.

Deploy this file as app.py in a Hugging Face Space (Gradio SDK).
Configure the HF webhook to point to: https://YOUR-SPACE.hf.space/webhooks/index
Set Space secrets: HF_TOKEN (write access to the dataset), WEBHOOK_SECRET (same as in webhook settings).

POST /vote accepts a JSON body to record a user vote (see votes.py for payload shape).
"""

from __future__ import annotations

import os
import threading

from fastapi import Request
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi

import config
from webhooks import app
import votes


async def _handle_vote(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    try:
        vote_entry, hf_token = votes.validate_vote_payload(body)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    try:
        api = HfApi()
        revision = os.environ.get("HF_REVISION")
        result = votes.process_vote(
            api, config.DEFAULT_REPO_ID, revision, vote_entry, hf_token
        )
        return JSONResponse(result)
    except Exception as e:
        config.logger.exception("Vote failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


def _launch_with_vote_route():
    app.launch(prevent_thread_lock=True)
    app.fastapi_app.post("/vote")(_handle_vote)
    # Block main thread: WebhooksServer may not expose _ui (e.g. when using default UI)
    ui = getattr(app, "_ui", None)
    if ui is not None and hasattr(ui, "block_thread"):
        ui.block_thread()
    else:
        threading.Event().wait()


if __name__ == "__main__":
    _launch_with_vote_route()

# Hugging Face Spaces (Gradio SDK) expects a top-level Blocks app named `demo`.
demo = app
