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


app.app.post("/vote")(_handle_vote)
app.launch(ssr_mode=False, prevent_thread_lock=True)
threading.Event().wait()
