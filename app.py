#!/usr/bin/env python3
"""Rebuild and upload the Hugging Face dataset global index via Webhook.

Deploy this file as app.py in a Hugging Face Space (Gradio SDK).
Configure the HF webhook to point to: https://YOUR-SPACE.hf.space/webhooks/index
Set Space secrets: HF_TOKEN (write access to the dataset), WEBHOOK_SECRET (same as in webhook settings).

POST /webhooks/vote accepts a JSON body to record a user vote (see votes.py for payload shape).
"""

from __future__ import annotations

import os

import uvicorn
from fastapi import Depends, Request
from fastapi.responses import JSONResponse
from huggingface_hub import HfApi

import config
import votes
import gradio as gr

from webhooks import app, demo, verify_webhook_secret


@app.post("/webhooks/vote", dependencies=[Depends(verify_webhook_secret)])
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


gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
