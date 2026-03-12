#!/usr/bin/env python3
"""Rebuild and upload the Hugging Face dataset global index via Webhook.

Deploy this file as app.py in a Hugging Face Space (Gradio SDK).
Configure the HF webhook to point to: https://YOUR-SPACE.hf.space/webhooks/indexation
Set Space secrets: HF_TOKEN (write access to the dataset), WEBHOOK_SECRET (same as in webhook settings).
"""

from __future__ import annotations

from webhooks import app

if __name__ == "__main__":
    app.launch()

# Hugging Face Spaces (Gradio SDK) expects a top-level Blocks app named `demo`.
demo = app
