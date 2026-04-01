"""Standalone re-indexation script for the climbing-holds dataset.

Dry-run by default: shows what would be committed without touching the repo.
Pass --commit to actually push changes to Hugging Face.

Usage:
  HF_TOKEN=xxx python reindex.py                              # dry-run on main
  HF_TOKEN=xxx python reindex.py --commit                     # commit on main
  HF_TOKEN=xxx python reindex.py --repo-id foo/bar --revision staging --commit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import PurePosixPath

from huggingface_hub import HfApi

import config
import global_index
import hf_repo
import holds as holds_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("reindex")


def run(*, repo_id: str, revision: str, token: str, commit: bool) -> None:
    api = HfApi(token=token)
    mode_label = "COMMIT" if commit else "DRY-RUN"
    logger.info("[%s] repo=%s  revision=%s", mode_label, repo_id, revision)

    current_index = hf_repo.load_global_index(repo_id, token, revision)
    if not isinstance(current_index, dict):
        raise RuntimeError("global_index must be a top-level JSON object.")

    allowed_references = global_index.ensure_allowed_references(current_index)
    metadata_paths, files_by_directory = hf_repo.list_dataset_files(api, repo_id, token, revision)
    logger.info("Found %d metadata files.", len(metadata_paths))

    initial_attention = global_index.prepare_initial_attention(current_index)
    holds_list, needs_attention, metadata_updates = holds_module.rebuild_holds(
        repo_id=repo_id,
        token=token,
        revision=revision,
        metadata_paths=metadata_paths,
        files_by_directory=files_by_directory,
        allowed_references=allowed_references,
        initial_attention=initial_attention,
    )

    updated_index = global_index.update_global_index(current_index, holds_list, needs_attention)
    train_jsonl_payload = holds_module.build_train_jsonl(holds_list)
    updated_index["stats"]["train_jsonl_hash"] = global_index.compute_train_jsonl_hash(train_jsonl_payload)

    new_votes_files: dict[str, list] = {}
    for metadata_path in metadata_paths:
        hold_dir = str(PurePosixPath(metadata_path).parent)
        votes_path = f"{hold_dir}/{config.VOTES_FILENAME}"
        if votes_path not in files_by_directory.get(hold_dir, []):
            new_votes_files[votes_path] = []

    total_holds = updated_index["stats"]["total_holds"]
    to_identify = updated_index["stats"]["to_identify"]
    has_changes = global_index.has_meaningful_changes(current_index, updated_index)

    logger.info("Results: %d holds total, %d to identify.", total_holds, to_identify)
    if metadata_updates:
        logger.info("Metadata normalization: %d file(s) would be updated.", len(metadata_updates))
        for path in sorted(metadata_updates):
            logger.info("  ~ %s", path)
    if new_votes_files:
        logger.info("Missing votes.json: %d file(s) would be created.", len(new_votes_files))
        for path in sorted(new_votes_files):
            logger.info("  + %s", path)
    if not has_changes and not metadata_updates and not new_votes_files:
        logger.info("No changes detected — nothing to commit.")
        return

    if has_changes:
        logger.info(
            "Index diff (needs_attention):\n%s",
            json.dumps(updated_index.get("needs_attention", {}), indent=2, ensure_ascii=False),
        )

    if not commit:
        logger.info(
            "[DRY-RUN] Would commit %d hold(s) to %s@%s. Pass --commit to push.",
            total_holds,
            repo_id,
            revision,
        )
        return

    hf_repo.commit_dataset_updates(
        api,
        repo_id=repo_id,
        token=token,
        revision=revision,
        global_index_payload=updated_index,
        train_jsonl_payload=train_jsonl_payload,
        metadata_updates=metadata_updates,
        new_votes_files=new_votes_files if new_votes_files else None,
    )
    logger.info(
        "[COMMIT] Done — %d holds, %d to identify. Committed to %s@%s.",
        total_holds,
        to_identify,
        repo_id,
        revision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-index the climbing-holds dataset on Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo-id",
        default=config.DEFAULT_REPO_ID,
        help=f"Dataset repo ID (default: {config.DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to operate on (default: main)",
    )
    parser.add_argument(
        "--commit",
        "--force",
        action="store_true",
        dest="commit",
        help="Actually push changes to Hugging Face (default: dry-run only)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN environment variable is not set.")
        sys.exit(1)

    try:
        run(repo_id=args.repo_id, revision=args.revision, token=token, commit=args.commit)
    except Exception as exc:
        logger.exception("Re-indexation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
