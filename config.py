"""Constants, paths, and logging configuration for the climbing-holds dataset indexer."""

from __future__ import annotations

import logging

DEFAULT_REPO_ID = "setrsoft/climbing-holds"
GLOBAL_INDEX_PATH = "meta/global_index.json"
GLOBAL_VOTES_PATH = "meta/votes.json"
LEGACY_GLOBAL_INDEX_PATH = "global_index.json"
TRAIN_JSONL_PATH = "train.jsonl"
METADATA_FILENAME = "metadata.json"
VOTES_FILENAME = "votes.json"
MESH_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl"}
MANAGED_ATTENTION_KEYS = {
    "invalid_hold_type_reference",
    "invalid_manufacturer_reference",
    "invalid_metadata",
    "invalid_status_reference",
    "missing_mesh",
    "unknown_hold_type",
    "unknown_manufacturer",
    "unknown_model",
}

logger = logging.getLogger("update_global_index")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
