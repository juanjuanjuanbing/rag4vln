# -*- coding: utf-8 -*-
"""Open KB via ``kb.KnowledgeBase``; build via functions in ``kb_build``."""

from .kb import (
    DOCUMENTS_FILENAME,
    KnowledgeBase,
    MANIFEST_FILENAME,
    SCENES_DIRNAME,
    SOURCE_EXPERIENCE,
    SOURCE_MEMORY,
    scene_json_path,
)
from .kb_build import (
    attach_view_images_to_kb,
    build_knowledgebase_from_memory,
    create_kb_folder,
    flatten_kb_documents,
    flatten_scene_to_docs,
    make_manifest,
    make_scene_description,
    save_manifest,
)

__all__ = [
    "KnowledgeBase",
    "build_knowledgebase_from_memory",
    "attach_view_images_to_kb",
    "flatten_kb_documents",
    "flatten_scene_to_docs",
    "SOURCE_MEMORY",
    "SOURCE_EXPERIENCE",
    "make_manifest",
    "make_scene_description",
    "save_manifest",
    "create_kb_folder",
    "MANIFEST_FILENAME",
    "SCENES_DIRNAME",
    "DOCUMENTS_FILENAME",
    "scene_json_path",
]
