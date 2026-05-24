# Copyright (c) Opendatalab. All rights reserved.
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
from mineru.utils.enum_class import BlockType


class _TitleLevel(BaseModel):
    id: int
    level: Literal[1, 2, 3, 4]


class _TitleLevelResponse(BaseModel):
    levels: list[_TitleLevel]


TITLE_BLOCK_TYPES = {
    BlockType.TITLE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}
MAX_TITLE_GROUP_WORKERS = 4


def _get_title_line_avg_height(block):
    if "line_avg_height" in block:
        return block["line_avg_height"]

    title_block_line_height_list = []
    for line in block.get("lines", []):
        bbox = line["bbox"]
        title_block_line_height_list.append(int(bbox[3] - bbox[1]))

    if len(title_block_line_height_list) > 0:
        return sum(title_block_line_height_list) / len(title_block_line_height_list)

    return int(block["bbox"][3] - block["bbox"][1])


def _collect_title_block_refs(page_info_list):
    title_block_refs = []
    title_types = set()

    for page_info in page_info_list:
        for block in page_info.get("para_blocks", []):
            block_type = block.get("type")
            if block_type in TITLE_BLOCK_TYPES:
                title_block_refs.append((page_info, block))
                title_types.add(block_type)

    return title_block_refs, title_types


def _build_title_dict(title_block_refs):
    title_dict = {}

    for i, (page_info, block) in enumerate(title_block_refs):
        title_dict[str(i)] = [
            merge_para_with_text(block),
            _get_title_line_avg_height(block),
            int(page_info["page_idx"]) + 1,
        ]

    return title_dict


def _build_title_optimize_prompt(title_dict):
    return f"""
    You are given a JSON object whose entries are every heading found in a single document. Each value is a list [heading_text, line_height, page_number]. Line_height is the average line height of the block containing the heading (larger usually = higher-level heading).

    Assign each heading an integer level from 1 to 4, where 1 is the top of the hierarchy (e.g. parts / chapters) and 4 is the deepest sub-heading.

    Rules:
    - Use the heading text semantics AND the line_height as your signals.
    - Levels must be continuous: never jump from level 1 directly to level 3 — go through 2 first.
    - Maximum depth is 4.
    - Return EVERY id from the input exactly once. Do not add, drop, rename, or merge ids.
    - The "levels" array must have the same number of entries as the input dict.

    Input:
    {title_dict}
    """


def _build_relative_title_optimize_prompt(title_dict):
    return f"""
    You are given a JSON object whose entries are every chapter / section / sub-section heading from a single document, EXCLUDING the document's main title (which has already been identified by the system and assigned level 1).

    Each value is a list [heading_text, line_height, page_number]. Line_height is the average line height of the block containing the heading (larger usually = higher-level heading).

    Assign each heading an integer level from 1 to 4, where 1 is the top of the hierarchy and 4 is the deepest sub-heading.

    Rules:
    - Use the heading text semantics AND the line_height as your signals.
    - Levels must be continuous: never jump from level 1 directly to level 3 — go through 2 first.
    - Maximum depth is 4.
    - Return EVERY id from the input exactly once. Do not add, drop, rename, or merge ids.
    - The "levels" array must have the same number of entries as the input dict.

    Input:
    {title_dict}
    """


def _request_title_levels(title_aided_config, title_dict, prompt_builder=None):
    if len(title_dict) == 0:
        return {}

    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )

    expected_keys = set(range(len(title_dict)))
    if prompt_builder is None:
        prompt_builder = _build_title_optimize_prompt
    title_optimize_prompt = prompt_builder(title_dict)

    logger.debug(f"Requesting LLM for title optimization with prompt: {title_optimize_prompt}")

    parse_params = {
        "model": title_aided_config["model"],
        "messages": [{"role": "user", "content": title_optimize_prompt}],
        "temperature": 0,
        "response_format": _TitleLevelResponse,
    }
    if "enable_thinking" in title_aided_config:
        parse_params["extra_body"] = {
            "enable_thinking": title_aided_config["enable_thinking"]
        }

    for retry in range(3):
        try:
            completion = client.chat.completions.parse(**parse_params)
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                logger.warning("Model returned no parsed payload.")
                continue

            result = {entry.id: entry.level for entry in parsed.levels}
            if set(result.keys()) == expected_keys:
                return result

            logger.warning(
                "The ids in the optimized title result do not match the input titles "
                f"(missing={expected_keys - result.keys()}, extra={result.keys() - expected_keys})."
            )
        except Exception as e:
            logger.exception(e)

    logger.error("Failed to obtain valid title levels after maximum retries.")
    return None


def _apply_levels_to_blocks(title_block_refs, levels_by_index):
    if levels_by_index is None:
        return

    for i, (_, block) in enumerate(title_block_refs):
        block["level"] = int(levels_by_index[i])


def _normalize_title_types(title_block_refs):
    for _, block in title_block_refs:
        if block.get("type") in [BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE]:
            block["type"] = BlockType.TITLE


def _get_title_block_identity(block):
    block_index = block.get("index")
    if block_index is not None:
        return ("index", block_index)

    return (
        "bbox_text",
        tuple(block.get("bbox", [])),
        merge_para_with_text(block),
    )


def _sync_para_titles_to_preproc(page_info_list):
    for page_info in page_info_list:
        para_title_map = {}
        for block in page_info.get("para_blocks", []):
            if block.get("type") in TITLE_BLOCK_TYPES:
                para_title_map[_get_title_block_identity(block)] = block

        if len(para_title_map) == 0:
            continue

        for block in page_info.get("preproc_blocks", []):
            if block.get("type") not in TITLE_BLOCK_TYPES:
                continue

            para_block = para_title_map.get(_get_title_block_identity(block))
            if para_block is None:
                continue

            block["type"] = para_block.get("type", block.get("type"))
            if "level" in para_block:
                block["level"] = para_block["level"]


def _run_single_pass_title_leveling(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(title_aided_config, title_dict)
    _apply_levels_to_blocks(title_block_refs, levels_by_index)


def _split_paragraph_title_groups(title_block_refs):
    groups = []
    current_group = []

    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            if current_group:
                groups.append(current_group)
                current_group = []
        elif block.get("type") == BlockType.PARAGRAPH_TITLE:
            current_group.append(title_ref)

    if current_group:
        groups.append(current_group)

    return groups


def _offset_paragraph_title_levels(levels_by_index):
    if not levels_by_index:
        return levels_by_index

    return {
        index: 2 if level == 1 else level
        for index, level in levels_by_index.items()
    }


def _request_paragraph_group_levels(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(
        title_aided_config,
        title_dict,
        prompt_builder=_build_relative_title_optimize_prompt,
    )
    return _offset_paragraph_title_levels(levels_by_index)


def _run_grouped_title_leveling(title_block_refs, title_aided_config):
    doc_title_refs = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)

    paragraph_title_groups = _split_paragraph_title_groups(title_block_refs)
    group_levels = []

    if len(paragraph_title_groups) > 1:
        max_workers = min(len(paragraph_title_groups), MAX_TITLE_GROUP_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _request_paragraph_group_levels,
                    title_group,
                    title_aided_config,
                )
                for title_group in paragraph_title_groups
            ]
            group_levels = [future.result() for future in futures]
    else:
        group_levels = [
            _request_paragraph_group_levels(title_group, title_aided_config)
            for title_group in paragraph_title_groups
        ]

    for title_group, levels_by_index in zip(paragraph_title_groups, group_levels):
        _apply_levels_to_blocks(title_group, levels_by_index)

    _normalize_title_types(doc_title_refs)
    for title_group in paragraph_title_groups:
        _normalize_title_types(title_group)


def llm_aided_title(page_info_list, title_aided_config):
    title_block_refs, title_types = _collect_title_block_refs(page_info_list)
    if len(title_block_refs) == 0:
        logger.info("No titles detected, skipping LLM-aided title optimization.")
        return

    has_doc_title = BlockType.DOC_TITLE in title_types
    has_paragraph_title = BlockType.PARAGRAPH_TITLE in title_types
    has_generic_title = BlockType.TITLE in title_types

    if has_doc_title and has_paragraph_title and not has_generic_title:
        _run_grouped_title_leveling(title_block_refs, title_aided_config)
        _sync_para_titles_to_preproc(page_info_list)
        return

    doc_title_refs = []
    title_refs_for_llm = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)
        else:
            title_refs_for_llm.append(title_ref)

    if len(title_refs_for_llm) > 0:
        _run_single_pass_title_leveling(title_refs_for_llm, title_aided_config)

    _normalize_title_types(doc_title_refs)
    _normalize_title_types(title_refs_for_llm)
    _sync_para_titles_to_preproc(page_info_list)
