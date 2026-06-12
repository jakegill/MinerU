# Copyright (c) Opendatalab. All rights reserved.
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from loguru import logger
from openai import OpenAI, RateLimitError
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

    max_retries = 5
    for retry in range(max_retries):
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
        except RateLimitError as e:
            if retry == max_retries - 1:
                logger.exception(e)
                break
            delay = min(2 ** retry, 30) + random.uniform(0, 1)
            logger.warning(
                f"Vertex rate limited (429); backing off {delay:.1f}s "
                f"(retry {retry + 1}/{max_retries})."
            )
            time.sleep(delay)
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


class _TextPruneVerdict(BaseModel):
    id: int
    verdict: Literal["keep", "prune"]


class _TextPruneResponse(BaseModel):
    verdicts: list[_TextPruneVerdict]


class _FormulaVerdict(BaseModel):
    id: int
    verdict: Literal["keep", "fix", "prune"]
    fixed_latex: str = ""


class _FormulaResponse(BaseModel):
    verdicts: list[_FormulaVerdict]


MAX_PRUNE_WORKERS = 4
TEXT_PRUNE_AUTO_KEEP_MIN_CHARS = 300
TEXT_PRUNE_EDGE_AUTO_KEEP_MIN_CHARS = 600
TEXT_PRUNE_EDGE_Y_BAND = 0.15
TEXT_PRUNE_REPEAT_MIN_COUNT = 3
TEXT_PRUNE_HEAD_CHARS = 180
TEXT_PRUNE_TAIL_CHARS = 40
TEXT_PRUNE_CHUNK_SIZE = 150
FORMULA_CHUNK_SIZE = 50
FORMULA_AUTO_KEEP_MAX_LATEX_CHARS = 4000
FORMULA_FIX_LEN_RATIO_MIN = 0.5
FORMULA_FIX_LEN_RATIO_MAX = 2.5


def _build_text_prune_prompt(candidate_dict):
    return f"""
    You are given a JSON object whose entries are text paragraphs extracted from a single document by a PDF layout-analysis pipeline. Each value is a list [text, page_number, y_position, repeat_count]:
    - text: the paragraph text (may be truncated, marked by "[…]").
    - page_number: the 1-based page the paragraph appears on.
    - y_position: vertical position of the paragraph on its page, where 0.0 is the top edge and 1.0 is the bottom edge.
    - repeat_count: how many near-identical occurrences of this text exist across the whole document.

    The layout system already removed obvious headers and footers; these paragraphs ESCAPED that filter. Most entries are legitimate content and only a few are junk. For each id, return verdict "prune" or "keep".

    PRUNE only clear junk:
    - Running heads or running footers: a book title, chapter title, or author name repeated across many pages (high repeat_count, y_position near 0.0 or 1.0).
    - Bare page numbers or folio lines captured as body text (e.g. "234", "xii", "Page 57 of 320").
    - Watermarks and stamps (e.g. "Downloaded from ...", library or website stamps), usually repeated across pages.
    - Publisher or copyright boilerplate repeated on many pages. A single ordinary copyright page is normal book content — do not prune it.
    - Garbled OCR noise: text that is mostly random characters and symbols with no recoverable meaning in any language.

    KEEP everything else. In particular, KEEP:
    - Short lines of dialogue, even one word long.
    - Poetry, verse, epigraphs, and attribution lines (e.g. "— Oscar Wilde").
    - List items, table-of-contents lines, and index entries.
    - Scene-break markers (e.g. "***"), section transitions, datelines, signatures.
    - Footnotes, references, captions, and short factual statements.
    - Text in any language or script, even if you cannot read it. Unfamiliar language is not garbled OCR; garbled OCR is symbol salad in any script.
    - Mathematical notation or variable names inside text. Math is not noise.

    Rules:
    - When uncertain, KEEP. A wrongly pruned paragraph is permanently lost; a kept junk line is only a cosmetic flaw.
    - A high repeat_count or an edge y_position is supporting evidence, never sufficient by itself: refrains and recurring phrases repeat legitimately, and footnotes legitimately sit at the bottom of the page. The text itself must look like junk.
    - Judge each id independently.
    - Return EVERY id from the input exactly once. Do not add, drop, rename, or merge ids.
    - The "verdicts" array must have the same number of entries as the input dict.

    Input:
    {candidate_dict}
    """


def _build_formula_prune_prompt(formula_dict):
    return f"""
    You are given a JSON object whose entries are display-mode LaTeX formulas extracted from a single document by a formula-recognition model. Each value is a list [latex, page_number].

    The document is rendered with KaTeX, so a formula that does not compile under KaTeX visibly breaks the page. Some entries are not formulas at all (layout noise misdetected as a formula), and some are real formulas whose LaTeX will not render. For each id, return exactly one verdict:

    - "keep": the entry is a plausible formula and its LaTeX is valid KaTeX. Most entries should be kept.
    - "fix": the entry is a real formula but its LaTeX has recognition errors or will not render under KaTeX. Also return the corrected LaTeX in "fixed_latex".
    - "prune": the entry is not a formula at all, or is so garbled that no faithful correction is possible.

    Rules for "fix" — apply ONLY these kinds of corrections:
    - Fix syntax errors: mismatched, missing, or extra tokens (braces, brackets, \\left/\\right pairs, \\begin/\\end environments). The corrected result must conform to LaTeX math syntax principles.
    - Replace commands or environments KaTeX does not support with KaTeX-supported equivalents, or remove them if they only affect styling.
    - Preserve all information from the original formula. Do not add any symbol, term, or content that is not present in the original. Do not simplify, re-derive, or restyle the mathematics.
    - "fixed_latex" must contain only the corrected formula — no surrounding $ or $$ delimiters, no explanation, no metadata.

    Rules for "prune":
    - Prune entries that are clearly not formulas: fragments of prose, page numbers, running heads, table scraps, or stray lines and punctuation.
    - Very short expressions (a single variable, number, or symbol) can be legitimate formulas in context. If an entry could plausibly be a real formula, prefer "keep" or "fix" over "prune".

    General rules:
    - When uncertain between "keep" and "fix", choose "keep". When uncertain between "fix" and "prune", choose "fix". Pruning destroys content.
    - "fixed_latex" must be non-empty if and only if the verdict is "fix"; leave it "" otherwise.
    - Return EVERY id from the input exactly once. Do not add, drop, rename, or merge ids.
    - The "verdicts" array must have the same number of entries as the input dict.

    Input:
    {formula_dict}
    """


def _request_verdicts(llm_config, prompt, response_format, expected_keys):
    """Same transport contract as _request_title_levels: structured output at
    temperature 0, exact-id validation, 5 retries with 429 backoff, None on
    total failure (callers treat None as keep-everything)."""
    client = OpenAI(
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
    )

    parse_params = {
        "model": llm_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": response_format,
    }
    if "enable_thinking" in llm_config:
        parse_params["extra_body"] = {
            "enable_thinking": llm_config["enable_thinking"]
        }

    max_retries = 5
    for retry in range(max_retries):
        try:
            completion = client.chat.completions.parse(**parse_params)
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                logger.warning("Model returned no parsed payload.")
                continue

            result = {entry.id: entry for entry in parsed.verdicts}
            if set(result.keys()) == expected_keys:
                return result

            logger.warning(
                "The ids in the verdict result do not match the input "
                f"(missing={expected_keys - result.keys()}, extra={result.keys() - expected_keys})."
            )
        except RateLimitError as e:
            if retry == max_retries - 1:
                logger.exception(e)
                break
            delay = min(2 ** retry, 30) + random.uniform(0, 1)
            logger.warning(
                f"Vertex rate limited (429); backing off {delay:.1f}s "
                f"(retry {retry + 1}/{max_retries})."
            )
            time.sleep(delay)
        except Exception as e:
            logger.exception(e)

    logger.error("Failed to obtain valid verdicts after maximum retries.")
    return None


def _chunk_refs(refs, chunk_size):
    return [refs[i: i + chunk_size] for i in range(0, len(refs), chunk_size)]


def _run_chunked_requests(chunks, request_fn):
    if len(chunks) > 1:
        max_workers = min(len(chunks), MAX_PRUNE_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(request_fn, chunk) for chunk in chunks]
            return [future.result() for future in futures]
    return [request_fn(chunk) for chunk in chunks]


def _remove_pruned_blocks(pruned_refs):
    pruned_by_page = {}
    for page_info, block in pruned_refs:
        pruned_by_page.setdefault(id(page_info), (page_info, []))[1].append(block)

    for page_info, blocks in pruned_by_page.values():
        pruned_ids = {id(b) for b in blocks}
        # Hard-remove from para_blocks: content_list concatenates para_blocks
        # with discarded_blocks, so moving them there would leak junk back in.
        page_info["para_blocks"] = [
            b for b in page_info["para_blocks"] if id(b) not in pruned_ids
        ]
        page_info.setdefault("pruned_blocks", []).extend(blocks)


def _normalize_repeat_key(text):
    return "".join(
        ch for ch in text.casefold() if not ch.isdigit() and not ch.isspace()
    )


def _compute_text_repeat_counts(page_info_list):
    counts = {}
    for page_info in page_info_list:
        for block in page_info.get("para_blocks", []):
            if block.get("type") != BlockType.TEXT:
                continue
            key = _normalize_repeat_key(merge_para_with_text(block))
            if key:
                counts[key] = counts.get(key, 0) + 1
    return counts


def _truncate_prune_payload_text(text):
    if len(text) <= TEXT_PRUNE_HEAD_CHARS + TEXT_PRUNE_TAIL_CHARS:
        return text
    return (
        f"{text[:TEXT_PRUNE_HEAD_CHARS]} […] {text[-TEXT_PRUNE_TAIL_CHARS:]}"
    )


def _normalized_block_y(page_info, block):
    page_height = page_info.get("page_size", [0, 0])[1]
    bbox = block.get("bbox")
    if not page_height or not bbox:
        return 0.5
    return round((bbox[1] + bbox[3]) / 2 / page_height, 3)


def _collect_text_prune_candidates(page_info_list, repeat_counts):
    candidates = []
    for page_info in page_info_list:
        for block in page_info.get("para_blocks", []):
            if block.get("type") != BlockType.TEXT:
                continue
            text = merge_para_with_text(block)
            if not text.strip():
                continue

            repeat_count = repeat_counts.get(_normalize_repeat_key(text), 1)
            y = _normalized_block_y(page_info, block)
            is_edge = (
                y <= TEXT_PRUNE_EDGE_Y_BAND or y >= 1 - TEXT_PRUNE_EDGE_Y_BAND
            )
            is_candidate = (
                len(text) < TEXT_PRUNE_AUTO_KEEP_MIN_CHARS
                or repeat_count >= TEXT_PRUNE_REPEAT_MIN_COUNT
                or (is_edge and len(text) < TEXT_PRUNE_EDGE_AUTO_KEEP_MIN_CHARS)
            )
            if not is_candidate:
                continue

            payload = [
                _truncate_prune_payload_text(text),
                int(page_info["page_idx"]) + 1,
                y,
                repeat_count,
            ]
            candidates.append((page_info, block, payload))
    return candidates


def _request_text_prune_verdicts(text_aided_config, chunk):
    candidate_dict = {str(i): payload for i, (_, _, payload) in enumerate(chunk)}
    return _request_verdicts(
        text_aided_config,
        _build_text_prune_prompt(candidate_dict),
        _TextPruneResponse,
        set(range(len(chunk))),
    )


def llm_aided_text_prune(page_info_list, text_aided_config):
    repeat_counts = _compute_text_repeat_counts(page_info_list)
    candidates = _collect_text_prune_candidates(page_info_list, repeat_counts)
    if len(candidates) == 0:
        logger.info("No text prune candidates, skipping LLM-aided text pruning.")
        return

    chunks = _chunk_refs(candidates, TEXT_PRUNE_CHUNK_SIZE)
    chunk_verdicts = _run_chunked_requests(
        chunks,
        lambda chunk: _request_text_prune_verdicts(text_aided_config, chunk),
    )

    pruned_refs = []
    failed_chunks = 0
    for chunk, verdicts in zip(chunks, chunk_verdicts):
        if verdicts is None:
            failed_chunks += 1
            continue
        for i, (page_info, block, _) in enumerate(chunk):
            if verdicts[i].verdict == "prune":
                pruned_refs.append((page_info, block))

    _remove_pruned_blocks(pruned_refs)
    logger.info(
        f"LLM text prune: {len(candidates)} candidates, "
        f"{len(pruned_refs)} pruned, {failed_chunks}/{len(chunks)} chunks failed."
    )


def _interline_equation_span(block):
    lines = block.get("lines", [])
    if not lines:
        return None
    spans = lines[0].get("spans", [])
    if not spans:
        return None
    return spans[0]


def _collect_formula_candidates(page_info_list):
    # Empty-LaTeX interline blocks render as cut images downstream; there is
    # nothing to judge or fix, so they are left alone.
    candidates = []
    for page_info in page_info_list:
        for block in page_info.get("para_blocks", []):
            if block.get("type") != BlockType.INTERLINE_EQUATION:
                continue
            span = _interline_equation_span(block)
            if span is None:
                continue
            latex = span.get("content", "")
            if not latex or len(latex) > FORMULA_AUTO_KEEP_MAX_LATEX_CHARS:
                continue
            candidates.append(
                (page_info, block, [latex, int(page_info["page_idx"]) + 1])
            )
    return candidates


def _validate_fixed_latex(original, fixed):
    fixed = fixed.strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if (
            fixed.startswith(left)
            and fixed.endswith(right)
            and len(fixed) > len(left) + len(right)
        ):
            fixed = fixed[len(left): -len(right)].strip()

    if not fixed or fixed == original or "```" in fixed:
        return None
    ratio = len(fixed) / max(len(original), 1)
    if not (FORMULA_FIX_LEN_RATIO_MIN <= ratio <= FORMULA_FIX_LEN_RATIO_MAX):
        return None
    unescaped = fixed.replace("\\{", "").replace("\\}", "")
    if unescaped.count("{") != unescaped.count("}"):
        return None
    if fixed.count("\\begin") != fixed.count("\\end"):
        return None
    if fixed.count("\\left") != fixed.count("\\right"):
        return None
    return fixed


def _request_formula_verdicts(formula_aided_config, chunk):
    formula_dict = {str(i): payload for i, (_, _, payload) in enumerate(chunk)}
    return _request_verdicts(
        formula_aided_config,
        _build_formula_prune_prompt(formula_dict),
        _FormulaResponse,
        set(range(len(chunk))),
    )


def llm_aided_formula_prune(page_info_list, formula_aided_config):
    candidates = _collect_formula_candidates(page_info_list)
    if len(candidates) == 0:
        logger.info("No formula candidates, skipping LLM-aided formula pruning.")
        return

    chunks = _chunk_refs(candidates, FORMULA_CHUNK_SIZE)
    chunk_verdicts = _run_chunked_requests(
        chunks,
        lambda chunk: _request_formula_verdicts(formula_aided_config, chunk),
    )

    pruned_refs = []
    fixed_count = 0
    failed_chunks = 0
    for chunk, verdicts in zip(chunks, chunk_verdicts):
        if verdicts is None:
            failed_chunks += 1
            continue
        for i, (page_info, block, payload) in enumerate(chunk):
            verdict = verdicts[i]
            if verdict.verdict == "prune":
                pruned_refs.append((page_info, block))
            elif verdict.verdict == "fix":
                fixed = _validate_fixed_latex(payload[0], verdict.fixed_latex)
                if fixed is not None:
                    _interline_equation_span(block)["content"] = fixed
                    fixed_count += 1

    _remove_pruned_blocks(pruned_refs)
    logger.info(
        f"LLM formula prune: {len(candidates)} candidates, "
        f"{len(pruned_refs)} pruned, {fixed_count} fixed, "
        f"{failed_chunks}/{len(chunks)} chunks failed."
    )


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
