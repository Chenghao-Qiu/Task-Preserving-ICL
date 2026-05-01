import json
import os
import re
from typing import Callable, Sequence

import numpy as np
import torch

from utils.dataprocess import TestDataset

INCLUDE_QUERY_UNIT: bool = False
ATTENTION_EXCLUDED_TOKENS: list[str] = [
    "'",
    ",",
    ".",
    "<0x0A>"
]
ATTENTION_FILTER_PROMPT_PHRASES: bool = True
ATTENTION_EXCLUDED_PHRASES: list[str] = [
    "sentence:",
    "The answer is",
]

def _clean_token_label(token: str) -> str:
    return token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n")


def _get_token_offsets(
    *,
    prompt: str,
    tokenizer,
    max_model_len: int,
    token_count: int,
) -> list[tuple[int, int]]:
    try:
        offset_inputs = tokenizer(
            prompt,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_model_len,
            padding=False,
        )
        return offset_inputs['offset_mapping'][:token_count]
    except (NotImplementedError, KeyError, TypeError, ValueError):
        return []


def _build_tick_positions(num_tokens: int) -> list[int]:
    if num_tokens <= 0:
        return []
    tick_step = max(1, num_tokens // 40)
    tick_positions = list(range(0, num_tokens, tick_step))
    if tick_positions[-1] != num_tokens - 1:
        tick_positions.append(num_tokens - 1)
    return tick_positions


def _build_word_units_from_offsets(
    *,
    prompt: str,
    tokens: Sequence[str],
    offsets: Sequence[tuple[int, int]],
) -> tuple[list[str], list[list[int]]]:
    word_spans = [(match.start(), match.end(), match.group(0)) for match in re.finditer(r"\S+", prompt)]
    if not word_spans:
        return list(tokens), [[i] for i in range(len(tokens))]

    unit_positions: list[list[int]] = []
    word_ptr = 0
    last_word_index = -1

    for token_idx, (token, (start, end)) in enumerate(zip(tokens, offsets)):
        if end <= start:
            unit_positions.append([token_idx])
            last_word_index = -1
            continue

        while word_ptr < len(word_spans) and word_spans[word_ptr][1] <= start:
            word_ptr += 1

        if word_ptr < len(word_spans):
            span_start, span_end, _ = word_spans[word_ptr]
            if start < span_end and end > span_start:
                if word_ptr == last_word_index:
                    unit_positions[-1].append(token_idx)
                else:
                    unit_positions.append([token_idx])
                    last_word_index = word_ptr
                continue

        unit_positions.append([token_idx])
        last_word_index = -1

    unit_labels: list[str] = []
    for positions in unit_positions:
        if not positions:
            unit_labels.append('')
            continue
        unit_label = ''.join(
            prompt[offsets[pos][0]:offsets[pos][1]]
            for pos in positions
            if offsets[pos][1] > offsets[pos][0]
        ).strip()
        if not unit_label:
            unit_label = ''.join(_clean_token_label(tokens[pos]) for pos in positions).strip() or tokens[positions[0]]
        unit_labels.append(unit_label)

    return unit_labels, unit_positions


def _build_unit_offsets_from_positions(
    *,
    token_offsets: Sequence[tuple[int, int]],
    unit_positions: Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    unit_offsets: list[tuple[int, int]] = []
    for positions in unit_positions:
        if not positions:
            unit_offsets.append((0, 0))
            continue
        starts = [token_offsets[pos][0] for pos in positions]
        ends = [token_offsets[pos][1] for pos in positions]
        unit_offsets.append((min(starts), max(ends)))
    return unit_offsets


def _build_word_units_fallback(
    *,
    tokenizer,
    tokens: Sequence[str],
) -> tuple[list[str], list[list[int]]]:
    unit_labels: list[str] = []
    unit_positions: list[list[int]] = []
    special_tokens = {
        token for token in getattr(tokenizer, "all_special_tokens", [])
        if isinstance(token, str) and token
    }

    for token_idx, token in enumerate(tokens):
        is_new_word = (
            token_idx == 0
            or token.startswith("Ġ")
            or token.startswith("▁")
            or token in special_tokens
            or _clean_token_label(token).strip() == ''
        )
        if is_new_word:
            unit_positions.append([token_idx])
        else:
            unit_positions[-1].append(token_idx)

    for positions in unit_positions:
        merged = tokenizer.convert_tokens_to_string([tokens[pos] for pos in positions]).strip()
        if not merged:
            merged = ''.join(_clean_token_label(tokens[pos]) for pos in positions).strip() or tokens[positions[0]]
        unit_labels.append(merged)

    return unit_labels, unit_positions


def _build_word_attention_units(
    *,
    prompt: str,
    tokenizer,
    max_model_len: int,
    tokens: Sequence[str],
    layer_head_maps: Sequence[torch.Tensor],
    offsets: Sequence[tuple[int, int]] = (),
    exclude_tokens: bool = False,
) -> tuple[list[str], list[torch.Tensor], list[tuple[int, int]], list[int]]:
    if offsets and len(offsets) == len(tokens):
        token_offsets = list(offsets)
    else:
        token_offsets = _get_token_offsets(
            prompt=prompt,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            token_count=len(tokens),
        )

    excluded_positions = (
        _find_excluded_token_positions(
            prompt=prompt,
            tokens=tokens,
            offsets=token_offsets,
        )
        if exclude_tokens and token_offsets
        else []
    )
    excluded_position_set = set(excluded_positions)
    kept_positions = [
        idx for idx in range(len(tokens))
        if idx not in excluded_position_set
    ]
    if exclude_tokens and not kept_positions:
        raise ValueError("All word-mode attention tokens were excluded; please keep at least one token.")

    working_tokens = [tokens[idx] for idx in kept_positions] if exclude_tokens else list(tokens)
    working_offsets = [token_offsets[idx] for idx in kept_positions] if exclude_tokens and token_offsets else token_offsets
    working_head_maps = []
    for head_maps in layer_head_maps:
        if exclude_tokens:
            sliced = head_maps[:, kept_positions]
            renorm_denom = sliced.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            working_head_maps.append(sliced / renorm_denom)
        else:
            working_head_maps.append(head_maps)

    try:
        if working_offsets and len(working_offsets) == len(working_tokens):
            unit_labels, unit_positions = _build_word_units_from_offsets(
                prompt=prompt,
                tokens=working_tokens,
                offsets=working_offsets,
            )
            unit_offsets = _build_unit_offsets_from_positions(
                token_offsets=working_offsets,
                unit_positions=unit_positions,
            )
        else:
            raise ValueError
    except (NotImplementedError, KeyError, TypeError, ValueError):
        unit_labels, unit_positions = _build_word_units_fallback(
            tokenizer=tokenizer,
            tokens=working_tokens,
        )
        unit_offsets = []

    layer_unit_maps = []
    for head_maps in working_head_maps:
        unit_map = torch.stack(
            [head_maps[:, positions].sum(dim=-1) for positions in unit_positions],
            dim=-1,
        )
        layer_unit_maps.append(unit_map)

    cleaned_unit_labels = [
        label if label else working_tokens[positions[0]]
        for label, positions in zip(unit_labels, unit_positions)
    ]
    return cleaned_unit_labels, layer_unit_maps, unit_offsets, excluded_positions


def _build_attention_units(
    *,
    attention_map_mode: str,
    prompt: str,
    tokenizer,
    max_model_len: int,
    tokens: Sequence[str],
    layer_head_maps: Sequence[torch.Tensor],
    offsets: Sequence[tuple[int, int]] = (),
    exemplar_metadata: Sequence[dict] = (),
) -> tuple[list[str], list[torch.Tensor], str, list[tuple[int, int]]]:
    if attention_map_mode == 'token':
        token_offsets = list(offsets) if offsets and len(offsets) == len(tokens) else []
        return list(tokens), list(layer_head_maps), 'tokens', token_offsets
    if attention_map_mode == 'unit':
        unit_labels, layer_unit_maps, _, _, _ = _build_exemplar_attention_units(
            prompt=prompt,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            tokens=tokens,
            offsets=offsets,
            layer_head_maps=layer_head_maps,
            exemplar_metadata=exemplar_metadata,
        )
        return unit_labels, layer_unit_maps, 'exemplars', []

    unit_labels, layer_unit_maps, unit_offsets, _ = _build_word_attention_units(
        prompt=prompt,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        tokens=tokens,
        layer_head_maps=layer_head_maps,
        offsets=offsets,
        exclude_tokens=False,
    )
    return unit_labels, layer_unit_maps, 'words', unit_offsets


def _find_focus_char_span(prompt: str, focus_text: str) -> tuple[int, int] | None:
    if not focus_text:
        return None

    start = prompt.rfind(focus_text)
    if start >= 0:
        return start, start + len(focus_text)

    stripped_text = focus_text.strip()
    if stripped_text and stripped_text != focus_text:
        start = prompt.rfind(stripped_text)
        if start >= 0:
            return start, start + len(stripped_text)

    return None


def _find_phrase_char_spans(text: str, phrases: Sequence[str]) -> list[tuple[int, int]]:
    if not ATTENTION_FILTER_PROMPT_PHRASES:
        return []

    lowered_text = text.lower()
    spans: list[tuple[int, int]] = []
    for phrase in phrases:
        if not phrase:
            continue
        lowered_phrase = phrase.lower()
        search_start = 0
        while True:
            match_start = lowered_text.find(lowered_phrase, search_start)
            if match_start < 0:
                break
            spans.append((match_start, match_start + len(phrase)))
            search_start = match_start + len(lowered_phrase)
    return spans


def _find_excluded_token_positions(
    *,
    prompt: str,
    tokens: Sequence[str],
    offsets: Sequence[tuple[int, int]],
) -> list[int]:
    excluded_positions = {
        idx for idx, token in enumerate(tokens)
        if _is_excluded_attention_token(token)
    }
    if ATTENTION_FILTER_PROMPT_PHRASES and offsets:
        excluded_phrase_spans = _find_phrase_char_spans(prompt, ATTENTION_EXCLUDED_PHRASES)
        for token_idx, (start, end) in enumerate(offsets):
            if any(end > span_start and start < span_end for span_start, span_end in excluded_phrase_spans):
                excluded_positions.add(token_idx)
    return sorted(excluded_positions)


def _find_top_fraction_positions(
    *,
    scores: np.ndarray,
    positions: Sequence[int],
    fraction: float = 1 / 3,
) -> list[int]:
    if not positions:
        return []
    top_count = max(1, int(np.ceil(len(positions) * fraction)))
    ranked_positions = sorted(positions, key=lambda idx: float(scores[idx]), reverse=True)
    return ranked_positions[:top_count]


def _build_sentence_highlight_groups(
    *,
    prompt: str,
    unit_offsets: Sequence[tuple[int, int]],
    sentence_texts: Sequence[str],
) -> list[list[int]]:
    if not prompt or not unit_offsets or not sentence_texts:
        return []

    try:
        sentence_spans = _find_sequential_char_spans(prompt, sentence_texts)
    except ValueError:
        return []

    highlight_groups: list[list[int]] = []
    for span_start, span_end in sentence_spans:
        positions = [
            unit_idx
            for unit_idx, (start, end) in enumerate(unit_offsets)
            if end > span_start and start < span_end
        ]
        if positions:
            highlight_groups.append(positions)
    return highlight_groups


def _build_group_highlight_indices(
    *,
    scores: np.ndarray,
    groups: Sequence[Sequence[int]],
    fraction: float = 1 / 3,
) -> list[int]:
    highlighted_positions: set[int] = set()
    for positions in groups:
        highlighted_positions.update(
            _find_top_fraction_positions(
                scores=scores,
                positions=positions,
                fraction=fraction,
            )
        )
    return sorted(highlighted_positions)


def _build_last_layer_highlight_indices(
    *,
    attention_matrix: np.ndarray,
    attention_map_mode: str,
    prompt: str = '',
    unit_offsets: Sequence[tuple[int, int]] = (),
    sentence_texts: Sequence[str] = (),
) -> list[int]:
    last_layer_scores = attention_matrix[-1]
    if attention_map_mode == 'unit':
        return _find_top_fraction_positions(
            scores=last_layer_scores,
            positions=list(range(len(last_layer_scores))),
        )

    sentence_groups = _build_sentence_highlight_groups(
        prompt=prompt,
        unit_offsets=unit_offsets,
        sentence_texts=sentence_texts,
    )
    if not sentence_groups:
        return _find_top_fraction_positions(
            scores=last_layer_scores,
            positions=list(range(len(last_layer_scores))),
        )

    return _build_group_highlight_indices(
        scores=last_layer_scores,
        groups=sentence_groups,
    )


def _build_word_mode_highlight_indices(
    *,
    attention_matrix: np.ndarray,
    prompt: str,
    unit_offsets: Sequence[tuple[int, int]],
    exemplar_texts: Sequence[str],
) -> tuple[list[int], list[int]]:
    last_layer_scores = attention_matrix[-1]
    exemplar_groups = _build_sentence_highlight_groups(
        prompt=prompt,
        unit_offsets=unit_offsets,
        sentence_texts=exemplar_texts,
    )
    exemplar_local_indices = _build_group_highlight_indices(
        scores=last_layer_scores,
        groups=exemplar_groups,
        fraction=1 / 3,
    ) if exemplar_groups else []
    global_top_indices = _find_top_fraction_positions(
        scores=last_layer_scores,
        positions=list(range(len(last_layer_scores))),
        fraction=0.25,
    )
    return exemplar_local_indices, global_top_indices


def _slice_attention_to_char_span(
    *,
    prompt: str,
    tokenizer,
    max_model_len: int,
    tokens: Sequence[str],
    layer_head_maps: Sequence[torch.Tensor],
    char_span: tuple[int, int],
) -> tuple[str, list[str], list[tuple[int, int]], list[torch.Tensor]] | None:
    span_start, span_end = char_span
    if span_end <= span_start:
        return None

    offsets = _get_token_offsets(
        prompt=prompt,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        token_count=len(tokens),
    )

    if not offsets:
        return None

    selected_positions = [
        token_idx
        for token_idx, (start, end) in enumerate(offsets)
        if end > span_start and start < span_end
    ]
    if not selected_positions:
        return None

    span_tokens = [tokens[pos] for pos in selected_positions]
    span_offsets = [
        (
            max(offsets[pos][0], span_start) - span_start,
            min(offsets[pos][1], span_end) - span_start,
        )
        for pos in selected_positions
    ]
    span_layer_head_maps = []
    for head_maps in layer_head_maps:
        sliced = head_maps[:, selected_positions]
        renorm_denom = sliced.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        span_layer_head_maps.append(sliced / renorm_denom)

    return prompt[span_start:span_end], span_tokens, span_offsets, span_layer_head_maps


def _filter_excluded_attention_units(
    *,
    units: Sequence[str],
    unit_maps: Sequence[torch.Tensor],
    prompt: str = '',
    offsets: Sequence[tuple[int, int]] = (),
) -> tuple[list[str], list[torch.Tensor], list[int]]:
    has_token_exclusions = any(token for token in ATTENTION_EXCLUDED_TOKENS)
    has_phrase_exclusions = ATTENTION_FILTER_PROMPT_PHRASES and any(phrase for phrase in ATTENTION_EXCLUDED_PHRASES)
    if not has_token_exclusions and not has_phrase_exclusions:
        return list(units), list(unit_maps), []

    excluded_tokens = {token for token in ATTENTION_EXCLUDED_TOKENS if token}
    excluded_positions = {
        idx for idx, unit in enumerate(units)
        if unit in excluded_tokens or _clean_token_label(unit) in excluded_tokens
    }
    if prompt and offsets and len(offsets) == len(units):
        excluded_positions.update(
            _find_excluded_token_positions(
                prompt=prompt,
                tokens=units,
                offsets=offsets,
            )
        )
    excluded_positions = sorted(excluded_positions)
    if not excluded_positions:
        return list(units), list(unit_maps), []

    kept_positions = [idx for idx in range(len(units)) if idx not in set(excluded_positions)]
    if not kept_positions:
        raise ValueError("All attention tokens were excluded; please keep at least one token.")

    filtered_units = [units[idx] for idx in kept_positions]
    filtered_unit_maps = []
    for unit_map in unit_maps:
        sliced = unit_map[:, kept_positions]
        renorm_denom = sliced.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        filtered_unit_maps.append(sliced / renorm_denom)

    return filtered_units, filtered_unit_maps, excluded_positions


def _is_excluded_attention_token(token: str) -> bool:
    excluded_tokens = {item for item in ATTENTION_EXCLUDED_TOKENS if item}
    return token in excluded_tokens or _clean_token_label(token) in excluded_tokens


def _find_sequential_char_spans(text: str, parts: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for part in parts:
        if not part:
            continue
        start = text.find(part, cursor)
        if start < 0:
            start = text.find(part)
        if start < 0:
            raise ValueError("Exemplar attention export failed because an exemplar could not be located in the focus span.")
        end = start + len(part)
        spans.append((start, end))
        cursor = end
    return spans


def _build_exemplar_attention_units(
    *,
    prompt: str,
    tokenizer,
    max_model_len: int,
    tokens: Sequence[str],
    layer_head_maps: Sequence[torch.Tensor],
    offsets: Sequence[tuple[int, int]] = (),
    exemplar_metadata: Sequence[dict],
    exclude_tokens: bool = False,
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor], list[int], list[int]]:
    if not exemplar_metadata:
        raise ValueError("Exemplar attention export failed because no exemplar metadata was provided.")

    if offsets and len(offsets) == len(tokens):
        token_offsets = list(offsets)
    else:
        token_offsets = _get_token_offsets(
            prompt=prompt,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            token_count=len(tokens),
        )

    if not token_offsets:
        raise ValueError("Exemplar attention export failed because prompt offsets are unavailable.")

    excluded_token_positions = set(
        _find_excluded_token_positions(
            prompt=prompt,
            tokens=tokens,
            offsets=token_offsets,
        )
    ) if exclude_tokens else set()

    exemplar_texts = [str(item.get('text', '')) for item in exemplar_metadata]
    exemplar_sources = [str(item.get('source', 'orig')) for item in exemplar_metadata]
    exemplar_spans = _find_sequential_char_spans(prompt, exemplar_texts)
    all_unit_spans = list(exemplar_spans)
    unit_labels = ['A' if source == 'adv' else 'O' for source in exemplar_sources]
    if INCLUDE_QUERY_UNIT:
        query_start = exemplar_spans[-1][1] if exemplar_spans else 0
        query_end = len(prompt)
        if query_end <= query_start:
            raise ValueError("Exemplar attention export failed because the query span is empty.")
        all_unit_spans.append((query_start, query_end))
        unit_labels.append('Q')

    unit_positions: list[list[int]] = []
    unit_token_counts: list[int] = []
    filtered_token_counts: list[int] = []
    for span_start, span_end in all_unit_spans:
        positions = [
            token_idx
            for token_idx, (start, end) in enumerate(token_offsets)
            if end > span_start and start < span_end
        ]
        if not positions:
            raise ValueError("Exemplar attention export failed because an exemplar span could not be aligned to tokens.")
        if exclude_tokens:
            original_count = len(positions)
            positions = [pos for pos in positions if pos not in excluded_token_positions]
            filtered_token_counts.append(original_count - len(positions))
        else:
            filtered_token_counts.append(0)
        unit_positions.append(positions)
        unit_token_counts.append(len(positions))

    layer_unit_maps = []
    layer_unit_avg_maps = []
    for head_maps in layer_head_maps:
        aggregated_units = []
        aggregated_unit_avgs = []
        for positions, token_count in zip(unit_positions, unit_token_counts):
            if positions:
                aggregated_sum = head_maps[:, positions].sum(dim=-1)
                aggregated_units.append(aggregated_sum)
                aggregated_unit_avgs.append(aggregated_sum / token_count)
            else:
                zeros = torch.zeros(head_maps.shape[0], dtype=head_maps.dtype, device=head_maps.device)
                aggregated_units.append(zeros)
                aggregated_unit_avgs.append(zeros)
        unit_map = torch.stack(aggregated_units, dim=-1)
        unit_avg_map = torch.stack(aggregated_unit_avgs, dim=-1)
        renorm_denom = unit_map.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        renorm_avg_denom = unit_avg_map.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        layer_unit_maps.append(unit_map / renorm_denom)
        layer_unit_avg_maps.append(unit_avg_map / renorm_avg_denom)

    return unit_labels, layer_unit_maps, layer_unit_avg_maps, unit_token_counts, filtered_token_counts


def _plot_attention_matrix(
    *,
    attention_matrix: np.ndarray,
    tokens: Sequence[str],
    unit_axis_label: str,
    title: str,
    figure_path: str,
    highlighted_indices: Sequence[int] = (),
    secondary_highlighted_indices: Sequence[int] = (),
) -> None:
    import matplotlib.pyplot as plt

    num_layers = attention_matrix.shape[0]
    num_tokens = len(tokens)
    display_matrix = attention_matrix.T
    last_layer_scores = attention_matrix[-1]

    main_fig_width = max(8, min(18, num_layers * 0.28))
    stats_col_width = 2.2
    fig_width = main_fig_width + stats_col_width + 0.8
    fig_height = max(10, num_tokens * 0.28)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[main_fig_width, stats_col_width, 0.45],
        wspace=0.06,
    )
    ax = fig.add_subplot(gs[0, 0])
    last_layer_ax = fig.add_subplot(gs[0, 1], sharey=ax)
    cax = fig.add_subplot(gs[0, 2])

    image = ax.imshow(display_matrix, aspect='auto', cmap='viridis', origin='upper')
    ax.set_title(title, pad=8)
    ax.set_xlabel("Layers")
    ax.set_ylabel(f"Prompt {unit_axis_label}")

    layer_tick_positions = _build_tick_positions(num_layers)
    ax.set_xticks(layer_tick_positions)
    ax.set_xticklabels([str(i) for i in layer_tick_positions], fontsize=8)

    token_tick_positions = list(range(num_tokens))
    token_tick_labels = [_clean_token_label(tokens[pos]) for pos in token_tick_positions]
    ax.set_yticks(token_tick_positions)
    ax.set_yticklabels(token_tick_labels, fontsize=7)

    last_layer_ax.set_title("Last layer", fontsize=9, pad=8)
    last_layer_ax.set_xlim(0, 1)
    last_layer_ax.set_xticks([])
    last_layer_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    for spine in last_layer_ax.spines.values():
        spine.set_visible(False)
    last_layer_ax.set_facecolor('#f7f7f7')
    primary_highlight_indices = set(highlighted_indices)
    secondary_highlight_indices = set(secondary_highlighted_indices)
    for row_idx, value in enumerate(last_layer_scores):
        if row_idx in primary_highlight_indices and row_idx in secondary_highlight_indices:
            highlight_color = '#d9d2f3'
        elif row_idx in primary_highlight_indices:
            highlight_color = '#f9d6a5'
        elif row_idx in secondary_highlight_indices:
            highlight_color = '#cfe2f3'
        else:
            highlight_color = None
        if highlight_color is not None:
            last_layer_ax.axhspan(row_idx - 0.5, row_idx + 0.5, color=highlight_color, zorder=0)
        last_layer_ax.text(
            0.5,
            row_idx,
            f"{value:.4f}",
            ha='center',
            va='center',
            fontsize=7,
        )

    fig.colorbar(image, cax=cax, label="Attention weight")
    fig.tight_layout()

    fig.savefig(figure_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def _build_plot_title(
    *,
    base_title: str,
    units: Sequence[str],
    attention_matrix: np.ndarray,
) -> str:
    if 'A' not in units:
        return base_title

    last_layer_scores = attention_matrix[-1]
    a_attention_ratio = sum(
        float(score) for unit, score in zip(units, last_layer_scores)
        if unit == 'A'
    )
    return f"{base_title}, last-layer A attention={a_attention_ratio:.4f}"


def _export_attention_matrix_raw_data(
    *,
    attention_matrix: np.ndarray,
    tokens: Sequence[str],
    unit_axis_label: str,
    title: str,
    output_path_prefix: str,
    highlighted_indices: Sequence[int] = (),
    secondary_highlighted_indices: Sequence[int] = (),
    metadata: dict | None = None,
) -> None:
    clean_tokens = [_clean_token_label(token) for token in tokens]
    last_layer_scores = attention_matrix[-1] if attention_matrix.shape[0] else np.array([])
    metadata_payload = {
        'title': title,
        'unit_axis_label': unit_axis_label,
        'num_layers': int(attention_matrix.shape[0]),
        'num_units': int(attention_matrix.shape[1]) if attention_matrix.ndim >= 2 else 0,
        'tokens': list(tokens),
        'clean_tokens': clean_tokens,
        'highlighted_indices': [int(idx) for idx in highlighted_indices],
        'secondary_highlighted_indices': [int(idx) for idx in secondary_highlighted_indices],
    }
    if metadata:
        metadata_payload.update(metadata)

    np.savez_compressed(
        f"{output_path_prefix}.npz",
        attention_matrix=attention_matrix,
        tokens=np.array(list(tokens), dtype=np.str_),
        clean_tokens=np.array(clean_tokens, dtype=np.str_),
        highlighted_indices=np.array(metadata_payload['highlighted_indices'], dtype=np.int64),
        secondary_highlighted_indices=np.array(metadata_payload['secondary_highlighted_indices'], dtype=np.int64),
        last_layer_scores=np.asarray(last_layer_scores),
    )
    with open(f"{output_path_prefix}.json", 'w', encoding='utf-8') as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)


def _save_attention_matrix_outputs(
    *,
    attention_matrix: np.ndarray,
    tokens: Sequence[str],
    unit_axis_label: str,
    title: str,
    figure_path: str,
    highlighted_indices: Sequence[int] = (),
    secondary_highlighted_indices: Sequence[int] = (),
    publication_mode: bool = False,
    metadata: dict | None = None,
) -> None:
    _plot_attention_matrix(
        attention_matrix=attention_matrix,
        tokens=tokens,
        unit_axis_label=unit_axis_label,
        title=title,
        figure_path=figure_path,
        highlighted_indices=highlighted_indices,
        secondary_highlighted_indices=secondary_highlighted_indices,
    )
    if publication_mode:
        output_path_prefix, _ = os.path.splitext(figure_path)
        _export_attention_matrix_raw_data(
            attention_matrix=attention_matrix,
            tokens=tokens,
            unit_axis_label=unit_axis_label,
            title=title,
            output_path_prefix=output_path_prefix,
            highlighted_indices=highlighted_indices,
            secondary_highlighted_indices=secondary_highlighted_indices,
            metadata=metadata,
        )


def export_attention_map_for_run(
    *,
    backend: str,
    is_qa_task: bool,
    test_csv: str,
    output_dir: str,
    run_idx: int,
    test_example_idx: int,
    tokenizer,
    model,
    input_device,
    max_model_len: int,
    label_token_ids: Sequence[int],
    labels: Sequence[str],
    create_prompt: Callable[[str], str],
    create_attention_focus_content: Callable[[str], str],
    attention_map_mode: str = 'token',
    exemplar_metadata: Sequence[dict] = (),
    publication_mode: bool = False,
) -> None:
    """Export attention heatmaps for a single run."""

    if backend != 'transformers':
        raise ValueError("Attention map export is only supported with --backend transformers.")
    if is_qa_task:
        raise ValueError("Attention map export is only implemented for classification/NLI tasks.")

    os.makedirs(output_dir, exist_ok=True)
    dataset = TestDataset(test_csv)
    if test_example_idx >= len(dataset):
        raise ValueError(
            f"Requested test_example_idx={test_example_idx}, but dataset only has {len(dataset)} rows."
        )

    sample = dataset[test_example_idx]
    attention_focus_text = create_attention_focus_content(sample['text'])
    prompt = create_prompt(sample['text'])
    exemplar_texts = [str(item.get('text', '')) for item in exemplar_metadata if str(item.get('text', ''))]
    sentence_texts = list(exemplar_texts)
    if sample['text']:
        sentence_texts.append(str(sample['text']))
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=max_model_len,
        padding=False,
    )
    input_ids = inputs['input_ids'].to(input_device)
    attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(input_device)

    with torch.no_grad():
        prompt_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=True,
        )

    next_token_logits = prompt_outputs.logits[:, -1, :]
    candidate_logits = next_token_logits[:, list(label_token_ids)]
    predicted_label_idx = int(torch.argmax(candidate_logits, dim=-1).item())
    predicted_label = labels[predicted_label_idx]
    if not prompt_outputs.attentions:
        raise ValueError("Model did not return attention tensors for the prompt.")

    valid_token_count = int(attention_mask[0].sum().item())
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, :valid_token_count].tolist())
    last_token_idx = valid_token_count - 1
    layer_head_maps = [
        layer_attention[0, :, last_token_idx, :valid_token_count].float()
        for layer_attention in prompt_outputs.attentions
    ]
    focus_char_span = _find_focus_char_span(prompt, attention_focus_text)
    if focus_char_span is None:
        raise ValueError("Attention map export failed because the exemplar+query span was not found in the prompt.")

    focus_slice = _slice_attention_to_char_span(
        prompt=prompt,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        tokens=tokens,
        layer_head_maps=layer_head_maps,
        char_span=focus_char_span,
    )
    if focus_slice is None:
        raise ValueError("Attention map export failed because the exemplar+query span could not be aligned to prompt offsets.")

    focus_prompt, focus_tokens, focus_offsets, focus_head_maps = focus_slice
    focus_units, focus_unit_maps, focus_unit_axis_label, focus_unit_offsets = _build_attention_units(
        attention_map_mode=attention_map_mode,
        prompt=focus_prompt,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        tokens=focus_tokens,
        offsets=focus_offsets,
        layer_head_maps=focus_head_maps,
        exemplar_metadata=exemplar_metadata,
    )
    excluded_positions: list[int] = []
    excluded_preview: list[tuple[int, str]] = []

    print(f"  Attention map mode: {attention_map_mode}")
    print(f"  Focus span (exemplar+query) char span: {focus_char_span}")
    if attention_map_mode == 'unit':
        _, _, exemplar_raw_unit_avg_maps, exemplar_token_counts, _ = _build_exemplar_attention_units(
            prompt=focus_prompt,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            tokens=focus_tokens,
            offsets=focus_offsets,
            layer_head_maps=focus_head_maps,
            exemplar_metadata=exemplar_metadata,
            exclude_tokens=False,
        )
        exemplar_filtered_units, exemplar_filtered_unit_maps, exemplar_filtered_unit_avg_maps, remaining_exemplar_token_counts, filtered_token_counts = _build_exemplar_attention_units(
            prompt=focus_prompt,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            tokens=focus_tokens,
            offsets=focus_offsets,
            layer_head_maps=focus_head_maps,
            exemplar_metadata=exemplar_metadata,
            exclude_tokens=True,
        )
        print(
            f"  Exemplar units: {len(focus_units)} "
            f"(O={sum(unit == 'O' for unit in focus_units)}, A={sum(unit == 'A' for unit in focus_units)})"
        )
        print(f"  Exemplar token count: {sum(exemplar_token_counts)}")
        print(f"  Filtered-out token count: {sum(filtered_token_counts)}")
        print(f"  Remaining exemplar token count: {sum(remaining_exemplar_token_counts)}")
    elif attention_map_mode == 'token':
        token_raw_units = list(focus_units)
        token_raw_unit_maps = list(focus_unit_maps)
        token_filtered_units, token_filtered_unit_maps, excluded_positions = _filter_excluded_attention_units(
            units=focus_units,
            unit_maps=focus_unit_maps,
            prompt=focus_prompt,
            offsets=focus_unit_offsets,
        )
        token_raw_unit_offsets = list(focus_unit_offsets)
        excluded_position_set = set(excluded_positions)
        token_filtered_unit_offsets = [
            offset for idx, offset in enumerate(token_raw_unit_offsets)
            if idx not in excluded_position_set
        ]
        token_raw_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in token_raw_unit_maps],
            axis=0,
        )
        token_filtered_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in token_filtered_unit_maps],
            axis=0,
        )
        excluded_preview = [(idx, token_raw_units[idx]) for idx in excluded_positions[:8]]
    else:
        token_raw_units = list(focus_units)
        token_raw_unit_maps = list(focus_unit_maps)
        token_raw_unit_offsets = list(focus_unit_offsets)
        token_filtered_units, token_filtered_unit_maps, token_filtered_unit_offsets, excluded_positions = _build_word_attention_units(
            prompt=focus_prompt,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            tokens=focus_tokens,
            layer_head_maps=focus_head_maps,
            offsets=focus_offsets,
            exclude_tokens=True,
        )
        token_raw_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in token_raw_unit_maps],
            axis=0,
        )
        token_filtered_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in token_filtered_unit_maps],
            axis=0,
        )
        excluded_preview = [(idx, focus_tokens[idx]) for idx in excluded_positions[:8]]
    if excluded_positions:
        suffix = '...' if len(excluded_positions) > 8 else ''
        print(f"  Excluded token positions ({len(excluded_positions)}): {excluded_preview}{suffix}")

    print(
        f"\nExporting attention maps for run {run_idx + 1} "
        f"(test example {test_example_idx + 1}, pred={predicted_label}, gold={sample['label']}) "
        f"to {output_dir}"
    )

    base_title = f"gold={sample['label']}, pred={predicted_label}, mode={attention_map_mode}"
    mode_suffix = '' if attention_map_mode == 'token' else f"_{attention_map_mode}"
    raw_output_metadata = {
        'run_idx': int(run_idx),
        'run_number': int(run_idx + 1),
        'test_example_idx': int(test_example_idx),
        'test_example_number': int(test_example_idx + 1),
        'gold_label': sample['label'],
        'predicted_label': predicted_label,
        'attention_map_mode': attention_map_mode,
    }
    if attention_map_mode == 'unit':
        exemplar_raw_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in focus_unit_maps],
            axis=0,
        )
        exemplar_raw_avg_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in exemplar_raw_unit_avg_maps],
            axis=0,
        )
        exemplar_filtered_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in exemplar_filtered_unit_maps],
            axis=0,
        )
        exemplar_filtered_avg_attention_matrix = np.stack(
            [head_maps.mean(dim=0).detach().cpu().numpy() for head_maps in exemplar_filtered_unit_avg_maps],
            axis=0,
        )
        exemplar_raw_highlight_indices = _build_last_layer_highlight_indices(
            attention_matrix=exemplar_raw_attention_matrix,
            attention_map_mode=attention_map_mode,
        )
        exemplar_raw_avg_highlight_indices = _build_last_layer_highlight_indices(
            attention_matrix=exemplar_raw_avg_attention_matrix,
            attention_map_mode=attention_map_mode,
        )
        exemplar_filtered_highlight_indices = _build_last_layer_highlight_indices(
            attention_matrix=exemplar_filtered_attention_matrix,
            attention_map_mode=attention_map_mode,
        )
        exemplar_filtered_avg_highlight_indices = _build_last_layer_highlight_indices(
            attention_matrix=exemplar_filtered_avg_attention_matrix,
            attention_map_mode=attention_map_mode,
        )
        _save_attention_matrix_outputs(
            attention_matrix=exemplar_raw_attention_matrix,
            tokens=focus_units,
            unit_axis_label=focus_unit_axis_label,
            title=_build_plot_title(
                base_title=base_title,
                units=focus_units,
                attention_matrix=exemplar_raw_attention_matrix,
            ),
            figure_path=os.path.join(output_dir, f"run_{run_idx + 1}{mode_suffix}_exemplar_raw.png"),
            highlighted_indices=exemplar_raw_highlight_indices,
            publication_mode=publication_mode,
            metadata={**raw_output_metadata, 'view': 'exemplar_raw'},
        )
        _save_attention_matrix_outputs(
            attention_matrix=exemplar_raw_avg_attention_matrix,
            tokens=focus_units,
            unit_axis_label=focus_unit_axis_label,
            title=_build_plot_title(
                base_title=base_title,
                units=focus_units,
                attention_matrix=exemplar_raw_avg_attention_matrix,
            ),
            figure_path=os.path.join(output_dir, f"run_{run_idx + 1}{mode_suffix}_exemplar_raw_avg.png"),
            highlighted_indices=exemplar_raw_avg_highlight_indices,
            publication_mode=publication_mode,
            metadata={**raw_output_metadata, 'view': 'exemplar_raw_avg'},
        )
        _save_attention_matrix_outputs(
            attention_matrix=exemplar_filtered_attention_matrix,
            tokens=exemplar_filtered_units,
            unit_axis_label=focus_unit_axis_label,
            title=_build_plot_title(
                base_title=base_title,
                units=exemplar_filtered_units,
                attention_matrix=exemplar_filtered_attention_matrix,
            ),
            figure_path=os.path.join(output_dir, f"run_{run_idx + 1}{mode_suffix}_exemplar_filtered.png"),
            highlighted_indices=exemplar_filtered_highlight_indices,
            publication_mode=publication_mode,
            metadata={**raw_output_metadata, 'view': 'exemplar_filtered'},
        )
        _save_attention_matrix_outputs(
            attention_matrix=exemplar_filtered_avg_attention_matrix,
            tokens=exemplar_filtered_units,
            unit_axis_label=focus_unit_axis_label,
            title=_build_plot_title(
                base_title=base_title,
                units=exemplar_filtered_units,
                attention_matrix=exemplar_filtered_avg_attention_matrix,
            ),
            figure_path=os.path.join(output_dir, f"run_{run_idx + 1}{mode_suffix}_exemplar_filtered_avg.png"),
            highlighted_indices=exemplar_filtered_avg_highlight_indices,
            publication_mode=publication_mode,
            metadata={**raw_output_metadata, 'view': 'exemplar_filtered_avg'},
        )
    else:
        token_raw_secondary_highlight_indices: list[int] = []
        token_filtered_secondary_highlight_indices: list[int] = []
        if attention_map_mode == 'word':
            token_raw_highlight_indices, token_raw_secondary_highlight_indices = _build_word_mode_highlight_indices(
                attention_matrix=token_raw_attention_matrix,
                prompt=focus_prompt,
                unit_offsets=token_raw_unit_offsets,
                exemplar_texts=exemplar_texts,
            )
            token_filtered_highlight_indices, token_filtered_secondary_highlight_indices = _build_word_mode_highlight_indices(
                attention_matrix=token_filtered_attention_matrix,
                prompt=focus_prompt,
                unit_offsets=token_filtered_unit_offsets,
                exemplar_texts=exemplar_texts,
            )
        else:
            token_raw_highlight_indices = _build_last_layer_highlight_indices(
                attention_matrix=token_raw_attention_matrix,
                attention_map_mode=attention_map_mode,
                prompt=focus_prompt,
                unit_offsets=token_raw_unit_offsets,
                sentence_texts=sentence_texts,
            )
            token_filtered_highlight_indices = _build_last_layer_highlight_indices(
                attention_matrix=token_filtered_attention_matrix,
                attention_map_mode=attention_map_mode,
                prompt=focus_prompt,
                unit_offsets=token_filtered_unit_offsets,
                sentence_texts=sentence_texts,
            )
        _save_attention_matrix_outputs(
            attention_matrix=token_raw_attention_matrix,
            tokens=token_raw_units,
            unit_axis_label=focus_unit_axis_label,
            title=base_title,
            figure_path=os.path.join(output_dir, f"run_{run_idx + 1}{mode_suffix}_token_raw.png"),
            highlighted_indices=token_raw_highlight_indices,
            secondary_highlighted_indices=token_raw_secondary_highlight_indices,
            publication_mode=publication_mode,
            metadata={**raw_output_metadata, 'view': 'token_raw'},
        )
        _save_attention_matrix_outputs(
            attention_matrix=token_filtered_attention_matrix,
            tokens=token_filtered_units,
            unit_axis_label=focus_unit_axis_label,
            title=base_title,
            figure_path=os.path.join(output_dir, f"run_{run_idx + 1}{mode_suffix}_token_filtered.png"),
            highlighted_indices=token_filtered_highlight_indices,
            secondary_highlighted_indices=token_filtered_secondary_highlight_indices,
            publication_mode=publication_mode,
            metadata={**raw_output_metadata, 'view': 'token_filtered'},
        )
