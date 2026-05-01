import csv
import random
from typing import Dict, List, Tuple

from utils.build_exemplar import (
    LABEL_COLUMNS,
    _extract_text,
    _normalize_label,
    _pick_first_available,
)
from utils.dataset_config import resolve_dataset_name


def _flip_binary_label(label: str, labels: Tuple[str, str]) -> str:
    if label == labels[0]:
        return labels[1]
    if label == labels[1]:
        return labels[0]
    raise ValueError(f"Cannot flip label '{label}' with labels={labels}")


def load_exemplar_candidates(
    file_path: str,
    dataset_name: str = '',
    labels: Tuple[str, ...] = (),
) -> List[Dict]:
    """Load exemplar candidates from CSV, preserving optional adversarial fields."""
    data: List[Dict] = []
    flip_orig_label_for_checklist = resolve_dataset_name(dataset_name) == 'sst2_checklist'
    if flip_orig_label_for_checklist and len(labels) != 2:
        raise ValueError("sst2_checklist label flipping requires exactly 2 labels")

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = _extract_text(row)
            label = _normalize_label(_pick_first_available(row, LABEL_COLUMNS, "label"))
            adv_text = str(row.get('adv_text', '')).strip() or None
            reference = str(row.get('reference', '')).strip() or None
            adv_reference = str(row.get('adv_reference', '')).strip() or None
            data.append({
                'text': text,
                'adv_text': adv_text,
                'reference': reference,
                'adv_reference': adv_reference,
                'label': _flip_binary_label(label, labels) if flip_orig_label_for_checklist else label,
                'adv_label': label,
            })
    return data


def _compute_balanced_counts(labels: List[str], total_needed: int) -> Dict[str, int]:
    if total_needed <= 0:
        return {label: 0 for label in labels}

    base_quota = total_needed // len(labels)
    remainder = total_needed % len(labels)
    counts = {label: base_quota for label in labels}
    for label in labels[:remainder]:
        counts[label] += 1
    return counts


def _format_adv_sample(sample: Dict, task_type: str) -> Dict:
    return {
        'text': sample.get('adv_text') or sample['text'],
        'label': sample.get('adv_label', sample['label']),
        'reference': (
            sample.get('reference')
            if task_type == 'qa'
            else (sample.get('adv_reference') or sample.get('reference'))
        ),
        'source': 'adv',
    }


def _format_orig_sample(sample: Dict) -> Dict:
    return {
        'text': sample['text'],
        'label': sample['label'],
        'reference': sample.get('reference'),
        'source': 'orig',
    }


def _arrange_selected_exemplars(
    adv_selected: List[Dict],
    orig_selected: List[Dict],
    adv_placement: str,
    task_type: str,
) -> List[Dict]:
    adv_formatted = [_format_adv_sample(sample, task_type) for sample in adv_selected]
    orig_formatted = [_format_orig_sample(sample) for sample in orig_selected]

    if adv_placement == 'random':
        selected = adv_formatted + orig_formatted
        random.shuffle(selected)
        return selected

    random.shuffle(adv_formatted)
    random.shuffle(orig_formatted)
    if adv_placement == 'head':
        return adv_formatted + orig_formatted
    if adv_placement == 'medium':
        middle_idx = len(orig_formatted) // 2
        return orig_formatted[:middle_idx] + adv_formatted + orig_formatted[middle_idx:]
    if adv_placement == 'tail':
        return orig_formatted + adv_formatted
    if adv_placement == 'custom':
        front_adv = adv_formatted[:2]
        tail_adv = adv_formatted[2:]
        return front_adv + orig_formatted + tail_adv
    raise ValueError(
        f"Unsupported adversarial exemplar placement: {adv_placement}. "
        "Expected one of: random, head, medium, tail, custom."
    )


def sample_mixed_exemplars(
    data: List[Dict],
    num_exemplars: int,
    num_adv_exemplars: int,
    sampling: str,
    adv_placement: str = 'random',
    task_type: str = 'classification',
) -> List[Dict]:
    """Sample exemplars and place adversarial rows according to adv_placement."""
    if task_type == 'math' and sampling == 'balanced':
        sampling = 'random'

    if num_exemplars <= 0:
        return []
    if num_adv_exemplars < 0:
        raise ValueError("--num-adv-exemplars must be >= 0")
    if num_adv_exemplars > num_exemplars:
        raise ValueError("--num-adv-exemplars cannot exceed --num-exemplars")
    if adv_placement not in {'random', 'head', 'medium', 'tail', 'custom'}:
        raise ValueError("--adv-placement must be one of: random, head, medium, tail, custom")
    if num_exemplars > len(data):
        raise ValueError(
            f"Requested {num_exemplars} exemplars, but exemplar pool only has {len(data)} rows."
        )

    adv_eligible = [
        sample for sample in data
        if sample.get('adv_text') or sample.get('adv_reference')
    ]
    if num_adv_exemplars > len(adv_eligible):
        raise ValueError(
            f"Requested {num_adv_exemplars} adversarial exemplars, but only "
            f"{len(adv_eligible)} rows contain adv_text."
        )

    if sampling == 'random':
        adv_selected = random.sample(adv_eligible, num_adv_exemplars)
        adv_selected_ids = {id(sample) for sample in adv_selected}
        remaining_pool = [sample for sample in data if id(sample) not in adv_selected_ids]
        orig_selected = random.sample(remaining_pool, num_exemplars - num_adv_exemplars)
        return _arrange_selected_exemplars(
            adv_selected=adv_selected,
            orig_selected=orig_selected,
            adv_placement=adv_placement,
            task_type=task_type,
        )

    samples_by_label: Dict[str, List[Dict]] = {}
    adv_by_label: Dict[str, List[Dict]] = {}
    for sample in data:
        label = sample['label']
        samples_by_label.setdefault(label, []).append(sample)
        if sample.get('adv_text') or sample.get('adv_reference'):
            adv_by_label.setdefault(label, []).append(sample)

    labels = sorted(samples_by_label.keys())
    if num_exemplars < len(labels):
        raise ValueError(
            f"--num-exemplars={num_exemplars} is smaller than label count={len(labels)}. "
            "Balanced sampling cannot keep all labels represented."
        )

    total_counts = _compute_balanced_counts(labels, num_exemplars)
    adv_counts = _compute_balanced_counts(labels, num_adv_exemplars)

    for label in labels:
        total_count = total_counts[label]
        adv_count = adv_counts[label]
        if len(samples_by_label[label]) < total_count:
            raise ValueError(
                f"Label '{label}' has only {len(samples_by_label[label])} samples, "
                f"but balanced sampling needs {total_count}."
            )
        if len(adv_by_label.get(label, [])) < adv_count:
            raise ValueError(
                f"Label '{label}' has only {len(adv_by_label.get(label, []))} adversarial samples, "
                f"but balanced sampling needs {adv_count}."
            )

    adv_selected_all = []
    orig_selected_all = []
    for label in labels:
        adv_selected = random.sample(adv_by_label.get(label, []), adv_counts[label])
        adv_selected_ids = {id(sample) for sample in adv_selected}
        remaining_label_pool = [
            sample for sample in samples_by_label[label]
            if id(sample) not in adv_selected_ids
        ]
        orig_selected = random.sample(
            remaining_label_pool,
            total_counts[label] - adv_counts[label],
        )
        adv_selected_all.extend(adv_selected)
        orig_selected_all.extend(orig_selected)

    return _arrange_selected_exemplars(
        adv_selected=adv_selected_all,
        orig_selected=orig_selected_all,
        adv_placement=adv_placement,
        task_type=task_type,
    )
