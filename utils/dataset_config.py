DATASET_CONFIGS = {
    'sst2': {
        'labels': ('negative', 'positive'),
        'task_type': 'nli',
        'exemplar_pool': 'exemplars/sst2.csv',
        'test_data': 'data/sst2/validation.csv',
    },
    'sst2_checklist': {
        'labels': ('negative', 'positive'),
        'task_type': 'nli',
        'exemplar_pool': 'references/checklist/sst2.csv',
        'test_data': 'data/sst2/validation.csv',
    },
    'mnli': {
        'labels': ('entailment', 'neutral', 'contradiction'),
        'task_type': 'nli',
        'exemplar_pool': 'exemplars/mnli.csv',
        'test_data': 'data/mnli/validation.csv',
    },
    'qqp': {
        'labels': ('not_duplicate', 'duplicate'),
        'task_type': 'nli',
        'exemplar_pool': 'exemplars/qqp.csv',
        'test_data': 'data/qqp/validation.csv',
    },
    'rte': {
        'labels': ('not_entailment', 'entailment'),
        'task_type': 'nli',
        'exemplar_pool': 'exemplars/rte.csv',
        'test_data': 'data/rte/validation.csv',
    },
    'proverqa': {
        'labels': ('A', 'B', 'C'),
        'task_type': 'qa',
        'exemplar_pool': 'data/proverqa/train.csv',
        'test_data': 'data/proverqa/eval.csv',
    },
    'problemathic': {
        'labels': (),
        'task_type': 'math',
        'exemplar_pool': 'exemplars/problemathic_simple.csv',
        'test_data': 'exemplars/problemathic_simple.csv',
    },
}

DATASET_ALIASES = {
    'sst2_checklist_simplified': 'sst2_checklist',
    'sst2_checklist_matched': 'sst2_checklist',
    'sst2_checklist_matched_instruction': 'sst2_checklist',
    'sst2_irrelevant': 'sst2',
    'sst2_irrelevant_instruction': 'sst2',
    'sst2_checklist_reversed': 'sst2',
    'proverqa_easy': 'proverqa',
    'proverqa_medium': 'proverqa',
    'proverqa_hard': 'proverqa',
    'problemathic_simple': 'problemathic',
    'problemathic_complex': 'problemathic',
}

DATASET_ALIAS_PREFIXES = {
    'proverqa_easy': 'proverqa',
    'proverqa_medium': 'proverqa',
    'proverqa_hard': 'proverqa',
}


def resolve_dataset_name(dataset_name: str) -> str:
    if dataset_name in DATASET_ALIASES:
        return DATASET_ALIASES[dataset_name]

    for prefix, target in DATASET_ALIAS_PREFIXES.items():
        if dataset_name == prefix or dataset_name.startswith(f'{prefix}_'):
            return target

    return dataset_name


def is_alias_dataset_name(dataset_name: str) -> bool:
    if dataset_name in DATASET_ALIASES:
        return True

    return any(
        dataset_name == prefix or dataset_name.startswith(f'{prefix}_')
        for prefix in DATASET_ALIAS_PREFIXES
    )


DATASET_CHOICES = list(DATASET_CONFIGS.keys()) + list(DATASET_ALIASES.keys())
