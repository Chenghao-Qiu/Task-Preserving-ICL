import json

from utils.build_exemplar import format_exemplar


def _normalize_reference(reference: str) -> str:
    normalized = reference.strip()
    if normalized.startswith('["') and normalized.endswith('"]'):
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, list) and parsed:
                return str(parsed[0]).strip()
        except json.JSONDecodeError:
            pass
    return normalized


def format_prompt_exemplar(
    text: str,
    label: str,
    reference: str = None,
    task_type: str = 'classification',
) -> str:
    """Keep pair inputs in their native format instead of prepending sentence: twice."""
    is_generative_task = task_type in {'qa', 'math'}
    normalized = text.strip().lower()
    is_pair_text = (
        normalized.startswith('premise:') and 'hypothesis:' in normalized
    ) or (
        normalized.startswith('question1:') and 'question2:' in normalized
    ) or (
        normalized.startswith('question:') and 'sentence:' in normalized
    )
    if is_generative_task:
        qa_output = json.dumps(
            {
                'reasoning': _normalize_reference(reference) if reference else "",
                'answer': label,
            },
            ensure_ascii=False,
            indent=2,
        )
        return f"{text}\n\n{qa_output}"
    if is_pair_text:
        return f"{text}\nThe answer is {label}."
    return format_exemplar(text, label)
