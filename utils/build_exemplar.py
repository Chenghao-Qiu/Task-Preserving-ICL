import os
import json
import csv
import random
import argparse
from typing import List, Dict


TEXT_COLUMNS = ["text", "sentence", "content"]
QUESTION1_COLUMNS = ["question1"]
QUESTION2_COLUMNS = ["question2"]
PREMISE_COLUMNS = ["premise", "sentence1"]
HYPOTHESIS_COLUMNS = ["hypothesis", "sentence2"]
LABEL_COLUMNS = ["label", "label_text", "sentiment", "class", "gold_label"]


def _pick_first_available(row: Dict, candidates: List[str], field_name: str) -> str:
    for key in candidates:
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return str(row[key]).strip()
    raise ValueError(f"Missing required field '{field_name}'. Tried columns: {candidates}")


def _normalize_label(raw_label: str) -> str:
    label = str(raw_label).strip().lower()
    if label in {"0", "negative", "neg"}:
        return "negative"
    if label in {"1", "positive", "pos"}:
        return "positive"
    if label in {"objective", "subjective"}:
        return label
    if label in {"duplicate", "dup"}:
        return "duplicate"
    if label in {"not_duplicate", "not duplicate", "non_duplicate", "non duplicate"}:
        return "not_duplicate"
    if label in {"not_entailment", "not entailment", "non_entailment", "non entailment"}:
        return "not_entailment"
    if label in {"entailment", "entails"}:
        return "entailment"
    if label in {"neutral"}:
        return "neutral"
    if label in {"2", "contradiction", "contradict", "contradicts"}:
        return "contradiction"
    return str(raw_label).strip()


def _extract_text(row: Dict) -> str:
    try:
        return _pick_first_available(row, TEXT_COLUMNS, "text")
    except ValueError:
        try:
            question1 = _pick_first_available(row, QUESTION1_COLUMNS, "question1")
            question2 = _pick_first_available(row, QUESTION2_COLUMNS, "question2")
            return f"question1: {question1}\nquestion2: {question2}"
        except ValueError:
            pass
        premise = _pick_first_available(row, PREMISE_COLUMNS, "premise")
        hypothesis = _pick_first_available(row, HYPOTHESIS_COLUMNS, "hypothesis")
        return f"premise: {premise}\nhypothesis: {hypothesis}"


def load_classification_data(file_path: str) -> List[Dict]:
    """
    Load classification data from CSV file
    
    Args:
        file_path: Path to CSV file containing text and label columns
    
    Returns:
        List of dictionaries with text and label
    """
    data: List[Dict] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = _extract_text(row)
            label = _normalize_label(_pick_first_available(row, LABEL_COLUMNS, "label"))
            data.append({
                'text': text,
                'label': label
            })
    return data


def load_sst2_data(file_path: str) -> List[Dict]:
    return load_classification_data(file_path)


def format_exemplar(text: str, label: str) -> str:
    """
    Format a single exemplar according to the specified format
    
    Args:
        text: The sentence content
        label: The label (positive or negative)
    
    Returns:
        Formatted exemplar string
    """
    exemplar = f"sentence: {text}\nThe answer is {label}."
    return exemplar


def _balanced_sample_by_label(data: List[Dict], num_samples: int) -> List[Dict]:
    if not data or num_samples <= 0:
        return []

    samples_by_label: Dict[str, List[Dict]] = {}
    for sample in data:
        label = sample['label']
        if label not in samples_by_label:
            samples_by_label[label] = []
        samples_by_label[label].append(sample)

    labels = sorted(samples_by_label.keys())
    num_labels = len(labels)
    total_needed = min(num_samples, len(data))

    if total_needed < num_labels:
        raise ValueError(
            f"num_samples={total_needed} is smaller than label count={num_labels}. "
            "Cannot keep balanced label distribution."
        )

    base_quota = total_needed // num_labels
    remainder = total_needed % num_labels

    target_count_by_label = {label: base_quota for label in labels}
    for label in labels[:remainder]:
        target_count_by_label[label] += 1

    for label, target_count in target_count_by_label.items():
        if len(samples_by_label[label]) < target_count:
            raise ValueError(
                f"Label '{label}' has only {len(samples_by_label[label])} samples, "
                f"but balanced sampling needs {target_count}."
            )

    sampled_data: List[Dict] = []
    for label in labels:
        sampled_data.extend(random.sample(samples_by_label[label], target_count_by_label[label]))

    random.shuffle(sampled_data)
    return sampled_data


def build_exemplars_from_csv(csv_file: str, output_file: str = None,
                             max_samples: int = None) -> List[str]:
    """
    Build exemplars from classification CSV data
    
    Args:
        csv_file: Path to input CSV file
        output_file: Optional path to save exemplars as JSON file
        max_samples: Optional maximum number of samples to process
    
    Returns:
        List of formatted exemplars
    """
    print(f"Loading data from {csv_file}...")
    data = load_classification_data(csv_file)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"Building exemplars for {len(data)} samples...")
    exemplars = []
    
    for i, sample in enumerate(data):
        exemplar = format_exemplar(sample['text'], sample['label'])
        exemplars.append(exemplar)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} samples...")
    
    if output_file:
        print(f"Saving exemplars to {output_file}...")
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(exemplars, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved {len(exemplars)} exemplars to {output_file}")
    
    return exemplars


def build_exemplars_by_label(csv_file: str, output_dir: str = None) -> Dict[str, List[str]]:
    """
    Build exemplars from classification data, separated by label
    
    Args:
        csv_file: Path to input CSV file
        output_dir: Optional directory to save exemplars as separate JSON files
    
    Returns:
        Dictionary mapping labels to their exemplars
    """
    print(f"Loading data from {csv_file}...")
    data = load_classification_data(csv_file)
    
    exemplars_by_label = {}
    
    print(f"Building exemplars separated by label...")
    for sample in data:
        label = sample['label']
        exemplar = format_exemplar(sample['text'], label)
        
        if label not in exemplars_by_label:
            exemplars_by_label[label] = []
        
        exemplars_by_label[label].append(exemplar)
    
    # Print statistics
    print(f"\nExemplar statistics:")
    for label, exemplars in exemplars_by_label.items():
        print(f"  {label}: {len(exemplars)} samples")
    
    if output_dir:
        print(f"\nSaving exemplars to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        for label, exemplars in exemplars_by_label.items():
            output_file = os.path.join(output_dir, f"{label}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(exemplars, f, ensure_ascii=False, indent=2)
            print(f"✓ Saved {len(exemplars)} {label} exemplars to {output_file}")
    
    return exemplars_by_label


def build_random_exemplars_txt(csv_file: str, output_file: str, num_samples: int = 16, 
                                seed: int = None) -> List[str]:
    """
    Build random sampled exemplars and save to txt file
    
    Args:
        csv_file: Path to input CSV file
        output_file: Path to output txt file
        num_samples: Number of exemplars to randomly sample (default 16)
        seed: Random seed for reproducibility
    
    Returns:
        List of formatted exemplars
    """
    if seed is not None:
        random.seed(seed)
    
    print(f"Loading data from {csv_file}...")
    data = load_classification_data(csv_file)
    
    print(f"Randomly sampling {num_samples} exemplars with balanced label distribution...")
    sampled_data = _balanced_sample_by_label(data, num_samples)

    sampled_stats: Dict[str, int] = {}
    for sample in sampled_data:
        label = sample['label']
        sampled_stats[label] = sampled_stats.get(label, 0) + 1

    print("Sampled label distribution:")
    for label in sorted(sampled_stats.keys()):
        print(f"  {label}: {sampled_stats[label]}")
    
    exemplars = []
    for sample in sampled_data:
        exemplar = format_exemplar(sample['text'], sample['label'])
        exemplars.append(exemplar)
    
    print(f"Saving to {output_file}...")
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for exemplar in exemplars:
            f.write(exemplar + '\n\n')
    
    print(f"✓ Saved {len(exemplars)} exemplars to {output_file}")
    
    return exemplars


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build ICL exemplars from CSV')
    parser.add_argument('--dataset', type=str, default='sst2',
                        help='Dataset name used in default paths, e.g. sst2, subj, mnli, qqp, or rte')
    parser.add_argument('--csv-file', type=str, default='',
                        help='Input CSV path. If empty, use ../data/<dataset>/train.csv')
    parser.add_argument('--num-samples', type=int, default=16,
                        help='Number of exemplars to sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-file', type=str, default='',
                        help='Output txt path. If empty, use ../exemplars/<dataset>_examples_<num>.txt')
    args = parser.parse_args()

    csv_file = args.csv_file or f'../data/{args.dataset}/train.csv'
    output_txt = args.output_file or f'../exemplars/{args.dataset}_examples_{args.num_samples}.txt'

    exemplars = build_random_exemplars_txt(
        csv_file=csv_file,
        output_file=output_txt,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # Display the exemplars
    print("\nGenerated exemplars:")
    print("=" * 60)
    for i, exemplar in enumerate(exemplars, 1):
        print(f"{i}.\n{exemplar}\n")
