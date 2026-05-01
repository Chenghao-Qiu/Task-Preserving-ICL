import os
import csv
from typing import List, Dict
from torch.utils.data import Dataset


TEXT_COLUMNS = ["text", "sentence", "content"]
PREMISE_COLUMNS = ["premise", "sentence1"]
HYPOTHESIS_COLUMNS = ["hypothesis", "sentence2"]
LABEL_COLUMNS = ["label", "label_text", "sentiment", "class", "gold_label"]


def _pick_first_available(row: Dict, candidates: List[str], field_name: str) -> str:
    for key in candidates:
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return str(row[key]).strip()
    raise ValueError(f"Missing required field '{field_name}'. Tried columns: {candidates}")


def _extract_text(row: Dict) -> str:
    try:
        return _pick_first_available(row, TEXT_COLUMNS, "text")
    except ValueError:
        premise = _pick_first_available(row, PREMISE_COLUMNS, "premise")
        hypothesis = _pick_first_available(row, HYPOTHESIS_COLUMNS, "hypothesis")
        return f"premise: {premise}\nhypothesis: {hypothesis}"


class ExemplarDataset(Dataset):
    """Dataset for loading exemplars"""
    
    def __init__(self, exemplar_file: str):
        self.exemplars = []
        with open(exemplar_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Split by double newline to get individual exemplars
            self.exemplars = content.split('\n\n')
            self.exemplars = [e.strip() for e in self.exemplars if e.strip()]
    
    def get_exemplars(self) -> str:
        """Get all exemplars as a single prompt"""
        return '\n\n'.join(self.exemplars)


class TestDataset(Dataset):
    """Dataset for loading test data"""
    
    def __init__(self, csv_file: str, max_samples: int = None):
        self.data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = _extract_text(row)
                label = _pick_first_available(row, LABEL_COLUMNS, "label")
                self.data.append({
                    'text': text,
                    'label': label
                })
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be > 0")
            self.data = self.data[:max_samples]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
