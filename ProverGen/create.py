import argparse
import csv
import json
import random
from pathlib import Path


LABEL_ID_MAP = {"A": 0, "B": 1, "C": 2}
INSTRUCTION = "Given the facts below, answer the question."
SPLIT_RATIO = 0.8
SPLIT_SEED = 42
MAX_VALIDATION_ROWS = 400
MAX_EXEMPLAR_ROWS = 200
DEFAULT_LEVEL_PATHS = {
    "easy": Path("ProverGen/outputs/final_data_easy_1_1_0_0/test-0_2000.json"),
    "medium": Path("ProverGen/outputs/final_data_medium_1_1_0_0/test-0_2000.json"),
    "hard": Path("ProverGen/outputs/final_data_hard_1_1_0_0/test-0_2000.json"),
}


def build_text(sample: dict, context_key: str = "context") -> str:
    context = str(sample.get(context_key, "")).strip()
    question = str(sample.get("question", "")).strip()
    options = sample.get("options", [])

    parts = [INSTRUCTION]
    if context:
        parts.append(f"Facts:\n{context}")
    if question:
        parts.append(f"Question:\n{question}")
    if options:
        parts.append(f"Options:\n{json.dumps(options, ensure_ascii=False)}")

    return "\n\n".join(parts)


def load_samples(input_path: Path) -> list[dict]:
    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported JSON structure in {input_path}")


def filter_samples_with_context(samples: list[dict], context_key: str = "context") -> tuple[list[dict], int]:
    filtered_samples: list[dict] = []
    dropped_count = 0

    for sample in samples:
        context = str(sample.get(context_key, "")).strip()
        if not context:
            dropped_count += 1
            continue
        filtered_samples.append(sample)

    return filtered_samples, dropped_count


def convert_samples(samples: list[dict]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []

    for sample in samples:
        label = str(sample.get("answer", "")).strip()
        rows.append(
            {
                "text": build_text(sample, "context"),
                "adv_text": build_text(sample, "adv_context"),
                "label": label,
                "label_id": LABEL_ID_MAP.get(label, -1),
                "reference": str(sample.get("reasoning", "")).strip(),
            }
        )

    return rows


def write_rows(output_path: Path, rows: list[dict[str, str | int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["text", "adv_text", "label", "label_id", "reference"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_validation_rows(output_path: Path, rows: list[dict[str, str | int]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["text", "label", "label_id", "reference"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "text": row["text"],
                    "label": row["label"],
                    "label_id": row["label_id"],
                    "reference": row["reference"],
                }
            )


def split_rows(rows: list[dict[str, str | int]], ratio: float = SPLIT_RATIO) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    shuffled_rows = rows.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled_rows)
    split_index = int(len(shuffled_rows) * ratio)
    return shuffled_rows[:split_index], shuffled_rows[split_index:]


def create_exemplar_csv(input_path: Path, output_path: Path) -> None:
    samples = load_samples(input_path)
    filtered_samples, dropped_count = filter_samples_with_context(samples)
    rows = convert_samples(filtered_samples)[:MAX_EXEMPLAR_ROWS]
    write_rows(output_path, rows)

    print(f"Loaded {len(samples)} samples from {input_path}")
    print(f"Dropped {dropped_count} samples with empty context")
    print(f"Saved {len(rows)} rows to {output_path}")


def create_split_outputs(input_path: Path, validation_output_path: Path, exemplar_output_path: Path) -> None:
    samples = load_samples(input_path)
    filtered_samples, dropped_count = filter_samples_with_context(samples)
    rows = convert_samples(filtered_samples)
    validation_rows, exemplar_rows = split_rows(rows)
    validation_rows = validation_rows[:MAX_VALIDATION_ROWS]
    exemplar_rows = exemplar_rows[:MAX_EXEMPLAR_ROWS]
    write_validation_rows(validation_output_path, validation_rows)
    write_rows(exemplar_output_path, exemplar_rows)

    print(f"Loaded {len(samples)} samples from {input_path}")
    print(f"Dropped {dropped_count} samples with empty context")
    print(f"Saved {len(validation_rows)} validation rows to {validation_output_path}")
    print(f"Saved {len(exemplar_rows)} exemplar rows to {exemplar_output_path}")


def create_all_level_csvs() -> None:
    for level, input_path in DEFAULT_LEVEL_PATHS.items():
        validation_output_path = Path(f"data/proverqa_{level}/validation.csv")
        exemplar_output_path = Path(f"exemplars/proverqa_{level}.csv")
        create_split_outputs(input_path, validation_output_path, exemplar_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ProverGen final JSON data into ProverQA exemplar CSV format."
    )
    parser.add_argument(
        "--input-file",
        help="Path to one ProverGen JSON file. If omitted, generate CSVs for easy/medium/hard.",
    )
    parser.add_argument(
        "--output-file",
        help="Path to one output CSV file. Must be used together with --input-file.",
    )
    args = parser.parse_args()

    if args.input_file or args.output_file:
        if not args.input_file or not args.output_file:
            parser.error("--input-file and --output-file must be provided together.")
        create_exemplar_csv(Path(args.input_file), Path(args.output_file))
        return

    create_all_level_csvs()


if __name__ == "__main__":
    main()
