"""
Generate prompt JSONL files from dialogue data for use with call_api.py.

For each data file in data/dialogue_data/, extracts prompt_with_hist_txt
as the 'prompt' field and preserves metadata fields needed for evaluation.

Usage:
    python3 generate_prompts.py \
        --data_dir data/dialogue_data \
        --output_dir output/prompts/claude_4_5_haiku_base
"""

import argparse
import json
import os


DATA_FILES = [
    "normal.jsonl",
    "private.jsonl",
    "action_analysis.jsonl",
    "action_una.jsonl",
    "action_plotqa.jsonl",
    "action_bg.jsonl",
    "action_correction.jsonl",
    "private_action_correction.jsonl",
]


def generate_prompt_jsonl(input_path, output_path):
    """Read a dialogue data JSONL and produce a prompt JSONL for call_api.py."""
    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            # Skip non-prompt rows (e.g., private_lib header in private.jsonl)
            if "prompt_with_hist_txt" not in data:
                continue
            entry = {
                "data_id": data["data_id"],
                "domain_name": data.get("domain_name", ""),
                "result_type": data.get("result_type", ""),
                "current_query": data.get("current_query", ""),
                "reference_answer": data.get("reference_answer", ""),
                "prompt": data["prompt_with_hist_txt"],
            }
            # Preserve eval_metrics if present (needed for code gen evaluation)
            if "eval_metrics" in data:
                entry["eval_metrics"] = data["eval_metrics"]
            # Preserve ref_code fields if present
            if "ref_code_hist" in data:
                entry["ref_code_hist"] = data["ref_code_hist"]
            if "ref_code_all" in data:
                entry["ref_code_all"] = data["ref_code_all"]
            # Preserve private_lib fields if present
            if "private_lib" in data:
                entry["private_lib"] = data["private_lib"]
            if "private_lib_json" in data:
                entry["private_lib_json"] = data["private_lib_json"]

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Generate prompt JSONL files from dialogue data.")
    parser.add_argument("--data_dir", type=str, default="data/dialogue_data",
                        help="Path to dialogue data directory")
    parser.add_argument("--output_dir", type=str, default="output/prompts/claude_4_5_haiku_base",
                        help="Path to output prompt directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in DATA_FILES:
        input_path = os.path.join(args.data_dir, filename)
        if not os.path.exists(input_path):
            print(f"Skipping {filename} (not found)")
            continue

        output_path = os.path.join(args.output_dir, filename)
        count = generate_prompt_jsonl(input_path, output_path)
        print(f"{filename}: {count} prompts generated -> {output_path}")

    print(f"\nAll prompt files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
