"""
Self-contained evaluation script for Tapilot-Crossing benchmark.
Works directly with JSONL response files from call_api.py.

Handles two evaluation modes:
1. Multi-choice: action_analysis, action_una, action_bg, action_plotqa
2. Code generation: normal, private, action_correction, private_action_correction

Usage:
    python3 evaluate.py \
        --response_dir output/responses/claude_4_5_haiku_base \
        --resource_dir data/resource \
        --model_name claude_4_5_haiku
"""

import argparse
import json
import os
import re
import sys
import signal
import tempfile
import shutil
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


# =============================================
# Multi-choice evaluation
# =============================================
MULTI_CHOICE_FILES = [
    "action_analysis.jsonl",
    "action_una.jsonl",
    "action_bg.jsonl",
    "action_plotqa.jsonl",
]

CODE_GEN_FILES = [
    "normal.jsonl",
    "private.jsonl",
    "action_correction.jsonl",
    "private_action_correction.jsonl",
]


def parse_correct_answer(reference_answer):
    """Extract correct answer letter from reference_answer field."""
    match = re.search(r'"correct_answer":\s*"([A-J])"', reference_answer)
    if match:
        return match.group(1).upper()
    match = re.search(r'"correct_answer":\s*([A-J])', reference_answer)
    if match:
        return match.group(1).upper()
    return ""


def extract_model_answer(response):
    """Extract answer letter from model response."""
    if '</choice>' in response and '\nAnswer:' not in response:
        match = re.search(r'\b([A-J])\.?\b[^<]*</choice>', response)
        if match:
            return match.group(1).upper()

    if '\nAnswer:' in response and '</choice>' not in response:
        match = re.search(r'\nAnswer:\s*\b([A-J])\.?\b', response)
        if match:
            return match.group(1).upper()

    if '\nAnswer:' not in response and '</choice>' not in response:
        match = re.search(r'(?:answer|choice|option)\s*(?:is|:)\s*\**\s*([A-J])\b', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        match = re.search(r'\b([A-J])\.\s', response)
        if match:
            return match.group(1).upper()
        match = re.search(r'\b([A-J])\b', response)
        if match:
            return match.group(1).upper()

    return ""


def evaluate_multi_choice(response_path):
    """Evaluate multi-choice tasks. Returns (correct, total, details)."""
    if not os.path.exists(response_path):
        return 0, 0, []

    results = []
    correct = 0
    total = 0

    with open(response_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            ref = entry.get("reference_answer", "")
            resp = entry.get("response", "")
            data_id = entry.get("data_id", "?")

            correct_ans = parse_correct_answer(ref)
            model_ans = extract_model_answer(resp)

            is_correct = (model_ans == correct_ans) if correct_ans else False
            total += 1
            if is_correct:
                correct += 1

            results.append({
                "data_id": data_id,
                "correct_answer": correct_ans,
                "model_answer": model_ans,
                "is_correct": is_correct,
            })

    return correct, total, results


# =============================================
# Code generation evaluation
# =============================================
CSV_MAP = {
    "credit_card_risk": "credit_customers.csv",
    "ATP_tennis": "atp_tennis.csv",
    "fast_food": "fastfood.csv",
    "laptop_price": "laptops_price.csv",
    "melb_housing": "melb_data.csv",
}


def extract_code_from_response(response):
    """Extract Python code from LLM response."""
    code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return max(code_blocks, key=len)

    if 'import ' in response:
        idx = response.find('import ')
        code = response[idx:]
        code = re.sub(r'```\s*$', '', code)
        code = re.sub(r"'''\s*$", '', code)
        return code

    return response


def fix_csv_paths(code, abs_resource):
    """Fix CSV file paths in code to point to absolute resource directory."""
    for domain, csv_name in CSV_MAP.items():
        csv_path = os.path.join(abs_resource, csv_name)
        code = code.replace(f'"{csv_name}"', f'"{csv_path}"')
        code = code.replace(f"'{csv_name}'", f"'{csv_path}'")
    code = re.sub(
        r'["\'][^\s"\']*/(credit_customers|atp_tennis|fastfood|laptops_price|melb_data)\.csv["\']',
        lambda m: f'"{os.path.join(abs_resource, m.group(1) + ".csv")}"',
        code
    )
    return code


def _run_subprocess(python_exe, script_path, args, cwd, timeout):
    """Run a subprocess with proper timeout and process group killing."""
    try:
        proc = subprocess.Popen(
            [python_exe, script_path] + args,
            cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # new process group for clean kill
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout.decode('utf-8', errors='replace'), stderr.decode('utf-8', errors='replace')
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait()
        return -1, "", "TIMEOUT"


def evaluate_single_code_gen(entry, abs_resource, timeout=30):
    """
    Evaluate a single code generation entry.
    Returns (data_id, is_correct, num_intents, error_msg)
    """
    data_id = entry.get("data_id", "?")
    result_type = entry.get("result_type", "")
    response = entry.get("response", "")
    ref_code_hist = entry.get("ref_code_hist", "")
    reference_answer = entry.get("reference_answer", "")
    eval_metrics = entry.get("eval_metrics", "")
    python_exe = sys.executable

    num_intents = len(result_type) if isinstance(result_type, list) else 1

    if not response or not eval_metrics:
        return data_id, False, num_intents, "missing response or eval_metrics"

    tmpdir = tempfile.mkdtemp(prefix=f"tapilot_eval_{data_id}_")
    try:
        os.makedirs(os.path.join(tmpdir, "ref_result"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "pred_result"), exist_ok=True)

        # Symlink CSV files and decision_company.py
        for csv_name in CSV_MAP.values():
            src = os.path.join(abs_resource, csv_name)
            dst = os.path.join(tmpdir, csv_name)
            if os.path.exists(src):
                os.symlink(src, dst)
        dc_src = os.path.join(abs_resource, "decision_company.py")
        if os.path.exists(dc_src):
            os.symlink(dc_src, os.path.join(tmpdir, "decision_company.py"))

        preamble = "import warnings\nwarnings.filterwarnings('ignore')\nimport matplotlib\nmatplotlib.use('Agg')\n"

        # === 1. Reference code ===
        ref_code_all = entry.get("ref_code_all", "")
        if ref_code_all:
            ref_code = ref_code_all
        else:
            ref_code = reference_answer
        if 'read_csv' not in ref_code and 'read_csv' in reference_answer:
            ref_code = reference_answer
        # Fix paths
        ref_code = fix_csv_paths(ref_code, abs_resource)
        ref_code = ref_code.replace("./pred_result/", "./ref_result/")

        ref_path = os.path.join(tmpdir, "_ref.py")
        with open(ref_path, "w") as f:
            f.write(preamble + ref_code)

        rc, _, stderr = _run_subprocess(python_exe, ref_path, [abs_resource], tmpdir, timeout)
        if rc != 0:
            err = "ref timeout" if stderr == "TIMEOUT" else f"ref failed: {stderr[:200]}"
            return data_id, False, num_intents, err

        # === 2. Predicted code ===
        extracted = extract_code_from_response(response)
        if ref_code_hist:
            pred_code = ref_code_hist + "\n\n" + extracted
        else:
            pred_code = extracted

        pred_code = fix_csv_paths(pred_code, abs_resource)
        pred_code = pred_code.replace("./ref_result/", "./pred_result/")
        pred_code = re.sub(r'plt\.show\(\)', '', pred_code)
        pred_code = re.sub(r'show_plots\(\)', '', pred_code)

        pred_path = os.path.join(tmpdir, "_pred.py")
        with open(pred_path, "w") as f:
            f.write(preamble + pred_code)

        rc, _, stderr = _run_subprocess(python_exe, pred_path, [abs_resource], tmpdir, timeout)
        if rc != 0:
            err = "pred timeout" if stderr == "TIMEOUT" else f"pred failed: {stderr[:200]}"
            return data_id, False, num_intents, err

        # === 3. Eval metrics ===
        eval_code = fix_csv_paths(eval_metrics, abs_resource)
        eval_path = os.path.join(tmpdir, "_eval.py")
        with open(eval_path, "w") as f:
            f.write(preamble + eval_code)

        rc, stdout, stderr = _run_subprocess(python_exe, eval_path, [abs_resource], tmpdir, timeout)
        if rc != 0:
            err = "eval timeout" if stderr == "TIMEOUT" else f"eval failed: {stderr[:200]}"
            return data_id, False, num_intents, err

        output = stdout.strip()
        if not output:
            return data_id, False, num_intents, "no eval output"

        # === 4. Parse result ===
        lines = output.strip().split("\n")
        all_pass = True
        for line in lines:
            line = line.strip()
            if line == "True":
                continue
            elif line == "False":
                all_pass = False
            else:
                try:
                    val = float(line)
                    if val <= 0.6:
                        all_pass = False
                except ValueError:
                    all_pass = False

        return data_id, all_pass, num_intents, ""

    except Exception as e:
        return data_id, False, num_intents, str(e)[:200]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def evaluate_code_gen_parallel(response_path, abs_resource, timeout=30, max_workers=8):
    """Evaluate code generation tasks in parallel."""
    if not os.path.exists(response_path):
        return 0, 0, []

    with open(response_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    correct = 0
    total = 0
    results = []
    n = len(entries)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, entry in enumerate(entries):
            fut = executor.submit(evaluate_single_code_gen, entry, abs_resource, timeout)
            future_to_idx[fut] = i

        done_count = 0
        for future in as_completed(future_to_idx):
            done_count += 1
            idx = future_to_idx[future]
            entry = entries[idx]
            try:
                data_id, is_correct, num_intents, error_msg = future.result()
            except Exception as e:
                data_id = entry.get("data_id", "?")
                is_correct = False
                num_intents = len(entry.get("result_type", "")) if isinstance(entry.get("result_type"), list) else 1
                error_msg = str(e)[:200]

            total += num_intents
            if is_correct:
                correct += num_intents
                status = "PASS"
            else:
                status = "FAIL"

            suffix = f" ({error_msg})" if error_msg else ""
            print(f"  [{done_count}/{n}] data_id={data_id} -> {status}{suffix}", flush=True)
            results.append({"data_id": data_id, "correct": is_correct, "error": error_msg})

    return correct, total, results


# =============================================
# Main
# =============================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Tapilot-Crossing benchmark responses.")
    parser.add_argument("--response_dir", type=str, required=True)
    parser.add_argument("--resource_dir", type=str, default="data/resource")
    parser.add_argument("--model_name", type=str, default="claude_4_5_haiku")
    parser.add_argument("--code_gen_timeout", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for code gen eval")
    args = parser.parse_args()

    abs_resource = os.path.abspath(args.resource_dir)

    print(f"{'='*60}")
    print(f"Tapilot-Crossing Evaluation: {args.model_name}")
    print(f"{'='*60}\n")

    overall_correct = 0
    overall_total = 0
    category_results = {}

    # --- Multi-choice (instant) ---
    print("--- MULTI-CHOICE EVALUATION ---\n")
    for filename in MULTI_CHOICE_FILES:
        response_path = os.path.join(args.response_dir, filename)
        category = filename.replace(".jsonl", "")
        if not os.path.exists(response_path):
            print(f"  {category}: SKIPPED (no response file)")
            continue

        correct, total, details = evaluate_multi_choice(response_path)
        overall_correct += correct
        overall_total += total
        acc = correct / total * 100 if total > 0 else 0
        category_results[category] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  {category}: {correct}/{total} = {acc:.2f}%")

    # --- Code generation (parallel) ---
    print(f"\n--- CODE GENERATION EVALUATION (workers={args.workers}, timeout={args.code_gen_timeout}s) ---\n")
    for filename in CODE_GEN_FILES:
        response_path = os.path.join(args.response_dir, filename)
        category = filename.replace(".jsonl", "")
        if not os.path.exists(response_path):
            print(f"  {category}: SKIPPED (no response file)")
            continue

        print(f"  Evaluating {category}...")
        correct, total, details = evaluate_code_gen_parallel(
            response_path, abs_resource,
            timeout=args.code_gen_timeout, max_workers=args.workers
        )
        overall_correct += correct
        overall_total += total
        acc = correct / total * 100 if total > 0 else 0
        category_results[category] = {"correct": correct, "total": total, "accuracy": acc}
        print(f"  >> {category}: {correct}/{total} = {acc:.2f}%\n")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model_name} (base)")
    print(f"{'='*60}")
    print(f"{'Category':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*60}")

    mc_correct = mc_total = 0
    cg_correct = cg_total = 0

    for cat, res in sorted(category_results.items()):
        print(f"{cat:<35} {res['correct']:>8} {res['total']:>8} {res['accuracy']:>9.2f}%")
        if cat in ["action_analysis", "action_una", "action_bg", "action_plotqa"]:
            mc_correct += res["correct"]
            mc_total += res["total"]
        else:
            cg_correct += res["correct"]
            cg_total += res["total"]

    print(f"{'-'*60}")
    if mc_total > 0:
        print(f"{'Multi-choice Total':<35} {mc_correct:>8} {mc_total:>8} {mc_correct/mc_total*100:>9.2f}%")
    if cg_total > 0:
        print(f"{'Code Generation Total':<35} {cg_correct:>8} {cg_total:>8} {cg_correct/cg_total*100:>9.2f}%")
    if overall_total > 0:
        print(f"{'OVERALL':<35} {overall_correct:>8} {overall_total:>8} {overall_correct/overall_total*100:>9.2f}%")
    print(f"{'='*60}")

    # Save results
    results_path = os.path.join(args.response_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "categories": category_results,
            "multi_choice_total": {"correct": mc_correct, "total": mc_total,
                                   "accuracy": mc_correct / mc_total * 100 if mc_total > 0 else 0},
            "code_gen_total": {"correct": cg_correct, "total": cg_total,
                               "accuracy": cg_correct / cg_total * 100 if cg_total > 0 else 0},
            "overall": {"correct": overall_correct, "total": overall_total,
                         "accuracy": overall_correct / overall_total * 100 if overall_total > 0 else 0},
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
