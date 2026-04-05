# Tapilot-Crossing Data

This directory contains all data for the Tapilot-Crossing benchmark: task definitions in `dialogue_data/` and supporting datasets in `resource/`.

---

## Dialogue Data (`dialogue_data/`)

The benchmark comprises **946 entries** across 8 JSONL files, split into two evaluation modes.

### Multi-Choice Tasks (426 entries)

| File | Entries | Category | Description |
|------|---------|----------|-------------|
| `action_analysis.jsonl` | 218 | Insight Mining | The agent interprets analysis results and extracts insights (e.g., trends, correlations) to support decision-making, beyond just generating code. |
| `action_una.jsonl` | 106 | Fast Fail | The agent detects that a question cannot be answered due to missing data or invalid assumptions and explicitly reports it. |
| `action_bg.jsonl` | 33 | Best Guess | For under-specified queries, the agent makes reasonable assumptions based on data or commonsense instead of asking for clarification. |
| `action_plotqa.jsonl` | 69 | Plot QA | The agent answers questions based on visualizations, requiring understanding of plots and relationships between variables. |

### Code Generation Tasks (520 entries)

| File | Entries | Category | Description |
|------|---------|----------|-------------|
| `normal.jsonl` | 283 | Normal | Fully specified queries where no interaction or clarification is needed; the agent directly produces code or answers. |
| `private.jsonl` | 206 | Private | Involves user-defined/private libraries. Tests the agent's ability to understand and use unseen APIs rather than relying on standard libraries. The first line contains the private library definition (`private_lib`, `private_lib_json` fields); data entries start from line 2. |
| `action_correction.jsonl` | 16 | Update Code | The agent fixes bugs or refines previously generated code based on user feedback or errors. |
| `private_action_correction.jsonl` | 15 | Private + Update Code | Combination of private and action_correction. The agent must both handle private libraries and iteratively fix/update code based on feedback. |

### Data Categories Explained

1. **action_analysis**: Corresponds to *Insight_Mining*. The agent interprets analysis results and extracts insights (e.g., trends, correlations) to support decision-making, beyond just generating code.

2. **action_bg (best guess)**: Corresponds to *Best_Guess*. For under-specified queries, the agent makes reasonable assumptions based on data or commonsense instead of asking for clarification.

3. **action_correction**: Corresponds to *Update_Code*. The agent fixes bugs or refines previously generated code based on user feedback or errors.

4. **action_plotqa**: Corresponds to *Plot_QA*. The agent answers questions based on visualizations, requiring understanding of plots and relationships between variables.

5. **action_una (unanswerable)**: Corresponds to *Fast_Fail*. The agent detects that a question cannot be answered due to missing data or invalid assumptions and explicitly reports it.

6. **normal**: Fully specified queries where no interaction or clarification is needed; the agent directly produces code or answers.

7. **private**: Involves user-defined/private libraries. Tests the agent's ability to understand and use unseen APIs rather than relying on standard libraries.

8. **private_action_correction**: Combination of *private* and *action_correction*. The agent must both handle private libraries and iteratively fix/update code based on feedback.

### Entries vs. Intents

The benchmark has **946 entries** but evaluation reports **1094 total intents**. This is because some code generation entries contain **multiple user intents** in a single entry (indicated by `result_type` being a list). For example, a single entry might ask the agent to both filter a dataframe and plot a chart, this counts as 2 intents. Each intent is evaluated independently: the entry passes only if all intents pass, and each intent contributes separately to the total score.

- Multi-choice: 426 entries = 430 intents
- Code generation: 520 entries = 664 intents
- **Total: 946 entries = 1094 intents**

### JSONL Schema

Each entry contains the following fields:

| Field | Description |
|-------|-------------|
| `data_id` | Unique identifier for the entry |
| `domain_name` | One of: `credit_card_risk`, `ATP_tennis`, `fast_food`, `laptop_price`, `melb_housing` |
| `result_type` | Expected output type: `dataframe`, `plot`, `value`, `list`, `multi_choice`, `unanswerable`, etc. |
| `current_query` | The user's current-turn query |
| `prompt_with_hist_txt` | Full prompt including system context and dialogue history (used as LLM input) |
| `prompt_with_hist_json` | Same prompt in OpenAI Chat message format |
| `reference_answer` | Ground truth code (code gen) or correct answer JSON (multi-choice) |
| `ref_code_hist` | Accumulated code from all previous dialogue turns |
| `ref_code_all` | `ref_code_hist` + current turn reference code (full executable reference) |
| `eval_metrics` | Python evaluation code that compares predicted vs reference outputs |

---

## Resource Data (`resource/`)

CSV datasets and the private function library used by the generated code during evaluation.

### Datasets

| File | Domain | Rows | Columns | Description |
|------|--------|------|---------|-------------|
| `credit_customers.csv` | credit_card_risk | 1,000 | 21 | Credit card risk assessment (loan approval, credit scoring, risk factors) |
| `atp_tennis.csv` | ATP_tennis | ~25,000 | 49 | ATP tennis tournament data (player stats, match outcomes, surface analysis) |
| `fastfood.csv` | fast_food | 515 | 17 | Fast food nutrition data (calorie analysis, health scoring) |
| `laptops_price.csv` | laptop_price | 325 | 13 | Laptop pricing data (price prediction, spec comparison) |
| `melb_data.csv` | melb_housing | ~13,500 | 21 | Melbourne housing data (housing prices, suburb analysis) |

### Private Function Library

| File | Description |
|------|-------------|
| `decision_company.py` | Python module with 133 custom functions used by `private` and `private_action_correction` tasks. Simulates a company's internal toolkit that the agent must learn to use from documentation alone. |
| `decision_company.json` | JSON version of the private function library (structured documentation). |
