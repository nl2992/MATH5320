# Submission Package

This folder contains all deliverables and supporting evidence for the MATH GR 5320 project.

## Contents

| File | Description |
|---|---|
| `00_combined_final_report.md` | **Primary submission document.** Integrates all five deliverables into one consistent report with full formula-sheet demonstration evidence. |
| `01_model_documentation.md` | Deliverable 1 — Model documentation (30 pts) |
| `02_software_design_documentation.md` | Deliverable 2 — Software design documentation (15 pts) |
| `03_test_plan.md` | Deliverable 3 — Test plan (20 pts) |
| `04_test_results.md` | Deliverable 4/5 — Test results (10 pts) |
| `05_guide_gap_review.md` | Working gap review memo |
| `06_prompt_coverage_matrix.md` | Requirement coverage matrix |
| `demo.ipynb` | **Formula-sheet demonstration notebook.** All 15 course sections (§1–§15) fully executed with outputs and assertions. |
| `demo.md` | **Front-end trace companion.** Screenshots of each Streamlit tab side-by-side with matching notebook outputs. |
| `coverage_report/` | HTML and XML coverage reports from the local pytest run |
| `test_artifacts/` | Captured artifacts: git hash, pytest output, coverage, environment, backtest results |

## Quick start

```bash
# Run all no-network unit tests (576 pass)
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py

# Execute the formula-sheet demo notebook
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 \
  --ExecutePreprocessor.kernel_name=python3 \
  submission/demo.ipynb

# Launch the Streamlit application
streamlit run app.py
```
