# Submission Package

This folder contains all deliverables and supporting evidence for the MATH GR 5320 project.

## Contents

| File | Purpose |
|---|---|
| `00_combined_final_report.md` | Primary final report |
| `01_model_documentation.md` | Model documentation |
| `02_software_design_documentation.md` | Software design documentation |
| `03_test_plan.md` | Test plan |
| `04_test_results.md` | Test results and validation evidence |
| `advanced_demo.ipynb` | Extended demo notebook covering the M7 portfolio, manual calibration, option-vol shocks, and backtesting |
| `advanced_demo.md` | Markdown companion with front-end proof and notebook-only validation tables |
| `demo.ipynb` | Formula and workflow demonstration notebook |
| `demo.md` | Front-end workflow trace with screenshots |
| `coverage_report/` | Coverage output generated from pytest-cov |
| `test_artifacts/` | Captured test outputs and reproducibility artifacts |

## Quick start

```bash
# Install validation and notebook tooling
pip install -r requirements-dev.txt
python -m ipykernel install --user --name math5320-risk-dev \
  --display-name "Python (MATH5320 Risk)"

# Run all no-network unit tests (624 pass as of current submission state)
python -m pytest tests/ --ignore=tests/integration_test.py --ignore=tests/integration_test_formula_sheet.py

# Run with coverage reporting (about 96% statement coverage)
python -m pytest tests/ --cov=src --cov-report=term-missing \
  --ignore=tests/integration_test.py \
  --ignore=tests/integration_test_formula_sheet.py

# Execute the formula-sheet demo notebook
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=180 \
  --ExecutePreprocessor.kernel_name=python3 \
  submission/demo.ipynb

# Execute the advanced M7 demo notebook
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3 \
  submission/advanced_demo.ipynb

# Launch the Streamlit application
streamlit run app.py
```

## Environment options

### App-only environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

or

```bash
conda env create -f environment.yml
conda activate math5320-risk
streamlit run app.py
```

### Full marking environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m ipykernel install --user --name math5320-risk-dev \
  --display-name "Python (MATH5320 Risk)"
```

or

```bash
conda env create -f environment-dev.yml
conda activate math5320-risk-dev
python -m ipykernel install --user --name math5320-risk-dev \
  --display-name "Python (MATH5320 Risk)"
```

`requirements.txt` is the lean runtime install. `requirements-dev.txt` adds
pytest, coverage, and notebook execution support.
