# Freight Rate Prediction Challenge

This repository contains a regression solution for predicting freight rates.

## Files added
- `score.py` — validation script provided by the assessment.
- `predict.py` — trains the model and generates final predictions.
- `validation_predictions.csv` — final load-level predictions for validation.
- `data/december_chart_inputs.csv` — filled December predictions with `predicted_rate`.
- `data/` folder with assessment input CSVs named to match the instructions.

## Setup
Install the required packages using the included `requirements.txt`.

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run predictions

```bash
.venv\Scripts\python.exe predict.py
```

This creates:
- `validation_predictions.csv`
- completed `data/december_chart_inputs.csv`

## Validate outputs
Run the scorer with:

```bash
.venv\Scripts\python.exe score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer will verify the output files and generate:
- `scorer_results/candidate_december.png`

## Approach
- Trained a tree-based regression model on `data/train_test.csv` using a time-based holdout split.
- Features include route, equipment, distance, weight, and date-derived signals.
- The final model is trained on all available labeled development data and is used to generate both validation and December predictions.
