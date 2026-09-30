# Project 4 — Agriculture Crop Production Prediction in India

## Overview
This project develops a regression pipeline for predicting **crop yield (Quintal/Hectare)** from crop, state and cultivation/production cost information. The supplied project documentation describes agriculture production in India and identifies crop, variety, state, production, season, cost and recommended-zone information as relevant concepts.

## Internship plan coverage
- Week 1: Python/Pandas/NumPy setup, EDA, missing-value checks and cleaned datasets.
- Week 2: Linear Regression, Decision Tree, Random Forest and XGBoost; comparison with RMSE and R².
- Week 3: Streamlit prediction interface, yield visualization and feature-importance output.
- Week 4: Documentation, reproducible repository and deployment instructions.

## Dataset note
The supplied ZIP contains multiple CSVs. For the predictive model, `datafile (1).csv` is used because it contains crop, state, cultivation costs, production cost and **Yield (Quintal/Hectare)**. The other CSVs are retained as source/reference data.

## Results
The generated model comparison is in `results/model_comparison.csv`. Visual outputs are in `results/`.

## Run
```bash
pip install -r requirements.txt
python src/train_models.py
streamlit run app.py
```

## Limitations
The supplied crop dataset is small (49 rows), so the reported test metrics should be treated as an academic demonstration rather than a production forecasting benchmark.
