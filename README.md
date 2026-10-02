# Freight Rate Prediction — Machine Learning Engineer Assessment

## Overview

This project predicts freight spot rates from historical load-level data. The objective was to train a regression model using the provided labeled development dataset, validate the modeling approach using a time-based split, generate predictions for the final validation dataset, and produce the required December 2025 prediction chart.

## Dataset

The development dataset contains 48,000 historical loads from January through October 2025.

The final validation dataset contains 12,000 loads from November and December 2025.

The target variable is posted_rate.

Input features include pickup and delivery locations, coordinates, distance, equipment type, weight, date, market index, and quote signal.

## Data Preparation

The following data-quality steps were applied:

- Missing numerical values were handled using median imputation.
- Missing categorical values were handled using the most frequent category.
- Negative weight values were treated as invalid and converted to missing values.
- Dates were converted into calendar features: month, day of week, day of month, and day of year.
- Categorical variables were one-hot encoded.
- Unknown categorical values were handled with handle_unknown='ignore'.
- Duplicate load IDs were checked.
- Extreme target values were retained because many were associated with long-distance loads.

## Validation Strategy

A chronological split was used because the final validation data represents future months.

- Training: January–September 2025
- Validation: October 2025
- Final training: January–October 2025
- Final predictions: November–December 2025

This approach better represents the real forecasting task of predicting future freight rates from historical data.

## Model Comparison

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Ridge Regression | $170.67 | $655.66 | 0.8160 |
| Random Forest | $219.48 | $675.48 | 0.8047 |
| HistGradientBoosting | $158.40 | $661.64 | 0.8126 |
| Log-target HistGradientBoosting | $116.85 | $648.56 | 0.8200 |
| Tuned Log-target HistGradientBoosting (L2=5) | $115.45 | $647.40 | 0.8206 |

The final model uses HistGradientBoosting with a log-transformed target through TransformedTargetRegressor.

The log transformation was useful because the freight-rate target has a strongly right-skewed distribution. Hyperparameter testing found that increasing L2 regularization from 1.0 to 5.0 improved the chronological validation RMSE and R2 while also slightly improving MAE.

## Final Model

After model selection, the final model was retrained using all 48,000 development observations from January through October 2025.

Final selected configuration:

- HistGradientBoostingRegressor
- 300 iterations
- Learning rate: 0.05
- Maximum leaf nodes: 31
- L2 regularization: 5.0
- Random state: 42
- Log-transformed target using TransformedTargetRegressor

The chronological validation performance of this configuration was:

- MAE: $115.45
- RMSE: $647.40
- R2: 0.8206

The pipeline performs:

1. Numerical median imputation.
2. Categorical most-frequent imputation.
3. One-hot encoding.
4. Calendar feature engineering.
5. HistGradientBoosting regression.
6. Log-transformed target modeling.

The final model generated predictions for all 12,000 rows in the final validation dataset.

## Outputs

### Validation predictions

validation_predictions.csv contains 12,000 rows with load_id and predicted_rate.

### December predictions

data/december-chart-inputs-filled.csv contains predictions for all 31 days of December 2025.

### December chart

scorer_results/candidate_december.png was generated using the provided assessment checker.

## Reproducing the Project

Install dependencies:

pip install -r requirements.txt

Run the prediction pipeline:

python train_predict.py

Run the official checker:

python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_filled.csv --output-dir scorer_results

The checker successfully validates 12,000 final predictions and 31 December predictions.

## Final Evaluation

The final evaluation metrics for the 12,000 November–December validation rows are calculated by the assessment provider after submission.

The metrics reported in this repository are from the development validation split using January–September for training and October for validation.

## Project Files

- train_predict.py — model training and prediction pipeline.
- score.py — provided assessment validation script.
- validation_predictions.csv — final 12,000 predictions.
- data/december-chart-inputs-filled.csv — completed December predictions.
- scorer_results/candidate_december.png — December prediction chart.
- requirements.txt — required Python packages.