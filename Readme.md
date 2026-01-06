# Demand Forecasting Notebook

## Overview
This Jupyter notebook implements a multi-model time series forecasting pipeline to predict future sales for store-item combinations using ensemble learning techniques.

## Technical Approach

### Data Processing Pipeline
- **Automated Data Ingestion**: Loads train/test datasets with datetime parsing
- **Feature Engineering**: Creates `store_item` composite key for multi-series analysis
- **Temporal Sorting**: Ensures chronological ordering for time series models
- **Missing Value Handling**: Built-in validation for data completeness

### Model Architecture
- **Gradient Boosting Ensemble**: 
  - LightGBM (Microsoft)
  - CatBoost (Yandex) 
  - XGBoost (Distributed)
- **Traditional Ensemble**:
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Time Series Validation**: Implements `TimeSeriesSplit` to prevent data leakage

### Key Features
- **Multi-Granular Forecasting**: Supports store-item level predictions
- **Automated Visualization**: Seaborn/Matplotlib integration for EDA
- **Memory Optimization**: Explicit garbage collection for large datasets
- **Error Metrics**: RMSE and MAE for model evaluation

### Technical Stack
```python
Core Libraries: pandas, numpy, scikit-learn
Boosting Frameworks: lightgbm, catboost, xgboost
Visualization: matplotlib, seaborn
```

## Usage Instructions

1. **Setup**:
```bash
pip install lightgbm catboost xgboost
```

2. **Configuration**:
- Set `DATA_PATH` variable to your dataset directory
- Ensure file structure: `train.csv`, `test.csv`, `sample_submission.csv`

3. **Execution Flow**:
```
Data Loading → EDA → Feature Engineering → 
Model Training → Cross-Validation → Prediction Generation
```

## Input Requirements
- CSV format with columns: `date`, `store`, `item`, `sales`
- Chronological date sequencing
- Consistent store-item identifiers across train/test sets

## Output
- Model performance metrics (RMSE/MAE)
- Forecast predictions for test period
- Visual analytics of sales patterns

