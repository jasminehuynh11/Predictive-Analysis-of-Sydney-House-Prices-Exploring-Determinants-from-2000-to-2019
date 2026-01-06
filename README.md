# Predictive Analysis of Sydney House Prices

## Overview

This project analyzes and predicts residential property sale prices in Sydney from 2000 to 2019 using machine learning models. The analysis includes data exploration, preprocessing, feature engineering, and evaluation of multiple regression models.

## Dataset

The dataset contains approximately 200,000 property sale records from Sydney spanning 2000 to 2019, obtained from realestate.com.au.

**Dataset Source:** [Sydney House Prices - Kaggle](https://www.kaggle.com/datasets/mihirhalai/sydney-house-prices/data)

### Variables

- **Target Variable:** `sellPrice` - Property selling price
- **Predictor Variables:**
  - `Date` - Sale date
  - `suburb` - Property suburb
  - `postalCode` - Postal code
  - `bed` - Number of bedrooms
  - `bath` - Number of bathrooms
  - `car` - Number of car spaces
  - `propType` - Property type (house, duplex, townhouse, etc.)

## Methodology

### Data Preprocessing

1. **Missing Value Imputation:** Median imputation for missing values in `bed` and `car` columns
2. **Outlier Treatment:** Capped extreme values in `bed`, `bath`, and `car` columns at 10
3. **Data Transformation:**
   - Box-Cox transformation applied to `sellPrice` to normalize distribution
   - Ordinal encoding for categorical variables (`suburb`, `propType`)
   - Datetime conversion and year extraction from `Date` column
4. **Feature Engineering:**
   - Removed irrelevant columns (`Id`)
   - Train-test split (80-20)
   - Standard scaling applied for distance-based models

### Models Evaluated

1. **Linear Regression** - Baseline model
2. **K-Nearest Neighbors (KNN)** - With 10-fold cross-validation
3. **Decision Tree** - With GridSearchCV hyperparameter tuning
4. **Random Forest** - With RandomizedSearchCV hyperparameter tuning

### Evaluation Metric

Root Mean Squared Error (RMSE) was used as the primary evaluation metric:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (predicted_i - actual_i)^2}$$

## Results

### Model Performance

| Model | RMSE | R² Score |
|-------|------|----------|
| Linear Regression | 1.089 | 0.4745 |
| KNN | 0.891 | 0.6485 |
| KNN (10-Fold CV) | 0.884 | - |
| Decision Tree | 0.883 | 0.6543 |
| Decision Tree (Tuned) | 0.760 | 0.7442 |
| Random Forest | 0.756 | 0.7468 |
| Random Forest (Tuned) | **0.711** | **0.7757** |

### Key Findings

- **Best Model:** Tuned Random Forest achieved the lowest RMSE (0.711) and highest R² score (0.7757), explaining 77.57% of price variance
- **Feature Importance:** Location (suburb and postal code) is the most significant predictor of property prices
- **Model Comparison:** Ensemble methods (Random Forest) outperformed linear and instance-based models
- **Hyperparameter Tuning:** Significantly improved model performance for both Decision Tree and Random Forest models

## Project Structure

```
.
├── sydney_house_prices_analysis.ipynb  # Main analysis notebook
└── README.md                            # Project documentation
```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mihirhalai/sydney-house-prices/data)
2. Place `SydneyHousePrices.csv` in the project directory
3. Open and run `sydney_house_prices_analysis.ipynb`

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

## Future Work

- **Time-Series Analysis:** Implement time series models to identify seasonality, trends, and cyclical patterns
- **External Factors:** Integrate economic indicators and population growth metrics
- **Geospatial Analysis:** Use geographical coordinates for spatial price variation analysis
- **Advanced Models:** Explore Gradient Boosting and other ensemble techniques

## Summary

This project demonstrates the application of multiple machine learning models to predict Sydney house prices. The tuned Random Forest model achieved the best performance, with location, property type, and property attributes identified as key price determinants. The analysis provides insights into Sydney's housing market dynamics from 2000-2019.
