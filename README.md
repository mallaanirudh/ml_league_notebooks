# Blueberry Yield Prediction - Regression Challenge

This project aims to predict blueberry yield using features like fruitset, seeds, fruitmass, and other environmental/pollinator indicators. The goal was to minimize Mean Absolute Error (MAE) on a hidden Kaggle test set.

## Dataset Overview

- **train.csv**: 15,000 samples with 18 features + target yield
- **test.csv**: 10,000 samples, no target
- **sample_submission.csv**: Format reference for predictions

## 1. Exploratory Data Analysis (EDA)

### Data Quality Assessment
- Checked data types and distributions
- Confirmed no missing values
- Calculated correlations with yield

### Key Feature Correlations
Identified top correlated features with yield:
- **fruitset**: +0.95
- **seeds**: +0.92
- **fruitmass**: +0.89
- **osmia**: +0.27

### Feature Exclusions
Dropped low/negatively correlated features:
- clonesize (-0.41)
- Row# (-0.01)
- RainingDays (-0.51)
- AverageRainingDays (-0.52)

## 2. Feature Engineering

### Initial Approach
Created manual interaction features:
- `fruitset_x_seeds = fruitset * seeds`
- `fruitmass_x_seeds = fruitmass * seeds`
- `combo_feature = fruitset * fruitmass * seeds`
- `fruit_density = fruitmass / (seeds + 1e-6)`

### Strategy Revision
Later reverted to simpler features due to overfitting risk identified through validation-leaderboard gap analysis.

## 3. Preprocessing

### Data Cleaning
- Handled outliers by capping extreme values
- Standardized features for models when needed

### Feature Selection Methods
- Correlation threshold filtering
- Permutation importance analysis
- Cross-validation performance tracking

## 4. Model Development

### Models Evaluated
1. **LinearRegression** (baseline)
2. **HistGradientBoostingRegressor** (best performer)
3. **XGBoostRegressor**
4. **RandomForestRegressor**
5. **StackingRegressor** (ensemble of HGB + XGB + RF)

### Hyperparameter Optimization
- Used **Optuna** for automated tuning
- Key parameters optimized:
  - learning_rate
  - max_iter
  - max_leaf_nodes
  - l2_regularization

### Validation Strategy
- Initial approach: 80/20 train-validation split
- Final approach: 5-fold cross-validation for better generalization

## 5. Model Evaluation

### Performance Tracking
Monitored two key metrics:
- **CV MAE**: Cross-validation performance on training data
- **Kaggle MAE**: Leaderboard performance on hidden test set

### Overfitting Detection
Identified significant overfitting when:
- CV MAE: ~246
- Kaggle MAE: ~258
- Gap of 12 points indicated poor generalization

## 6. Generalization Strategy

### Feature Simplification
Switched to minimal, robust feature set:
- fruitset
- seeds
- fruitmass

### Model Simplification
- Avoided over-feature-engineering
- Removed log-transform when it caused leaderboard MAE drift
- Applied stronger regularization techniques

### Final Model Configuration
Implemented ultra-conservative hyperparameters:
- Shallow trees (max_depth=2-4)
- High regularization (l2_regularization=5.0-10.0)
- Large minimum leaf sizes
- Reduced learning rates


