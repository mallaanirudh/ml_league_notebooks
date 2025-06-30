Blueberry Yield Prediction - Regression Challenge
This project aims to predict blueberry yield using features like fruitset, seeds, fruitmass, and other environmental/pollinator indicators. The goal was to minimize Mean Absolute Error (MAE) on a hidden Kaggle test set.

 Dataset Overview
train.csv: 15,000 samples with 18 features + target yield

test.csv: 10,000 samples, no target

sample_submission.csv: Format reference for predictions

 1. Exploratory Data Analysis (EDA)
 Checked data types and distributions

 Confirmed no missing values

 Calculated correlations with yield

 Identified top correlated features:

fruitset (+0.95)

seeds (+0.92)

fruitmass (+0.89)

 Dropped low/negatively correlated features like:

clonesize, Row#, RainingDays, AverageRainingDays
 2. Feature Engineering
 Created manual interaction features:

fruitset_x_seeds = fruitset * seeds

fruitmass_x_seeds = fruitmass * seeds

combo_feature = fruitset * fruitmass * seeds

fruit_density = fruitmass / (seeds + 1e-6)

 Later reverted to simpler features due to overfitting risk

 3. Preprocessing
Handled outliers by capping/extreme value control

 Standardized features for models (when needed)

 Feature selection via:

Correlation threshold

Permutation importance

 4. Model Development
 Models Tried:
LinearRegression (baseline)

HistGradientBoostingRegressor (best performer)

XGBoostRegressor

RandomForestRegressor

StackingRegressor (ensemble of HGB + XGB + RF)

 Tuning & Validation:
Used Optuna for hyperparameter tuning:

learning_rate, max_iter, max_leaf_nodes, l2_regularization

Cross-validation:

 Switched from 80/20 to 5-fold CV for better generalization

 5. Model Evaluation
Tracked:

MAE on validation set (CV)

Leaderboard MAE (on Kaggle test set)

Identified overfitting when:

CV MAE: ~246

Kaggle MAE: ~258

 6. Generalization Strategy
Switched to minimal, robust features:

fruitset, seeds, fruitmass

 Avoided over-feature-engineering

 Removed log-transform when it caused leaderboard MAE drift
 Tried both HistGradientBoosting and XGBoost with clean input

