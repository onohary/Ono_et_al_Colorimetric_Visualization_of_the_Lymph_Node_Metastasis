# -*- coding: utf-8 -*-
"""
@author: Haruka Ono
"""

# Import necessary libraries
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings('ignore')


# ====== Load dataset ========================================================================
Pdot_total_data = pd.read_csv('data/Accumulated_Pdot_amount.csv')
dataset = Pdot_total_data[Pdot_total_data["Day"].isin([1, 5, 9, 12])].reset_index(drop=True)



# ====== Preprocess data ========================================================================
y = dataset.iloc[:, 0].copy()
x_original = dataset.iloc[:, 1:].copy()
print('Sample number : {0}'.format(x_original.shape[0]))



# ====== Cross-validation settings ========================================================================
methods = ['lr', 'ridge', 'knn', 'rf', 'svr'] # regressors
outer_fold_number = 5 # Fold count for outer cross-validation
inner_fold_number = 5 # Fold count for inner cross-validation



# ====== Hyperparameter settings ========================================================================
# Ridge: Set of lambda (regularization strength) values to search (2^-10 to 2^10)
ridge_lambdas = 2 ** np.arange(-10, 11, dtype=float)

# kNN: Set of k (number of neighbors) values to search (1 to 17)
ks_in_knn = np.arange(1, 18, 1)

# RF: Fixed number of trees/estimators
random_forest_number_of_trees = 100

# RF: Set of max_features ratios to search (0.1 to 1.0)
random_forest_x_variables_rates = np.arange(1, 11, dtype=float) / 10

# SVM: Set of C (regularization) values to search (2^-10 to 2^10)
nonlinear_svm_cs = 2 ** np.arange(-10, 11, dtype=float)

# SVM: Set of gamma (kernel coefficient) values to search (2^-20 to 2^10)
nonlinear_svm_gammas = 2 ** np.arange(-20, 11, dtype=float)



# ====== Outer cross-validation ========================================================================
outer_cv = StratifiedKFold(n_splits=outer_fold_number, shuffle=True, random_state=999)



# ====== Model training and evaluation ========================================================================
for method in methods:
    results = []  # List to store results from each outer fold
    for fold_number_in_outer_cv, (train_idx, test_idx) in enumerate(outer_cv.split(x_original, y), 1):
        # Split into training and test data for outer cross-validation
        x_train = x_original.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        x_test = x_original.iloc[test_idx, :]
        y_test = y.iloc[test_idx]

        # Autocaling
        autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
        autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

        # ====== Inner cross-validation for hyperparameter tuning ======
        inner_cv = StratifiedKFold(n_splits=inner_fold_number, shuffle=True, random_state=999) # Inner CV setup (StratifiedKFold maintains day ratios)

        # lr
        if method == 'lr':
            # Final model
            model = LinearRegression()

        # ridge
        elif method == 'ridge':
            # Hyperparameter tuning using R^2 in inner CV
            ridge_in_cv = GridSearchCV(Ridge(), {'alpha': ridge_lambdas}, cv=inner_cv)
            ridge_in_cv.fit(autoscaled_x_train, y_train)

            # Final model
            model = Ridge(alpha=ridge_in_cv.best_params_['alpha'])

        # knn
        elif method == 'knn':
            # k should not exceed the number of training samples in inner CV
            n_samples_inner_train = int(len(y_train) * (inner_fold_number - 1) / inner_fold_number)
            valid_ks = [k for k in ks_in_knn if k <= n_samples_inner_train]
            if len(valid_ks) == 0:
                valid_ks = [1]

            # Hyperparameter tuning using MSE in inner CV
            knn_in_cv = GridSearchCV(KNeighborsRegressor(),
                                    {'n_neighbors': valid_ks},
                                    cv=inner_cv)
            knn_in_cv.fit(autoscaled_x_train, y_train)

            # Final model
            model = KNeighborsRegressor(n_neighbors=knn_in_cv.best_params_['n_neighbors'])

        # rf
        elif method == 'rf':
            max_features_candidates = [int(max(np.ceil(x_train.shape[1] * r), 1))
                                    for r in random_forest_x_variables_rates]

            # Hyperparameter tuning using OOB scores
            oob_scores = []
            for mf in max_features_candidates:
                rf_model = RandomForestRegressor(
                    n_estimators=random_forest_number_of_trees,
                    max_features=mf,
                    oob_score=True,
                    random_state=0,
                    n_jobs=-1
                )
                rf_model.fit(autoscaled_x_train, y_train)
                oob_scores.append(rf_model.oob_score_)

            # Select the max_features with the highest OOB score
            best_mf = max_features_candidates[np.argmax(oob_scores)]

            # Final model
            model = RandomForestRegressor(
                n_estimators=random_forest_number_of_trees,
                max_features=best_mf,
                oob_score=True,
                random_state=0,
                n_jobs=-1
            )

        # svr
        elif method == 'svr':
            # Hyperparameter tuning using R^2 in inner CV
            svr_in_cv = GridSearchCV(SVR(),
                                        {'C': nonlinear_svm_cs,
                                        'gamma': nonlinear_svm_gammas},
                                        cv=inner_cv)
            svr_in_cv.fit(autoscaled_x_train, y_train)

            # Final model
            model = SVR(C=svr_in_cv.best_params_['C'],
                        gamma=svr_in_cv.best_params_['gamma'])

        # Train the final model on the entire training set
        model.fit(autoscaled_x_train, y_train)
        # Predict on the test set
        y_pred = model.predict(autoscaled_x_test)

        # Store predictions
        # Collect results for all outer folds
        df_out = pd.DataFrame({
            'outer_fold_number': fold_number_in_outer_cv, # Outer fold number
            'Acctual_days_post_inoculation': y.iloc[test_idx].values, # Actual days post inoculation
            'Predicted_days_post_inoculation': y_pred, # Predicted days post inoculation
            'F8BT': x_original.iloc[test_idx, 0].values, # Original feature values
            'MEHPPV': x_original.iloc[test_idx, 1].values,  # Original feature values
            'TPP_PFO': x_original.iloc[test_idx, 2].values # Original feature values
        })

        # Append results of the current outer fold
        results.append(df_out)

    # Combine results from all outer folds and save to CSV
    result_df = pd.concat(results, ignore_index=True)
    result_df.to_csv(f'machine_learning/results/prediction_{method}.csv', index=False)



# ====== Calculate R^2 values and RMSE ========================================================================
r2_results = {} # Dictionary to store R^2 values for each method
rmse_results = {} # Dictionary to store RMSE values for each method

for method in methods:
    # Read prediction results
    df = pd.read_csv(f'machine_learning/results/prediction_{method}.csv')

    # Calculate R^2 and RMSE for each outer fold
    r2_list = [] # List to store R^2 values for each fold
    rmse_list = [] # List to store RMSE values for each fold

    for fold in sorted(df['outer_fold_number'].unique()):
        fold_df = df[df['outer_fold_number'] == fold] # Data for the current fold
        y_true = fold_df['Acctual_days_post_inoculation'] # True values
        y_pred = fold_df['Predicted_days_post_inoculation'] # Predicted values
        r2 = r2_score(y_true, y_pred) # R^2 calculation
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # RMSE calculation
        r2_list.append(r2) # Append R^2 to the list
        rmse_list.append(rmse) # Append RMSE to the list
    
    # Store results in the dictionaries
    r2_results[method] = r2_list 
    rmse_results[method] = rmse_list

# Create DataFrames for R^2 and RMSE results
folds = sorted(df['outer_fold_number'].unique())
r2_df = pd.DataFrame(r2_results)
r2_df.insert(0, 'Outer fold', folds)
rmse_df = pd.DataFrame(rmse_results)
rmse_df.insert(0, 'Outer fold', folds)

# Save R^2 and RMSE results to CSV files
r2_df.to_csv('results/r2_values.csv', index=False)
rmse_df.to_csv('results/rmse_values.csv', index=False)

