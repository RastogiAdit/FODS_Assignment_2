# not sure this correct once check


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Data Preprocessing
# Load the dataset
df = pd.read_csv('house_price.csv')

# Convert the 'date' column to datetime objects (if it exists)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Calculate the mean for numeric columns only and handle missing values
numeric_df = df.select_dtypes(include=np.number)
df[numeric_df.columns] = numeric_df.fillna(numeric_df.mean())

# Separate features and target variable
X = df.drop('price', axis=1)
y = df['price']

# **CHANGE:** Handle 'date' column separately
if 'date' in X.columns:
    # Extract features from 'date' (e.g., year, month, day)
    X['year'] = X['date'].dt.year
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X = X.drop('date', axis=1)  # Drop 'date' after extracting useful features

# Apply Standardization to numeric features only (excluding 'date')
X_to_scale = X.select_dtypes(include=np.number)  # Select only numeric columns for scaling

# Explicitly cast the columns to float64 to avoid type issues
X_to_scale = X_to_scale.astype(np.float64)

# Scale the numeric columns
X_scaled = X_to_scale.copy()  # Work on a copy to avoid altering the original
X_scaled[:] = (X_scaled - X_scaled.mean()) / X_scaled.std()

# Insert the scaled numeric features back into the original dataset
X[X_scaled.columns] = X_scaled

# Step 2: Perform train-test split manually (80:20 split)
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))

X_train = X.iloc[indices[:train_size]]
y_train = y.iloc[indices[:train_size]].values
X_test = X.iloc[indices[train_size:]]
y_test = y.iloc[indices[train_size:]].values

# Step 3: Model Evaluation - Implementing Mean Squared Error manually
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# A simple Linear Regression implementation using manual methods (for forward and backward selection)
def fit_linear_regression(X_train, y_train):
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept
    X_train_bias = X_train_bias.astype(np.float64)  # Ensure X_train_bias is float64
    theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
    return theta

def predict_linear_regression(X, theta):
    X_bias = np.c_[np.ones(X.shape[0]), X]  # Add intercept
    X_bias = X_bias.astype(np.float64)  # Ensure X_bias is float64
    return X_bias @ theta

# Step 4: Greedy Forward Selection
def greedy_forward_selection(X_train, y_train, X_test, y_test):
    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    best_train_error = float('inf')
    best_test_error = float('inf')

    while remaining_features:
        best_feature = None
        best_feature_train_error = float('inf')
        best_feature_test_error = float('inf')

        # Try adding each remaining feature to the model and evaluate performance
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_train_selected = X_train.iloc[:, current_features]
            X_test_selected = X_test.iloc[:, current_features]

            # Fit linear regression model
            theta = fit_linear_regression(X_train_selected, y_train)
            y_train_pred = predict_linear_regression(X_train_selected, theta)
            y_test_pred = predict_linear_regression(X_test_selected, theta)

            # Calculate errors
            train_error = mean_squared_error(y_train, y_train_pred)
            test_error = mean_squared_error(y_test, y_test_pred)

            if test_error < best_feature_test_error:
                best_feature = feature
                best_feature_train_error = train_error
                best_feature_test_error = test_error

        if best_feature is None or best_feature_test_error >= best_test_error:
            break  # No improvement found, stop the process

        # Add the best feature to the selected set
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

        best_train_error = best_feature_train_error
        best_test_error = best_feature_test_error

    return selected_features, len(selected_features), best_train_error, best_test_error

# Step 5: Greedy Backward Selection
def greedy_backward_selection(X_train, y_train, X_test, y_test):
    selected_features = list(range(X_train.shape[1]))
    best_train_error = float('inf')
    best_test_error = float('inf')

    while len(selected_features) > 1:
        worst_feature = None
        worst_feature_train_error = float('inf')
        worst_feature_test_error = float('inf')

        # Try removing each feature and evaluate performance
        for feature in selected_features:
            current_features = [f for f in selected_features if f != feature]
            X_train_selected = X_train.iloc[:, current_features]
            X_test_selected = X_test.iloc[:, current_features]

            # Fit linear regression model
            theta = fit_linear_regression(X_train_selected, y_train)
            y_train_pred = predict_linear_regression(X_train_selected, theta)
            y_test_pred = predict_linear_regression(X_test_selected, theta)

            # Calculate errors
            train_error = mean_squared_error(y_train, y_train_pred)
            test_error = mean_squared_error(y_test, y_test_pred)

            if test_error < worst_feature_test_error:
                worst_feature = feature
                worst_feature_train_error = train_error
                worst_feature_test_error = test_error

        if worst_feature is None or worst_feature_test_error >= best_test_error:
            break  # No improvement found, stop the process

        # Remove the worst feature
        selected_features.remove(worst_feature)

        best_train_error = worst_feature_train_error
        best_test_error = worst_feature_test_error

    return selected_features, len(selected_features), best_train_error, best_test_error

# Running Greedy Forward and Backward Selection
forward_selected_features, forward_feature_count, forward_train_error, forward_test_error = greedy_forward_selection(X_train, y_train, X_test, y_test)
backward_selected_features, backward_feature_count, backward_train_error, backward_test_error = greedy_backward_selection(X_train, y_train, X_test, y_test)

# Output Results
print("Forward Selection:")
print("Selected Features:", forward_selected_features)
print("Number of Features:", forward_feature_count)
print("Training Error:", forward_train_error)
print("Testing Error:", forward_test_error)

print("\nBackward Selection:")
print("Selected Features:", backward_selected_features)
print("Number of Features:", backward_feature_count)
print("Training Error:", backward_train_error)
print("Testing Error:", backward_test_error)