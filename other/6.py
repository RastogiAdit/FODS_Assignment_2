import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = 'german_credit_data.csv'
data = pd.read_csv('german_credit_data.csv')

# Step 1: Data Cleaning 
# Drop irrelevant columns
data_cleaned = data.drop(columns=["Unnamed: 0"])

# Handle missing values (fill with mode)
data_cleaned["Saving accounts"].fillna(data_cleaned["Saving accounts"].mode()[0], inplace=True)
data_cleaned["Checking account"].fillna(data_cleaned["Checking account"].mode()[0], inplace=True)

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data_cleaned, drop_first=True)

# Step 2: Define numerical features for boxplots and outlier detection
numerical_features = ["Age", "Job", "Credit amount", "Duration"]

# Step 3: Boxplots Before Outlier Removal
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    plt.boxplot(data_encoded[feature], vert=False)
    plt.title(f"Boxplot of {feature}")
    plt.xlabel(feature)
plt.tight_layout()
plt.show()

# Step 4: Outlier Detection and Removal Using IQR Method
def remove_outliers_iqr(df, features):
    cleaned_df = df.copy()
    for feature in features:
        Q1 = cleaned_df[feature].quantile(0.25)  # First quartile
        Q3 = cleaned_df[feature].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1                            # Interquartile Range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Removing outliers
        cleaned_df = cleaned_df[(cleaned_df[feature] >= lower_bound) & (cleaned_df[feature] <= upper_bound)]
    return cleaned_df

# Remove outliers from numerical features
data_no_outliers = remove_outliers_iqr(data_encoded, numerical_features)

# Display shape of dataset before and after outlier removal
print("Original Dataset Shape:", data_encoded.shape)
print("Dataset Shape After Outlier Removal:", data_no_outliers.shape)

# Step 5: Boxplots After Outlier Removal
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    plt.boxplot(data_no_outliers[feature], vert=False)
    plt.title(f"Boxplot of {feature} After Outlier Removal")
    plt.xlabel(feature)
plt.tight_layout()
plt.show()

# Step 6: Summary of Cleaned Data
print("Summary of Cleaned Data (After Outlier Removal):")
print(data_no_outliers.describe())