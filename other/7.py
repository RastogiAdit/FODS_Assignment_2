import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Data Import and Cleaning
file_path = 'gym_membership.csv'
data = pd.read_csv(file_path)

# Drop irrelevant columns
irrelevant_columns = ['id', 'birthday', 'days_per_week', 'fav_group_lesson',
                      'avg_time_check_in', 'avg_time_check_out', 'name_personal_trainer']
data_cleaned = data.drop(columns=irrelevant_columns, errors='ignore')

# Handle missing values
for column in data_cleaned.columns:
    if data_cleaned[column].dtype == 'object':  # Categorical column
        data_cleaned[column].fillna(data_cleaned[column].mode()[0], inplace=True)
    else:  # Numeric column
        data_cleaned[column].fillna(data_cleaned[column].mean(), inplace=True)

# Convert boolean columns to numeric
boolean_columns = ['attend_group_lesson', 'drink_abo', 'personal_training', 'uses_sauna']
for column in boolean_columns:
    if column in data_cleaned.columns:
        data_cleaned[column] = data_cleaned[column].astype(int)

# Encode categorical variables
categorical_columns = ['gender', 'abonoment_type', 'fav_drink']
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)

# Step 2: Data Scaling
scaler = StandardScaler()
numeric_columns = data_encoded.select_dtypes(include=['number']).columns
data_scaled = scaler.fit_transform(data_encoded[numeric_columns])

# Step 3: PCA Application
pca = PCA()
pca_data = pca.fit_transform(data_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Scree Plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual Variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative Variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot and Cumulative Variance')
plt.legend()
plt.show()

# Find components covering 80-90% of the variance
num_components = np.argmax(cumulative_variance >= 0.9) + 1
print(f"Number of components to cover 90% variance: {num_components}")

# Reduce dimensionality to 2 components
pca_2 = PCA(n_components=2)
pca_data_2d = pca_2.fit_transform(data_scaled)

# Step 4: Visualization of PCA Results
# Step 4: Visualization of PCA Results
# 2D Scatter Plot
abonement_types = data['abonoment_type'].unique()
abonement_colors = {atype: i for i, atype in enumerate(abonement_types)}
colors = [abonement_colors[atype] for atype in data['abonoment_type']]

plt.figure(figsize=(10, 7))
plt.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], c=colors, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Scatter Plot of PCA Results')
plt.colorbar(label='Abonement Type')
plt.show()

# Loadings Plot (Feature Contributions to Components)
loadings = pca_2.components_.T  # Loadings for the first two components
plt.figure(figsize=(10, 7))
plt.scatter(pca_data_2d[:, 0], pca_data_2d[:, 1], alpha=0.5, label='Data Points')

# Plot the loadings of the first two principal components
for i, feature in enumerate(data_encoded.columns):
    if i < len(loadings):  # Avoid accessing out-of-bounds indices
        plt.arrow(0, 0, loadings[i, 0] * 5, loadings[i, 1] * 5, color='red', alpha=0.7, head_width=0.1)
        plt.text(loadings[i, 0] * 5.5, loadings[i, 1] * 5.5, feature, color='black')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Loadings Plot')
plt.legend(['Data Points', 'Feature Vectors'])
plt.grid()
plt.show()