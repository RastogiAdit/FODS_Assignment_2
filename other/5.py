#changes needed, change hardcoded data to input. give the code with this prompt to gpt


import numpy as np

# Function to calculate entropy
def entropy(probabilities):
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small constant to avoid log(0)

# Function to calculate joint entropy
def joint_entropy(joint_probabilities):
    return -np.sum(joint_probabilities * np.log2(joint_probabilities + 1e-10))  # Adding a small constant

# Function to calculate mutual information
def mutual_information(X, y):
    # Entropy of the target variable
    target_probabilities = np.bincount(y) / len(y)
    H_y = entropy(target_probabilities)
    print(f"Entropy of Target (H_y): {H_y}")

    # Initialize an array to store mutual information for each feature
    I_x_y = []

    # Loop through each feature
    for i in range(X.shape[1]):
        feature = X[:, i]

        # Entropy of the feature
        feature_probabilities = np.bincount(feature) / len(feature)
        H_x = entropy(feature_probabilities)
        print(f"Entropy of Feature {i+1} (H_x): {H_x}")

        # Joint probabilities (feature, target)
        joint_probabilities = np.zeros((2, 2))
        for j in range(len(y)):
            joint_probabilities[feature[j], y[j]] += 1

        joint_probabilities /= len(y)
        print(f"Joint Probabilities (Flattened): {joint_probabilities.flatten()}")

        # Joint entropy
        H_xy = joint_entropy(joint_probabilities.flatten())
        print(f"Joint Entropy of Feature {i+1} and Target (H_xy): {H_xy}")

        # Mutual information
        I = H_x + H_y - H_xy
        print(f"Mutual Information of Feature {i+1} and Target (I_x_y): {I}")

        I_x_y.append(I)

    return I_x_y

# Function for feature selection based on mutual information
def feature_selection(X, y, k):
    # Compute mutual information for each feature
    I_x_y = mutual_information(X, y)

    # Select the indices of the top k features
    selected_features = np.argsort(I_x_y)[-k:]  # Sort in descending order and select top k
    print(f"Selected Features (1-based indexing): {selected_features + 1}")  # Output in 1-based indexing

# Example usage

# Input data (binary features and target)
X = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 1, 0],
    [1, 0, 1, 0]
])

y = np.array([1, 1, 0, 1, 0, 1])

# Number of features to select
k = 2

# Run feature selection
feature_selection(X, y, k)