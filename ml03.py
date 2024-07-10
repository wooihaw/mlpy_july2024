# Import the necessary module from scikit-learn
from sklearn.datasets import load_iris
# Load the Iris dataset
dataset = load_iris()
# Extract features and target variables
X = dataset.data
y = dataset.target
# Display feature names and target names
print("Feature Names:", dataset.feature_names)
print("Target Names:", dataset.target_names)