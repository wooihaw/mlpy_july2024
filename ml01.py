from sklearn.datasets import make_classification
# Create a synthetic dataset for classification
X, y = make_classification(n_samples=100, n_features=20, 
                           n_informative=2, n_redundant=2, 
                           n_classes=2, random_state=42)
# X contains the feature matrix
# y contains the target labels