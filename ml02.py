from sklearn.datasets import make_blobs
# Create a synthetic dataset using make_blobs
X, y = make_blobs(n_samples=100, centers=3, 
                  n_features=2, random_state=42)
# X contains the feature matrix
# y contains the cluster labels
