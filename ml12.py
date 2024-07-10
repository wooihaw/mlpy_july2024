# Standardize data (0 mean, 1 stdev)
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas import read_csv
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv('data/pima-indians-diabetes.data.csv', names=header)
# Separate into features and target
X = df.drop(columns=['class'])
y = df['class']
scaler = StandardScaler()
scaledX = scaler.fit_transform(X)
# Check mean and standard deviation of all columns
print(f'mean={np.mean(scaledX, axis=0)}, variance={np.var(scaledX, axis=0)}')
