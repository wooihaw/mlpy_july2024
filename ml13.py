# Robust scaling (0 median, 1 IQR)
import numpy as np
from sklearn.preprocessing import RobustScaler
from pandas import read_csv
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv('data/pima-indians-diabetes.data.csv', names=header)
# Separate into features and target
X = df.drop(columns=['class'])
y = df['class']
scaler = RobustScaler()
scaledX = scaler.fit_transform(X)
# Check median and IQR of all columns
q3, q1 = np.percentile(scaledX, [75 ,25], axis=0)
print(f'median={np.median(scaledX, axis=0)}, IQR={q3-q1}')