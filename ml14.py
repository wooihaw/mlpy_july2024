# Create 2 new features
import pandas as pd
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('data/pima-indians-diabetes.data.csv', names=header)
bins = [0, 30, 50, 100] 
labels = ['Young', 'Middle-aged', 'Senior']
df['new_feature1'] = pd.cut(df['age'], bins=bins, labels=labels)
df['new_feature2'] = df['mass'].rolling(window=3).mean()
print(df.head())
