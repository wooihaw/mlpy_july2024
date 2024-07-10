import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/data_cleaning.csv')
print(df)

# Remove duplicated rows
df = df.drop_duplicates()
# Remove duplicated columns
df = df.T.drop_duplicates().T
print(df)

# Mark missing values as NaN
df = df.apply(pd.to_numeric, errors='coerce')
print(df)
print(df.info())

# Remove columns with no variance
variance = df.var()
columns_to_drop = variance[variance == 0].index
df = df.drop(columns=columns_to_drop)
print(df)

# Calculate the number of outliers for each feature
outliers = {}
for column in df.columns[:-1]:  # Excluding the last column (target)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
# Print the number of outliers for each feature
for column in outliers:
    print(f"Feature ‘{column}' has {outliers[column]} outliers")

# Clip outliers
for column in df.columns[:-1]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower_bound, upper_bound)
print(df)
