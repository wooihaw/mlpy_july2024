import pandas as pd
import numpy as np
df = pd.DataFrame({'Age': [17, 23, 0, 38, 54, 67, 32],
                   'Height': [160, 172, 150, 165, 163, 158, 175],
                   'Weight':[50, 68, 43, 52, 47, 49, 0]})
df = df.replace({0: np.nan}) # replace missing value (0) with NaN
print(df)
print(df.isnull().sum())
df = df.dropna() # drop rows with NaN
print(df)