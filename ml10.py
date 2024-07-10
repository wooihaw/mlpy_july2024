# Handling categorical data
import pandas as pd
df = pd.DataFrame({'year':[2015, 2017, 2013, 2018, 2020],
                   'make':['Toyota', 'Honda', 'Perodua', 'Hyundai', 'Toyota'],
                   'engine':[1.5, 1.8, 1.3, 1.6, 1.8],
                   'review':['moderate', 'good', 'poor', 'moderate', 'good']})
mapping = {'poor':1, 'moderate':2, 'good':3}
df['review'] = df['review'].map(mapping) # encode ordinal data
df = pd.get_dummies(df) # encode nominal data
print(df)
