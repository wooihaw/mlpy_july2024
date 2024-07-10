# Use read_csv() to load data from CSV file
from pandas import read_csv
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv('data/pima-indians-diabetes.data.csv', names=header)
print(df.head(3)) # print the first 3 rows of data
# separate data into features and target
X = df.drop(columns=['class'])
y = df['class']
print(df.shape, X.shape, y.shape) # print the dimension of the dataframe, X & y
