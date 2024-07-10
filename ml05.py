# Print statistical summary and class breakdown
from pandas import read_csv
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv('data/pima-indians-diabetes.data.csv', names=header)
print(df.describe())  # print the statistical summary of the data
class_counts = df.groupby('class').size()
print(class_counts)  # print the class breakdown of the data