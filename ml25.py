# Gaussian Naive Bayes Classification
from pandas import read_csv
from sklearn.model_selection import train_test_split as split
from sklearn.naive_bayes import GaussianNB
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv('data/pima-indians-diabetes.data.csv', names=names)
X = df.values[:,:-1]
y = df.values[:,-1]
X_train, X_test, y_train, y_test = split(X, y, test_size=0.25, random_state=42)
gnb = GaussianNB().fit(X_train, y_train)
print(f'Accuracy: {gnb.score(X_test, y_test):.2%}')
