# SVM Classification
from pandas import read_csv
from sklearn.model_selection import train_test_split as split
from sklearn.svm import SVC
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = read_csv('data/pima-indians-diabetes.data.csv', names=names)
array = df.values
X = array[:,:-1]
y = array[:,-1]
X_train, X_test, y_train, y_test = split(X, y, test_size=0.25, random_state=42)
svc = SVC().fit(X_train, y_train)
print(f'Accuracy: {100 * svc.score(X_test, y_test):.2f} %')
