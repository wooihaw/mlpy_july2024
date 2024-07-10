import matplotlib.pyplot as plt
import pandas as pd
# Load the dataset
header = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('data/pima-indians-diabetes.data.csv', names=header)
# Histogram
df['age'].hist()
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plotting boxplots for the features
df.boxplot(column=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age'])
plt.title('Box plot of Mass and Age')
plt.show()

# Scatter Plot
df.plot.scatter(x='age', y='mass')
plt.title('Scatter plot of Age vs Mass')
plt.xlabel('Age')
plt.ylabel('Mass')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
plt.imshow(df.corr(), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Heatmap')
plt.show()

# Density Plot
df['mass'].plot(kind='density')
plt.title('Density Plot of Mass')
plt.xlabel('Mass')
plt.show()

# Pie Chart
df['class'].value_counts().plot(kind='pie', autopct='%.1f%%')
plt.title('Class Distribution')
plt.show()
