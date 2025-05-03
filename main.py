# Task 1: Load and Explore the Dataset 
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('iris.data')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

df.head()
df.info()
df.describe()
print(df['species'].value_counts())

# Task 2: Basic Data Analysis 
## mean, median, standard deviation
df['sepal_length'].max() #7.9
df['sepal_width'].mean() #3.0540000000000003
df['petal_length'].median() #4.35
df['petal_width'].std() #0.7631607417008414

grouped_means = df.groupby('species').mean()
print(grouped_means)

# Task 3: Data Visualization
# 1. Line chart showing trends over time
mean_values = df.groupby('species').mean()

plt.figure(figsize=(10, 6))
for column in mean_values.columns:
    plt.plot(mean_values.index, mean_values[column], marker='o', label=column)

plt.title('Mean Feature Values per Iris Species')
plt.xlabel('Species')
plt.ylabel('Mean Value (cm)')
plt.legend(title='Feature')
plt.show()

# 2. Bar chart showing the comparison of a numerical value across categories (e.g., average petal length per species)
avgPetalLength = df.groupby('species')['petal_length'].mean()

plt.figure(figsize=(8, 5))
avgPetalLength.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram of a numerical column to understand its distribution.
plt.figure(figsize=(10, 5))
plt.hist(df['sepal_length'], bins=10, color='lightgreen') #edgecolor='black'
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot to visualize the relationship between two numerical columns (e.g., sepal length vs. petal length).
plt.figure(figsize=(8, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal_length'], subset['petal_length'], label=species)

plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()