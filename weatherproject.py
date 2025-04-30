# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset

# Load the Iris dataset using sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column for species
df['species'] = iris.target

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check the shape of the dataset
print("\nDataset Shape:")
print(df.shape)

# Task 2: Basic Data Analysis

# Basic statistics of numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Grouping by species and calculating the mean of each numerical column
grouped_data = df.groupby('species').mean()
print("\nGrouped Data (Mean by Species):")
print(grouped_data)

# Correlation matrix to understand the relationships between numerical columns
print("\nCorrelation Matrix:")
print(df.corr())

# Task 3: Data Visualization

# 1. Line Chart: Average Petal Length by Species
df.groupby('species')['petal length (cm)'].mean().plot(kind='line')
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 2. Bar Chart: Average Sepal Length by Species
df.groupby('species')['sepal length (cm)'].mean().plot(kind='bar')
plt.title("Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Sepal Length (cm)")
plt.show()

# 3. Histogram: Distribution of Petal Length
df['petal length (cm)'].plot(kind='hist', bins=20, edgecolor='black')
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

# Optional: Customizing with Seaborn
# Pairplot to visualize relationships between all numerical columns, colored by species
sns.pairplot(df, hue='species')
plt.show()

# Error Handling (in case the dataset was from CSV)
# Try to load a CSV file if needed
# try:
#     df = pd.read_csv('your_dataset.csv')
#     print(df.head())
# except FileNotFoundError:
#     print("Dataset file not found. Please check the file path.")

