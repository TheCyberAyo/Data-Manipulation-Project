import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading: Load the dataset into a pandas DataFrame
file_path = 'output.csv'
try:
    data = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}.")
except FileNotFoundError:
    print(f"File {file_path} not found.")
    exit()

# Data Loading: Display basic information about the dataset
print(f"Shape: {data.shape}\nData types:\n{data.dtypes}\nSummary:\n{data.describe()}")

# Data Cleaning: Identify and handle missing data appropriately
cleaned_data = data.dropna(subset=['price', 'km'])
print(f"Rows dropped: {len(data) - len(cleaned_data)}")

for col in ['engine', 'mileage', 'power', 'seats']:
    if col in cleaned_data.columns:
        cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)

# Data Cleaning: Remove or address any duplicate data
initial_count = len(cleaned_data)
cleaned_data.drop_duplicates(inplace=True)
print(f"Duplicates removed: {initial_count - len(cleaned_data)}")

# Data Exploration: Use descriptive statistics and visualize data distributions
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(cleaned_data['price'], bins=50, color='blue', kde=True)
plt.title('Price Distribution')
plt.subplot(1, 2, 2)
sns.histplot(cleaned_data['km'], bins=50, color='green', kde=True)
plt.title('Kilometers Driven Distribution')
plt.tight_layout()
plt.show()

# Data Manipulation
cleaned_data['price_per_km'] = cleaned_data['price'] / cleaned_data['km']

# Data Analysis: Summary by seats, using pivot-like grouping and aggregation
seats_summary = cleaned_data.groupby('seats').agg(
    average_price=('price', 'mean'), average_km=('km', 'mean'), count=('price', 'size')).reset_index()
print(seats_summary)

# Data Visualization: Visualization of average price by seats
plt.figure(figsize=(10, 5))
sns.barplot(x='seats', y='average_price', data=seats_summary, palette='viridis')
plt.title('Average Price by Number of Seats')
plt.tight_layout()
plt.show()

# Data Exploration and Visualization
plt.figure(figsize=(8, 8))
fuel_distribution = cleaned_data['fuel'].value_counts()
plt.pie(fuel_distribution, labels=fuel_distribution.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
plt.title('Car Distribution by Fuel Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='km', y='price', hue='fuel', data=cleaned_data, palette='coolwarm')
plt.title('Price vs Kilometers Driven by Fuel Type')
plt.show()

# Data Analysis: Summary by transmission type, using grouping and aggregation
transmission_summary = cleaned_data.groupby('transmission').agg(average_price=('price', 'mean'), count=('price', 'size')).reset_index()
plt.figure(figsize=(8, 6))
sns.barplot(x='transmission', y='average_price', data=transmission_summary, palette='viridis')
plt.title('Average Price by Transmission Type')
plt.show()

# Data Analysis: Brand summary
print(cleaned_data.groupby('brand').agg(average_price=('price', 'mean'), average_km=('km', 'mean'), count=('price', 'size')).head(10))

print("Analysis complete.")