import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'output.csv'  # Change this to your dataset path
try:
    data = pd.read_csv(file_path)
    print(f"Data loaded successfully from {file_path}.")
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()

# Display basic information about the dataset
print(f"Dataset shape: {data.shape}")
print("Data types:")
print(data.dtypes)
print("Summary statistics:")
print(data.describe())

# Data Cleaning
# Dropping rows where price or km are missing, as they are essential for analysis
cleaned_data = data.dropna(subset=['price', 'km'])
print(f"Data cleaned: {len(data) - len(cleaned_data)} rows dropped.")

# Filling missing values for other columns using mean
for column in ['engine', 'mileage', 'power', 'seats']:
    if column in cleaned_data.columns:
        mean_value = cleaned_data[column].mean()
        cleaned_data[column] = cleaned_data[column].fillna(mean_value)
        print(f"Missing values in '{column}' filled with mean: {mean_value:.2f}")

# Removing duplicates if any
initial_count = len(cleaned_data)
cleaned_data.drop_duplicates(inplace=True)
print(f"Duplicates removed: {initial_count - len(cleaned_data)} rows dropped.")

# Data Exploration
# Visualize the cleaned data distributions
plt.figure(figsize=(12, 6))

# Subplot 1: Price Distribution
plt.subplot(1, 2, 1)
sns.histplot(cleaned_data['price'], bins=50, color='blue', kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Subplot 2: Kilometers Driven Distribution
plt.subplot(1, 2, 2)
sns.histplot(cleaned_data['km'], bins=50, color='green', kde=True)
plt.title('Distribution of Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Frequency')

plt.tight_layout()

# Show the plot live
plt.show()

# Data Manipulation: Create a new column for price per kilometer
cleaned_data['price_per_km'] = cleaned_data['price'] / cleaned_data['km']
print("New column 'price_per_km' created.")

# Data Analysis: Summary statistics by number of seats
seats_summary = cleaned_data.groupby('seats').agg(
    average_price=('price', 'mean'),
    average_km=('km', 'mean'),
    count=('price', 'size')
).reset_index()
print("Summary statistics by number of seats:")
print(seats_summary)

# Visualization of average price by number of seats
plt.figure(figsize=(10, 5))
sns.barplot(x='seats', y='average_price', data=seats_summary, palette='viridis')
plt.title('Average Price by Number of Seats')
plt.xlabel('Number of Seats')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()

# Show the bar plot live
plt.show()

# Final Report
print("Data analysis and visualization completed.")