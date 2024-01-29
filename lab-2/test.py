import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming df is your DataFrame and it has a column 'salary'
# Sample data for demonstration
data = {'salary': [50000, 60000, 55000, 48000, 70000, 65000, 52000, 58000, 63000, 51000]}
df = pd.DataFrame(data)

# Calculate mean, standard deviation, and variance
mean_salary = df['salary'].mean()
std_dev_salary = df['salary'].std()
variance_salary = df['salary'].var()

# Create histogram
plt.hist(df['salary'], bins=10, alpha=0.7, color='blue', edgecolor='black')

# Add a line for the mean
plt.axvline(mean_salary, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_salary:.2f}')

# Add lines for standard deviation
plt.axvline(mean_salary - std_dev_salary, color='green', linestyle='dashed', linewidth=2, label=f'Standard Deviation: {std_dev_salary:.2f}')
plt.axvline(mean_salary + std_dev_salary, color='green', linestyle='dashed', linewidth=2)

# Add title and labels
plt.title('Salary Distribution with Mean and Standard Deviation')
plt.xlabel('Salary')
plt.ylabel('Frequency')

# Add legend
plt.legend()

# Show plot
plt.show()
