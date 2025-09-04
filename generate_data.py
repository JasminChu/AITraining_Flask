import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 1000 records
num_records = 1000

# Generate random but realistic ages and incomes
ages = np.random.randint(15, 70, size=num_records) # A slightly wider range for more variety
incomes = np.random.randint(1000, 200001, size=num_records)

# Determine loan approval based on more logical rules
loan_approved = []
for i in range(num_records):
    # Rule 1: Clear "Fail" conditions
    # People under 18 or with very low income are denied.
    if ages[i] < 18 or incomes[i] < 20000:
        loan_approved.append(0)  # Denied
        continue # Move to the next person

    # Rule 2: Strong "Pass" conditions
    # People with high age and income are almost always approved.
    if ages[i] > 50 and incomes[i] > 100000:
        if np.random.rand() < 0.95: # 95% chance of approval
            loan_approved.append(1)
        else:
            loan_approved.append(0)
        continue

    # Rule 3: The "In-Between" cases
    # For everyone else, use a probability based on their age and income.
    probability = (ages[i] / 70) * 0.4 + (incomes[i] / 200000) * 0.6
    if np.random.rand() < probability:
        loan_approved.append(1) # Approved
    else:
        loan_approved.append(0) # Denied

# Create a DataFrame from the generated data
data = {
    'age': ages,
    'income': incomes,
    'loan_approved': loan_approved
}
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
df.to_excel("loan_data.xlsx", index=False)

print("Successfully generated loan_data.xlsx with 1000 records.")