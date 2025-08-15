import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Prepare the Data
# Features are Age and Income
# Labels are either 1 (loan approved) or 0 (loan denied)
data = {
    'age': [25, 35, 45, 20, 30, 50],
    'income': [40000, 60000, 80000, 30000, 50000, 90000],
    'loan_approved': [1, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

X = df[['age', 'income']]
y = df['loan_approved']

# 2. Create and Train the Model
model = DecisionTreeClassifier()
model.fit(X, y)

# 3. Save the trained model to a file
joblib.dump(model, 'model.pkl')

print("Model trained and saved to model.pkl")