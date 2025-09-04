import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Prepare the Data
df = pd.read_excel("loan_data.xlsx")
X = df[['age', 'income']]
y = df['loan_approved']

# 2. Create and Train the Model
model = DecisionTreeClassifier()
model.fit(X, y)

# 3. Save the trained model to a file
joblib.dump(model, 'model.pkl')

print("Model trained and saved to model.pkl")