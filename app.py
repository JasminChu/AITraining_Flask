# venv\Scripts\activate
# flask run
# http://127.0.0.1:5000

from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained model when the app starts
decision_tree_model = joblib.load("model.pkl")

# We need some data to evaluate the model
data = {
    'age': [25, 35, 45, 20, 30, 50],
    'income': [40000, 60000, 80000, 30000, 50000, 90000],
    'loan_approved': [1, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)
X_test = df[['age', 'income']]
y_test = df['loan_approved']

app = Flask(__name__)

@app.route("/")
def home():
    my_name = "Coding partner"
    return render_template("index.html", name=my_name)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/greet", methods=["GET", "POST"])
def greet_user():
    if request.method == "POST":
        user_name = request.form["name"]
        return render_template("greeting.html", name=user_name)
    return render_template("greet_form.html")

@app.route("/predict", methods=["GET", "POST"])
def make_prediction():
    if request.method == "POST":
        age = int(request.form["age"])
        income = int(request.form["income"])

        # Make individual prediction
        y_pred_single = decision_tree_model.predict([[age, income]])
        if y_pred_single[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Denied"
        
        # Generate evaluation metrics and output as a dictionary
        y_pred_all = decision_tree_model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred_all)
        class_report = classification_report(y_test, y_pred_all, output_dict=True)
        
        # --- NEW CODE: Extract accuracy as a separate variable ---
        accuracy = (class_report['accuracy'])*100
        # --- END OF NEW CODE ---
        
        return render_template("results.html", 
                               result=result, 
                               conf_matrix=conf_matrix, 
                               class_report=class_report,
                               accuracy=accuracy) # Pass the new variable
    
    return render_template("prediction_form.html")

if __name__ == "__main__":
    app.run(debug=True)