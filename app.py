from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load pre-trained models
xgb_model = joblib.load("./models/xgboost_fraud_model.pkl")
rf_model = joblib.load("./models/random_forest_fraud_model.pkl")

# Features used in training
features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
            'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# -------------------------
# 1️⃣ Landing Page
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------------
# 2️⃣ About Dataset Page (Visualizations)
# -------------------------
@app.route('/about')
def about():
    # Load dataset (Update path if needed)
    df = pd.read_csv(r"D:\College\6th semester\Data Science\Project\Dataset 1\Synthetic_Financial_datasets_log.csv")

    # Convert 'type' to one-hot encoding
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # ✅ Remove non-numeric columns
    if 'nameOrig' in df.columns:
        df = df.drop(columns=['nameOrig', 'nameDest'])

    # ✅ Generate Heatmap only on numeric data
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    
    heatmap_path = os.path.join("static", "heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    return render_template('about.html', heatmap="static/heatmap.png", columns=df.columns)

# -------------------------
# 3️⃣ Check Fraud Page
# -------------------------
from hashlib import md5  # Use hashing for encoding categorical values

@app.route('/check_fraud', methods=['GET', 'POST'])
def check_fraud():
    result_xgb, result_rf = None, None  # Ensure values are always initialized
    transaction_types = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

    # Model expects these features
    expected_features = ['step', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'] + transaction_types

    if request.method == 'POST':
        try:
            # Collect numeric inputs
            user_input = [float(request.form[feature]) for feature in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
            
            # Handle checkboxes (If unchecked, set them to 0)
            for feature in transaction_types:
                user_input.append(float(request.form.get(feature, 0)))  # Default to 0 if not checked

            # ✅ Encode nameOrig & nameDest (Using Hashing)
            nameOrig = request.form.get('nameOrig', 'C000000000')  # Default if missing
            nameDest = request.form.get('nameDest', 'M000000000')  # Default if missing
            
            nameOrig_encoded = int(md5(nameOrig.encode()).hexdigest(), 16) % 10**6  # Hash to 6-digit number
            nameDest_encoded = int(md5(nameDest.encode()).hexdigest(), 16) % 10**6  # Hash to 6-digit number
            
            # Insert encoded values in correct positions
            user_input.insert(2, nameOrig_encoded)
            user_input.insert(5, nameDest_encoded)

            # Convert input into DataFrame
            input_df = pd.DataFrame([user_input], columns=expected_features)

            # Ensure all values are float (avoid dtype issues)
            input_df = input_df.astype(float)

            # Make predictions
            prediction_xgb = xgb_model.predict(input_df)[0]
            prediction_rf = rf_model.predict(input_df)[0]

            result_xgb = "🚨 FRAUD DETECTED" if prediction_xgb == 1 else "✅ SAFE TRANSACTION"
            result_rf = "🚨 FRAUD DETECTED" if prediction_rf == 1 else "✅ SAFE TRANSACTION"

        except Exception as e:
            return f"Error: {e}"

    return render_template('check_fraud.html', result_xgb=result_xgb, result_rf=result_rf, features=expected_features)

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
