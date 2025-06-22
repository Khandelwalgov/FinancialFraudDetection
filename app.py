from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

xgb_model = joblib.load("./models/xgboost_fraud_model.pkl")
rf_model = joblib.load("./models/random_forest_fraud_model.pkl")

features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
            'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    # df = pd.read_csv(r"D:\College\6th semester\Data Science\Project\Dataset 1\Synthetic_Financial_datasets_log.csv")

    # df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # if 'nameOrig' in df.columns:
    #     df = df.drop(columns=['nameOrig', 'nameDest'])
    # print(df.columns)
    # print(df.corr())
    column_names = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
        'newbalanceDest', 'isFraud', 'isFlaggedFraud', 'type_CASH_OUT',
        'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]

    correlation_matrix = [
        [1.000, 0.022, -0.010, -0.010, 0.028, 0.026, 0.032, 0.003, -0.013, 0.003, 0.005, 0.007],
        [0.022, 1.000, -0.003, -0.008, 0.294, 0.459, 0.077, 0.012, -0.004, -0.023, -0.197, 0.366],
        [-0.010, -0.003, 1.000, 0.999, 0.066, 0.042, 0.010, 0.004, -0.201, -0.021, -0.189, -0.082],
        [-0.010, -0.008, 0.999, 1.000, 0.068, 0.042, -0.008, 0.004, -0.211, -0.022, -0.194, -0.087],
        [0.028, 0.294, 0.066, 0.068, 1.000, 0.130, 0.027, 0.005, 0.086, 0.009, -0.231, 0.130],
        [0.026, 0.459, 0.042, 0.042, 0.130, 1.000, 0.026, 0.004, 0.093, 0.006, -0.238, 0.192],
        [0.032, 0.077, 0.010, -0.008, 0.027, 0.026, 1.000, 0.003, 0.011, -0.003, -0.026, 0.054],
        [0.003, 0.012, 0.004, 0.004, 0.005, 0.004, 0.003, 1.000, -0.001, -0.000, -0.001, 0.005],
        [-0.013, -0.004, -0.201, -0.211, 0.086, 0.093, 0.011, -0.001, 1.000, -0.060, -0.526, -0.223],
        [0.003, -0.023, -0.021, -0.022, 0.009, 0.006, -0.003, -0.000, -0.060, 1.000, -0.058, -0.024],
        [0.005, -0.197, -0.189, -0.194, -0.231, -0.238, -0.026, -0.001, -0.526, -0.058, 1.000, -0.216],
        [0.007, 0.366, -0.082, -0.087, 0.130, 0.192, 0.054, 0.005, -0.223, -0.024, -0.216, 1.000]
    ]
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    
    heatmap_path = os.path.join("static", "heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()

    return render_template('about.html', heatmap="static/heatmap.png", columns=column_names)


from hashlib import md5  

@app.route('/check_fraud', methods=['GET', 'POST'])
def check_fraud():
    result_xgb, result_rf = None, None  # Ensure values are always initialized
    transaction_types = ['type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

    expected_features = ['step', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'] + transaction_types

    if request.method == 'POST':
        try:
            user_input = [float(request.form[feature]) for feature in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
            
            for feature in transaction_types:
                user_input.append(float(request.form.get(feature, 0)))  # Default to 0 if not checked

            nameOrig = request.form.get('nameOrig', 'C000000000')  # Default if missing
            nameDest = request.form.get('nameDest', 'M000000000')  # Default if missing
            
            nameOrig_encoded = int(md5(nameOrig.encode()).hexdigest(), 16) % 10**6  # Hash to 6-digit number
            nameDest_encoded = int(md5(nameDest.encode()).hexdigest(), 16) % 10**6  # Hash to 6-digit number
            
            user_input.insert(2, nameOrig_encoded)
            user_input.insert(5, nameDest_encoded)

            input_df = pd.DataFrame([user_input], columns=expected_features)

            input_df = input_df.astype(float)

            prediction_xgb = xgb_model.predict(input_df)[0]
            prediction_rf = rf_model.predict(input_df)[0]

            result_xgb = "🚨 FRAUD DETECTED" if prediction_xgb == 1 else "✅ SAFE TRANSACTION"
            result_rf = "🚨 FRAUD DETECTED" if prediction_rf == 1 else "✅ SAFE TRANSACTION"

        except Exception as e:
            return f"Error: {e}"

    return render_template('check_fraud.html', result_xgb=result_xgb, result_rf=result_rf, features=expected_features)

if __name__ == '__main__':
    app.run(debug=True)
