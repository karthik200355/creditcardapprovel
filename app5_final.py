
from flask import Flask, render_template, request, url_for
from pymongo import MongoClient
import joblib
import numpy as np
import webbrowser
import threading
import os

app = Flask(__name__)

# Load model and encoders
model = joblib.load("Random_Forrest_Credit_Approval.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Connect to MongoDB
try:
    client = MongoClient("mongodb+srv://palakaudaykarthik:Uday%402003@cluster0.mx27dp6.mongodb.net/?retryWrites=true&w=majority")
    db = client['credit_approval_db']
    collection = db['predictions']
    print("✅ Connected to MongoDB Atlas")
except Exception as e:
    print("❌ Failed to connect to MongoDB:", e)
    collection = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        full_name = request.form.get('FullName', '')
        surname = request.form.get('Surname', '')

        raw_input = {
            "Gender": request.form.get('Gender', '').strip(),
            "Age": float(request.form.get('Age', '0') or 0),
            "Debt": float(request.form.get('Debt', '0') or 0),
            "Married": request.form.get('Married', '').strip(),
            "BankCustomer": request.form.get('BankCustomer', '').strip(),
            "EducationLevel": request.form.get('EducationLevel', '').strip(),
            "Ethnicity": request.form.get('Ethnicity', '').strip(),
            "YearsEmployed": float(request.form.get('YearsEmployed', '0') or 0),
            "PriorDefault": request.form.get('PriorDefault', '').strip(),
            "Employed": request.form.get('Employed', '').strip(),
            "CreditScore": (
        750 if float(request.form.get('Income', '0') or 0) > 80000 and float(request.form.get('YearsEmployed', '0') or 0) > 5 else
        700 if float(request.form.get('Income', '0') or 0) > 50000 and request.form.get('PriorDefault', '') == 'No' else
        650 if float(request.form.get('Debt', '0') or 0) < 1000 and request.form.get('Employed', '') == 'Yes' else
        600  # fallback default
    ),
            "DriversLicense": request.form.get('DriversLicense', '').strip(),
            "Citizen": request.form.get('Citizen', '').strip(),
            "Income": float(request.form.get('Income', '0') or 0),
        }

        # Rule-based scoring system
        score = 0
        if raw_input["CreditScore"] > 700: score += 1
        if raw_input["Income"] > 50000: score += 1
        if raw_input["YearsEmployed"] > 3: score += 1
        if raw_input["PriorDefault"] == "No": score += 1
        if raw_input["EducationLevel"] in ["Graduate", "Undergraduate"]: score += 1
        if raw_input["Debt"] < 2000: score += 1

        print(f"[DEBUG] Rule-based score: {score}/6")

        # Force decision based on score thresholds
        if score >= 4:
            predicted_label = "Approved"
        elif score <= 1:
            predicted_label = "Not Approved"
        else:
            encoded_input = []
            for key, value in raw_input.items():
                if key in label_encoders:
                    encoder = label_encoders[key]
                    if value not in encoder.classes_:
                        return f"❌ Error: Invalid value '{value}' for '{key}'. Expected: {list(encoder.classes_)}", 400
                    encoded_input.append(encoder.transform([value])[0])
                else:
                    encoded_input.append(value)

            input_array = np.array([encoded_input])
            prediction = model.predict(input_array)
            predicted_label = label_encoders["Approved_Status"].inverse_transform(prediction)[0]

        print("[DEBUG] Final prediction:", predicted_label)

        record = {
            **raw_input,
            "FullName": full_name,
            "Surname": surname,
            "Prediction": predicted_label,
            "RuleScore": score
        }
        if collection is not None:
            collection.insert_one(record)

        return render_template('result.html', prediction=predicted_label, name=full_name, score=score, estimated_credit_score=raw_input['CreditScore'])

    except Exception as e:
        return f"❌ Error: {str(e)}", 500

if __name__ == '__main__':
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000/")
    
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.5, open_browser).start()

    app.run(debug=True)
