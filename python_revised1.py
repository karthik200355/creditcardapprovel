from flask import Flask, render_template, url_for, request
from pymongo import MongoClient
import joblib

app = Flask(__name__)

# MongoDB Atlas connection
client = MongoClient("mongodb+srv://palakaudaykarthik:Uday%402003@cluster0.mx27dp6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client['credit_approval_db']
collection = db['predictions']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['Gender'])
        age = float(request.form['Age'])
        debt = float(request.form['Debt'])
        married = int(request.form['Married'])
        bank_customer = int(request.form['BankCustomer'])
        education = int(request.form['EducationLevel'])
        ethnicity = int(request.form['Ethnicity'])
        years_employed = float(request.form['YearsEmployed'])
        prior_default = int(request.form['PriorDefault'])
        employed = int(request.form['Employed'])
        credit_score = int(request.form['CreditScore'])
        drivers_license = int(request.form['DriversLicense'])
        citizen = int(request.form['Citizen'])
        income = float(request.form['Income'])

        # Input list for prediction
        input_data = [[gender, age, debt, married, bank_customer, education,
                       ethnicity, years_employed, prior_default, employed,
                       credit_score, drivers_license, citizen, income]]

        # Load model and make prediction
        model = joblib.load('Random_Forrest_Credit_Approval.pkl')
        my_prediction = model.predict(input_data)

        # Store record in MongoDB
        record = {
            "Gender": gender,
            "Age": age,
            "Debt": debt,
            "Married": married,
            "BankCustomer": bank_customer,
            "EducationLevel": education,
            "Ethnicity": ethnicity,
            "YearsEmployed": years_employed,
            "PriorDefault": prior_default,
            "Employed": employed,
            "CreditScore": credit_score,
            "DriversLicense": drivers_license,
            "Citizen": citizen,
            "Income": income,
            "Prediction": str(my_prediction[0])
        }

        collection.insert_one(record)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
