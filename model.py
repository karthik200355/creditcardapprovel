# Importing Dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

def imputeWithMode(df):
    """
    Going through each columns and checking the type is object
    if it is object, impute it with most frequent value
    """
    for col in df:
        if df[col].dtypes == 'object':
            df[col] = df[col].fillna(df[col].mode().iloc[0])

def main():
    credit = pd.read_csv('credit-approval_csv - pandas.csv')

    # Replace "?" with NaN
    credit.replace('?', np.NaN, inplace = True)

    # Convert Age to numeric
    credit["Age"] = pd.to_numeric(credit["Age"])

    credit.fillna(credit.mean(), inplace=True)


    for col in credit:
        if credit[col].dtypes == 'object':
            credit[col] = credit[col].fillna(credit[col].mode().iloc[0])

    credit_drop=credit.drop(["ZipCode"],axis=1)

    # Label Encoder
    LE = LabelEncoder()
    #Using label encoder to convert into numeric types
    for col in credit_drop:
        if credit_drop[col].dtypes=='object':
            credit_drop[col]=LE.fit_transform(credit_drop[col])

    #convert to categorical data to dummy data
    credit_dummies = pd.get_dummies(credit_drop, columns=[ "Married","EducationLevel", "Citizen", "DriversLicense", "Ethnicity"])

    credit_dummies.to_numpy

    X,y = credit_dummies.iloc[:,credit_dummies.columns != 'Approved'] , credit_dummies["Approved"]

    # Spliting the data into training and testing sets
    X_train, X_test, y_train, Y_test = train_test_split(X,
                                    y,
                                    test_size=0.2,
                                    random_state=123)

    # Scaling X_train and X_test
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX_train = scaler.fit_transform(X_train)
    rescaledX_test = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(rescaledX_train, y_train)
    y_pred = rf.predict(rescaledX_test)
    print(rf.score(rescaledX_test, Y_test))

    joblib.dump(rf, 'Random_Forrest_Credit_Approval.pkl')

if __name__ == '__main__':
    main()