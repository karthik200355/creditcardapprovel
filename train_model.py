import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Load your cleaned dataset
df = pd.read_csv("revised_credit_approval.csv")

# Drop unused columns
df = df.drop(columns=["ZipCode", "Approved"], errors='ignore')

# Encode categorical variables
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Drop missing values
df = df.dropna()

# Features and label
X = df.drop("Approved_Status", axis=1)
y = df["Approved_Status"]

# Show label encoding
print("✅ Label classes in Approved_Status:", label_encoders["Approved_Status"].classes_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest with balanced class weight
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate model
from sklearn.metrics import classification_report
print("\n✅ Model Performance:\n")
print(classification_report(y_test, model.predict(X_test)))

# Save model and encoders
joblib.dump(model, "Random_Forrest_Credit_Approval.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Model and encoders saved.")
