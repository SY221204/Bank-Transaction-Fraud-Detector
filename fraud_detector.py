#  Bank Transaction Fraud Detector - Streamlit Version

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st


# 2. Load Dataset

df = pd.read_csv("bank_fraud_dataset.csv")


# 3. Data Cleaning & Preprocessing

# Encode Categorical Columns
type_encoder = LabelEncoder()
loc_encoder = LabelEncoder()

df["Type"] = type_encoder.fit_transform(df["Type"])
df["Location"] = loc_encoder.fit_transform(df["Location"])

# Features & Target
X = df.drop(["TransactionID", "IsFraud"], axis=1)
y = df["IsFraud"]

# Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Train-Test Split & Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)


# STREAMLIT APP

st.set_page_config(page_title="Bank Fraud Detector", layout="wide")
st.title(" Bank Transaction Fraud Detector")
st.write("Detect fraudulent transactions automatically using Machine Learning.")

# Show evaluation metrics
st.subheader(" Model Evaluation")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {accuracy:.2f}")
st.text("Classification Report:\n" + classification_report(y_test, y_pred))
st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

# EDA Section
st.subheader(" Exploratory Data Analysis")

# Fraud vs Non-Fraud
fig1, ax1 = plt.subplots(figsize=(6,4))
sns.countplot(x="IsFraud", data=df, palette="Set2", ax=ax1)
ax1.set_title("Fraud vs Non-Fraud Transactions")
st.pyplot(fig1)

# Amount Distribution
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.histplot(df["Amount"], bins=30, kde=True, color="teal", ax=ax2)
ax2.set_title("Distribution of Transaction Amounts")
st.pyplot(fig2)

# Fraud by Type
fig3, ax3 = plt.subplots(figsize=(6,4))
fraud_by_type = df.groupby("Type")["IsFraud"].mean() * 100
fraud_by_type.plot(kind="bar", color="coral", ax=ax3)
ax3.set_title("Fraud Percentage by Transaction Type")
st.pyplot(fig3)

# Fraud by Location
fig4, ax4 = plt.subplots(figsize=(8,5))
fraud_by_loc = df.groupby("Location")["IsFraud"].mean() * 100
fraud_by_loc.plot(kind="bar", color="purple", ax=ax4)
ax4.set_title("Fraud Percentage by Location")
st.pyplot(fig4)


# Prediction Section
st.subheader(" Predict a New Transaction")

amount = st.number_input("Enter Transaction Amount (₹):", min_value=1.0, step=100.0)
t_type = st.selectbox("Select Transaction Type", ["ATM", "Card", "Online", "BankTransfer"])
location = st.selectbox("Select Location", ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Unknown"])

if st.button("Predict Transaction Status"):
    # Handle unseen categories
    if t_type not in type_encoder.classes_:
        t_type = "Online"
    if location not in loc_encoder.classes_:
        location = "Unknown"

    # Create dataframe for new input
    new_data = pd.DataFrame([[amount, t_type, location]], columns=["Amount", "Type", "Location"])

    # Encode
    new_data["Type"] = type_encoder.transform(new_data["Type"])
    new_data["Location"] = loc_encoder.transform(new_data["Location"])

    # Scale
    new_data_scaled = scaler.transform(new_data)

    # Predict
    prediction = model.predict(new_data_scaled)[0]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✔ Legitimate Transaction")
