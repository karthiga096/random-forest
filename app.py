import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="Student Grade Prediction")

st.title("🎓 Student Grade Prediction using Random Forest")

# Try loading CSV automatically
file_path = "StudentMarksDataset.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    uploaded_file = st.file_uploader("Upload StudentMarksDataset.csv", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()   # 👈 stops silently (NO MESSAGE)

# Grade function
def grade_class(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    elif marks >= 60:
        return "D"
    elif marks >= 50:
        return "E"
    else:
        return "F"

df["Grade"] = df["Std_Marks"].apply(grade_class)

# Encode categorical columns
le_branch = LabelEncoder()
le_course = LabelEncoder()

df["Std_Branch_enc"] = le_branch.fit_transform(df["Std_Branch"])
df["Std_Course_enc"] = le_course.fit_transform(df["Std_Course"])

# Features & target
X = df[["Std_Branch_enc", "Std_Course_enc", "Std_Marks"]]
y = df["Grade"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    random_state=42
)
model.fit(X, y)

# User input UI
st.subheader("Enter Student Details")

branch = st.selectbox("Branch", df["Std_Branch"].unique())
course = st.selectbox("Course", df["Std_Course"].unique())
marks = st.slider("Marks", 0, 100, 75)

if st.button("Predict Grade"):
    b = le_branch.transform([branch])[0]
    c = le_course.transform([course])[0]

    prediction = model.predict([[b, c, marks]])
    st.success(f"🎯 Predicted Grade: **{prediction[0]}**")
