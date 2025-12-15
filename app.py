import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# -----------------------------
# Load dataset and train model
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("StudentMarksDataset.csv")  # Make sure the CSV is in the same folder
    return df

df = load_data()

# Assign grades
def Grade_class(marks):
    if marks >= 90:
        return 'A'
    elif marks >= 80:
        return 'B'
    elif marks >= 70:
        return 'C'
    elif marks >= 60:
        return 'D'
    elif marks >= 50:
        return 'E'
    else:
        return 'F'

df["Grade"] = df["Std_Marks"].apply(Grade_class)

# Encode categorical features
le_branch = LabelEncoder()
le_course = LabelEncoder()
df["Std_Branch_enc"] = le_branch.fit_transform(df["Std_Branch"])
df["Std_Course_enc"] = le_course.fit_transform(df["Std_Course"])

# Features and target
X = df[["Std_Branch_enc", "Std_Course_enc", "Std_Marks"]]
y = df["Grade"]

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
rf_model.fit(X, y)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Student Grade Prediction App 🎓")

branch_list = df["Std_Branch"].unique().tolist()
course_list = df["Std_Course"].unique().tolist()

branch_input = st.selectbox("Select Branch", branch_list)
course_input = st.selectbox("Select Course", course_list)
marks_input = st.number_input("Enter Marks", min_value=0, max_value=100, value=75)

if st.button("Predict Grade"):
    # Encode input
    branch_enc = le_branch.transform([branch_input])[0]
    course_enc = le_course.transform([course_input])[0]

    # Create feature array
    input_data = [[branch_enc, course_enc, marks_input]]

    # Predict
    grade_pred = rf_model.predict(input_data)[0]

    st.success(f"Predicted Grade: {grade_pred}")

