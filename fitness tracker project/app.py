import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

import warnings
warnings.filterwarnings('ignore')

# Cache model training to optimize performance
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    model.fit(X_train, y_train)
    return model

# App Title and Description
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.markdown(
    "This app predicts the calories burned based on user parameters like `Age`, `Gender`, `BMI`, etc. "
    "Adjust the values on the sidebar and see your estimated calorie burn."
)

# Sidebar for User Inputs
st.sidebar.header("⚙️ User Input Parameters")
def user_input_features():
    col1, col2 = st.sidebar.columns(2)
    age = col1.slider("Age", 10, 100, 30)
    bmi = col2.slider("BMI", 15, 40, 20)
    duration = col1.slider("Duration (min)", 0, 35, 15)
    heart_rate = col2.slider("Heart Rate", 60, 130, 80)
    body_temp = col1.slider("Body Temperature (°C)", 36, 42, 38)
    gender = 1 if st.sidebar.radio("Gender", ["Male", "Female"]) == "Male" else 0
    
    return pd.DataFrame({
        "Age": [age], "BMI": [bmi], "Duration": [duration], 
        "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]
    })

df = user_input_features()

# Display User Parameters
col1, col2 = st.columns(2)
with col1:
    st.subheader("📌 Your Selected Parameters")
    st.dataframe(df)

# Load Data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])

# Feature Engineering
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)
exercise_train, exercise_test = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Prepare Training Data
features = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]
exercise_train = pd.get_dummies(exercise_train[features], drop_first=True)
exercise_test = pd.get_dummies(exercise_test[features], drop_first=True)
X_train, y_train = exercise_train.drop("Calories", axis=1), exercise_train["Calories"]

# Train Model
model = train_model(X_train, y_train)

# Align Input Data & Predict
prediction = model.predict(df.reindex(columns=X_train.columns, fill_value=0))

with col2:
    st.subheader("🔥 Estimated Calories Burned")
    st.success(f"{round(prediction[0], 2)} kcal")

# Show Progress Toast Instead of Progress Bar
st.toast("🔄 Processing your input...")

# Display Similar Results
with st.expander("📊 Similar Results in Dataset"):
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    st.dataframe(similar_data.sample(5))

# Insights Compared to Dataset
with st.expander("📈 General Insights"):
    st.write(f"- You are older than **{round(sum(exercise_df['Age'] < df['Age'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")
    st.write(f"- Your exercise duration is longer than **{round(sum(exercise_df['Duration'] < df['Duration'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")
    st.write(f"- Your heart rate is higher than **{round(sum(exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")
    st.write(f"- Your body temperature is higher than **{round(sum(exercise_df['Body_Temp'] < df['Body_Temp'].values[0]) / len(exercise_df) * 100, 2)}%** of users.")

