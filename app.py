import streamlit as st
import pandas as pd
import pickle


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Vehicle Resale Price Predictor",
    page_icon="🚗",
    layout="centered"
)


# --------------------------------------------------
# Load Trained Model and Scaler
# --------------------------------------------------

@st.cache_resource
def load_model():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🚗 Vehicle Resale Price Predictor")

st.write(
    "Enter the details of a used vehicle to estimate "
    "its resale price."
)


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("Vehicle Details")

col1, col2 = st.columns(2)


with col1:

    car_age = st.number_input(
        "Car Age (years)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    mileage = st.number_input(
        "Mileage (km)",
        min_value=1000,
        max_value=300000,
        value=60000,
        step=1000
    )

    condition_name = st.selectbox(
        "Condition",
        ["Poor", "Fair", "Good", "Excellent"],
        index=2
    )

    owner_count = st.number_input(
        "Number of Previous Owners",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )


with col2:

    original_price = st.number_input(
        "Original Price (₹ Lakh)",
        min_value=0.5,
        max_value=100.0,
        value=8.0,
        step=0.5
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol", "Diesel"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual", "Automatic"]
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Resale Price", type="primary"):

    # Convert condition to numerical value
    condition_mapping = {
        "Poor": 1,
        "Fair": 2,
        "Good": 3,
        "Excellent": 4
    }

    condition = condition_mapping[condition_name]


    # Feature Engineering
    mileage_per_year = mileage / car_age

    estimated_value = (
        original_price * (1 - 0.10 * car_age)
    )

    estimated_value = max(
        estimated_value,
        0.1 * original_price
    )


    # --------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------

    new_car = pd.DataFrame([{

        "Car_Age": car_age,

        "Mileage_km": mileage,

        "Condition": condition,

        "Owner_Count": owner_count,

        "Original_Price_Lakh": original_price,

        "Mileage_Per_Year": mileage_per_year,

        "Estimated_Value": estimated_value,

        "Fuel_Type_Diesel":
            1 if fuel_type == "Diesel" else 0,

        "Fuel_Type_Petrol":
            1 if fuel_type == "Petrol" else 0,

        "Transmission_Manual":
            1 if transmission == "Manual" else 0
    }])


    # --------------------------------------------------
    # Scale input
    # --------------------------------------------------

    new_car_scaled = scaler.transform(new_car)


    # --------------------------------------------------
    # Make prediction
    # --------------------------------------------------

    prediction = model.predict(new_car_scaled)[0]


    # Prevent negative prediction
    prediction = max(prediction, 0)


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.success(
        f"### Estimated Resale Price: ₹{prediction:.2f} Lakh"
    )

    st.write(
        f"Approximately ₹{prediction * 100000:,.0f}"
    )


# --------------------------------------------------
# Additional Information
# --------------------------------------------------

st.divider()

with st.expander("How does this model work?"):

    st.write("""
    The model uses Linear Regression to estimate the resale
    price of a vehicle.

    The prediction is based on:

    • Car age
    • Mileage
    • Vehicle condition
    • Number of previous owners
    • Original price
    • Mileage per year
    • Estimated vehicle value
    • Fuel type
    • Transmission type

    The input features are standardized using the same
    StandardScaler used during model training.
    """)