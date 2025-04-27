# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Configuration ---
# Define the filenames of your pickled model and the columns list
MODEL_FILE = 'D:\Desktop\data analytics projects\Amazon delivery time\simple_delivery_model.pkl' # Make sure this matches the name you saved your model as
COLUMNS_FILE = 'D:\Desktop\data analytics projects\Amazon delivery time\model_columns.pkl'       # Make sure this matches the name you saved your columns list as

# Define the original feature names that the user will input
# These are the columns you selected before one-hot encoding, plus the target 'Delivery_Time'
# We need the original categorical columns to create input fields
ORIGINAL_FEATURES = [
    'Agent_Rating', 'Agent_Age', 'Weather', 'Traffic',
    'Vehicle', 'Area', 'Category'
]

# Define the possible values for categorical features based on your training data
# This is needed to create select boxes in Streamlit and ensure correct one-hot encoding
# IMPORTANT: Replace these lists with the actual unique values from your training data
WEATHER_OPTIONS = ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Unknown', 'Fog', 'Windy'] # Add all unique weather types
TRAFFIC_OPTIONS = ['High', 'Jam', 'Low', 'Medium'] # Add all unique traffic types
VEHICLE_OPTIONS = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle'] # Add all unique vehicle types
AREA_OPTIONS = ['Urban', 'Metropolitian', 'Semi-Urban'] # Add all unique area types
CATEGORY_OPTIONS = [
    'Clothing', 'Electronics', 'Sports', 'Cosmetics', 'Toys', 'Grocery',
    'Home & Kitchen', 'Furniture', 'Food', 'Books', 'Fashion', 'Pet Supplies',
    'Stationery', 'Medicine', 'Vehicles', 'Accessories'
] # Add all unique category types


# --- Function to Load the Model and Columns ---
@st.cache_resource # Cache the loading to avoid reloading on every interaction
def load_assets(model_path, columns_path):
    """Loads the pickled model and the list of columns."""
    try:
        with open(model_path, 'rb') as model_f:
            model = pickle.load(model_f)
        with open(columns_path, 'rb') as columns_f:
            model_columns = pickle.load(columns_f)
        return model, model_columns
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path} or Columns file not found at {columns_path}")
        st.stop() # Stop the app if files aren't found
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

# --- Load the model and columns ---
model, model_columns = load_assets(MODEL_FILE, COLUMNS_FILE)

# --- Streamlit App Layout ---
st.title("Amazon Delivery Time Predictor")
st.write("Enter the order details below to predict the delivery time.")

# --- Input Fields for Original Features ---
input_data = {}
st.sidebar.header("Order Details")

# Create input fields for each original feature
# Use appropriate input widgets based on data type
input_data['Agent_Age'] = st.sidebar.number_input("Agent Age", min_value=15, max_value=60, value=30, step=1)
input_data['Agent_Rating'] = st.sidebar.number_input("Agent Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
input_data['Weather'] = st.sidebar.selectbox("Weather Conditions", WEATHER_OPTIONS)
input_data['Traffic'] = st.sidebar.selectbox("Traffic Density", TRAFFIC_OPTIONS)
input_data['Vehicle'] = st.sidebar.selectbox("Vehicle Type", VEHICLE_OPTIONS)
input_data['Area'] = st.sidebar.selectbox("Delivery Area", AREA_OPTIONS)
input_data['Category'] = st.sidebar.selectbox("Product Category", CATEGORY_OPTIONS)


# --- Create a DataFrame from Inputs ---
# Create a DataFrame with a single row from the input data
input_df = pd.DataFrame([input_data])

st.subheader("Input Data Provided:")
st.write(input_df)

# --- Preprocess Input Data ---
# Apply the same preprocessing steps as used during training
try:
    # 1. Handle missing values (Agent_Rating and Weather) - Although Streamlit inputs
    #    reduce missing data risk, we include this for robustness if input method changes.
    #    For Agent_Rating, impute with median (you'd need to know the median from training)
    #    For Weather, fill with 'Unknown' if somehow empty (selectbox prevents this)
    #    NOTE: Getting the exact training median here would require saving it or recalculating.
    #    For simplicity with number_input, we assume no missing Agent_Rating.
    #    For selectbox, we assume no missing Weather.

    # 2. Perform One-Hot Encoding
    input_encoded = pd.get_dummies(input_df, columns=['Weather', 'Traffic', 'Vehicle', 'Area', 'Category'])

    # 3. Align columns with the training data columns (model_columns)
    #    This is crucial to ensure the input DataFrame has the same columns in the same order
    #    as the data the model was trained on. Missing columns will be filled with 0.
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    st.subheader("Processed Input Data (One-Hot Encoded & Aligned):")
    st.write(input_aligned)

except Exception as e:
    st.error(f"Error during data preprocessing: {e}")
    st.stop()


# --- Prediction Button ---
if st.button("Predict Delivery Time"):
    # --- Make Prediction ---
    try:
        # Use the preprocessed and aligned input data for prediction
        predicted_time = model.predict(input_aligned)

        # The model predicts a numerical value (delivery time in minutes)
        st.success(f"Predicted Delivery Time: **{predicted_time[0]:.2f} minutes**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Instructions for the user ---
st.markdown("""
---
**How to use this app:**

1.  Make sure you have saved your trained scikit-learn `RandomForestRegressor` model as `simple_delivery_model.pkl`.
2.  Save the list of column names used to train the model (after one-hot encoding) as `model_columns.pkl`.
3.  Place both `.pkl` files in the same directory as this Python script (`amazon_delivery_app.py`).
4.  Open your terminal or command prompt.
5.  Navigate to the directory where you saved the files.
6.  Run the command: `streamlit run amazon_delivery_app.py`
7.  The app will open in your web browser. Enter the order details in the sidebar.
8.  Click "Predict Delivery Time" to get the estimated delivery time.
""")

