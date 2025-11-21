import streamlit as st
import pandas as pd
import pickle


profit_model = pickle.load(open("models/profit_model.pkl", "rb"))
failure_model = pickle.load(open("models/failure_model.pkl", "rb"))


df = pd.read_csv("data/micro_business_dataset_500.csv")

st.title("AI Micro-Entrepreneur ")
st.write("Enter details below to predict monthly profit and failure risk.")


city = st.selectbox("City", df['City'].unique())
product = st.selectbox("Product / Service", df['Product/Service'].unique())
marketing_channel = st.selectbox("Marketing Channel", df['Marketing_Channel'].unique())

startup_cost = st.number_input("Startup Cost (PKR)", min_value=0)
cost_per_unit = st.number_input("Cost per Unit (PKR)", min_value=0)
price_per_unit = st.number_input("Price per Unit (PKR)", min_value=0)

if st.button("Predict"):
    
    
    input_df = pd.DataFrame(columns=profit_model.feature_names_in_)
    input_df.loc[0] = 0  # initialize all columns to 0

    # Step 2: Fill numeric values
    input_df['Startup_Cost_PKR'] = startup_cost
    input_df['Cost_per_Unit'] = cost_per_unit
    input_df['Price_per_Unit'] = price_per_unit

    
    col_map = {
        f"City_{city}": 1,
        f"Product/Service_{product}": 1,
        f"Marketing_Channel_{marketing_channel}": 1
    }

    for col in col_map:
        if col in input_df.columns:
            input_df[col] = 1

    # Step 4: Predictions
    predicted_profit = profit_model.predict(input_df)[0]
    predicted_failure = failure_model.predict_proba(input_df)[0][1]  

    # Step 5: Display Results
    st.subheader(" Prediction Results")
    st.write(f"**Estimated Monthly Profit:** {round(predicted_profit, 2)} PKR")
    st.write(f"**Failure Risk Probability:** {round(predicted_failure, 2)}")

