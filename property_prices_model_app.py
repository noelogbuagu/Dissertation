import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Load dataset to define feature ranges
df = pd.read_csv(
    "/Users/Blurryface/Documents/GitHub/Dissertation/clean_data/final_df.csv")
df['year'] = pd.to_datetime(df['transfer_date']).dt.year
df['month'] = pd.to_datetime(df['transfer_date']).dt.month

# Sidebar for model input features
st.sidebar.header('Model Input Features')

with st.sidebar.form(key='model_input_form'):
    Property_type = st.selectbox(
        'property_type', df.property_type.unique())
    Old_or_New = st.selectbox(
        'is_old_or_new', df.is_old_or_new.unique())
    Property_tenure = st.selectbox(
        'property_tenure', df.property_tenure.unique())
    Town = st.selectbox('town', df.town.unique())
    District = st.selectbox('district', df.district.unique())
    Transaction_category = st.selectbox(
        'ppd_transaction_category', df.ppd_transaction_category.unique())
    Longitude = st.slider(
        'longitude', df.longitude.min(), df.longitude.max(), df.longitude.mean())
    Latitude = st.slider(
        'latitude', df.latitude.min(), df.latitude.max(), df.latitude.mean())
    Employment_rate = st.slider('employment_rate', df.employment_rate.min(
    ), df.employment_rate.max(), df.employment_rate.mean())
    Average_price = st.slider('average_price', df.average_price.min(
    ), df.average_price.max(), df.average_price.mean())
    Index = st.slider(
        'index', df['index'].min(), df['index'].max(), df['index'].mean())
    One_month_change = st.slider('1m%_change', df['1m%_change'].min(
    ), df['1m%_change'].max(), df['1m%_change'].mean())
    Twelve_month_change = st.slider('12m%_change', df['12m%_change'].min(
    ), df['12m%_change'].max(), df['12m%_change'].mean())
    Sales_volume = st.slider('sales_volume', df.sales_volume.min(
    ), df.sales_volume.max(), df.sales_volume.mean())
    Gross_median_weekly_pay = st.slider('gross_median_weekly_pay', df.gross_median_weekly_pay.min(
    ), df.gross_median_weekly_pay.max(), df.gross_median_weekly_pay.mean())
    Minor_misc_crime_density = st.slider('minor_misc_crime_density', df.minor_misc_crime_density.min(
    ), df.minor_misc_crime_density.max(), df.minor_misc_crime_density.mean())
    Anti_social_behavior_density = st.slider('anti_social_behavior_density', df.anti_social_behavior_density.min(
    ), df.anti_social_behavior_density.max(), df.anti_social_behavior_density.mean())
    Violent_crime_density = st.slider('violent_crime_density', df.violent_crime_density.min(
    ), df.violent_crime_density.max(), df.violent_crime_density.mean())
    Property_crime_density = st.slider('property_crime_density', df.property_crime_density.min(
    ), df.property_crime_density.max(), df.property_crime_density.mean())
    Drug_related_crime_density = st.slider('drug_related_crime_density', df.drug_related_crime_density.min(
    ), df.drug_related_crime_density.max(), df.drug_related_crime_density.mean())
    flood_proximity = st.selectbox(
        'flood_proximity', df.flood_proximity.unique())
    Year = st.slider('year', min_value=2013,
                     max_value=2023, value=2023)
    Month = st.slider('month', min_value=1,
                      max_value=12, value=1, format="%d")

    # Form submission button
    submit_button = st.form_submit_button(label='Predict')

# Introductory message
if not submit_button:
    st.title("Welcome to the Property Price Prediction App")
    st.write(
        """
        This app predicts the price of a property based on various factors such as property type, location, and crime density.
        
        To get started:
        1. Adjust the input parameters in the sidebar.
        2. Press the "Predict" button to see the estimated property price.
        
        The app will display the specified input features and the predicted price after you press the "Predict" button.
        """)
    

if submit_button:
    # Gather all the inputs into a DataFrame
    user_input_df = pd.DataFrame({
        'property_type': [Property_type],
        'is_old_or_new': [Old_or_New],
        'property_tenure': [Property_tenure],
        'town': [Town],
        'district': [District],
        'ppd_transaction_category': [Transaction_category],
        'longitude': [Longitude],
        'latitude': [Latitude],
        'employment_rate': [Employment_rate],
        'average_price': [Average_price],
        'index': [Index],
        '1m%_change': [One_month_change],
        '12m%_change': [Twelve_month_change],
        'sales_volume': [Sales_volume],
        'gross_median_weekly_pay': [Gross_median_weekly_pay],
        'minor_misc_crime_density': [Minor_misc_crime_density],
        'anti_social_behavior_density': [Anti_social_behavior_density],
        'violent_crime_density': [Violent_crime_density],
        'property_crime_density': [Property_crime_density],
        'drug_related_crime_density': [Drug_related_crime_density],
        'flood_proximity': [flood_proximity],
        'year': [Year],
        'month': [Month]
    })

    # Display the specified input features
    st.header('Specified Model Input')
    st.write(user_input_df)
    st.write('---')

    
    try:
        # Load the trained pipeline (preprocessor + model)
        with open("/Users/Blurryface/Documents/GitHub/Dissertation/final_model_pipeline.pkl", "rb") as f:
            model_pipeline = pickle.load(f)

        # Apply the same preprocessing to the input data
        user_input_preprocessed = model_pipeline.named_steps['preprocessor'].transform(
            user_input_df)

        st.header('Property Price Prediction')
        # Predict based on user input
        prediction = model_pipeline.named_steps['model'].predict(
            user_input_preprocessed)
        st.write(f"Predicted Property Price: £{np.exp(prediction[0]):,.2f}")


        # Display Feature Importance
        feature_importances = model_pipeline.named_steps['model'].feature_importances_
        feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out(
        )
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(
            by='Importance', ascending=True)
        top_10_features = importance_df.head(10)

        st.header('Feature Importance')

        # Plot with Plotly for interactivity
        fig = px.bar(top_10_features, x='Importance', y='Feature',
                     orientation='h', title='Top 10 Feature Importance')
        fig.update_layout(xaxis_title='Importance',
                          yaxis_title='Feature', height=600)
        st.plotly_chart(fig)

        
    except ValueError as e:
        st.error(f"Error making prediction: {e}")

    st.write('---')
