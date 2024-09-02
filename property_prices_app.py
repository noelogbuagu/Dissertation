import streamlit as st

eda_page = st.Page("property_EDA_app.py", title="Exploratory Data Analysis",
                      icon=":material/search:")
model_page = st.Page("property_prices_model_app.py", title="Model Demonstration",
                     icon=":material/bar_chart:")

pg = st.navigation([eda_page, model_page])
st.set_page_config(page_title="Home", page_icon=":material/home:")

# i can write an introduction here


pg.run()
