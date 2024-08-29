import streamlit as st

eda_page = st.Page("property_EDA_app.py", title="Exploratory Data Analysis",
                      icon=":material/add_circle:")
model_page = st.Page("property_prices_model_app.py", title="Model Demonstration",
                      icon=":material/delete:")

pg = st.navigation([eda_page, model_page])
st.set_page_config(page_title="Home", page_icon=":material/edit:")

# i can write an introduction here


pg.run()
