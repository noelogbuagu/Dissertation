import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Streamlit app
st.title("Property Price Prediction App")
st.write(
    """    
    ## Predict the price of a property located in **Merseyside, UK** based on its features
    This app allows users to input property features, view visualizations of the data, and make predictions using a trained Random Forest model.
    """
)
st.write('---')


# load dataset
df = pd.read_csv(
    "/Users/Blurryface/Documents/GitHub/Dissertation/clean_data/final_df.csv")
X = df.drop('price', axis=1)
y = df['price']


st.write("## Data Overview")
st.dataframe(df, use_container_width=True)

# Sidebar filters
st.sidebar.header("Filter Data")

# Property Type Filter
property_types = st.sidebar.multiselect(
    "Select Property Types",
    options=df['property_type'].unique(),
    default=df['property_type'].unique()
)

# Convert to datetime and strip time component
df['transfer_date'] = pd.to_datetime(df['transfer_date']).dt.date

# Date range slider
min_date = df['transfer_date'].min()
max_date = df['transfer_date'].max()
# st.write(f"Min Date: {min_date}, Max Date: {max_date}")

date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Price Range Filter
min_price = int(df['price'].min())
max_price = int(df['price'].max())

price_range = st.sidebar.slider(
    "Select Price Range (£)",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)


# Filter the dataframe based on the selected property types
filtered_df = df[
    (df['property_type'].isin(property_types)) &
    (df['transfer_date'].between(date_range[0], date_range[1])) &
    (df['price'].between(price_range[0], price_range[1]))
]


st.write("## Property Price Prediction EDA")
st.write("""
This section provides an exploratory data analysis (EDA) of the property price data before proceeding to the predictive modeling.
""")


# Distribution of Property Prices
st.subheader("Distribution of Property Prices")
st.histogram = plt.figure(figsize=(10, 6))
sns.histplot(filtered_df['price'], kde=True, bins=50)
plt.xlabel('Price [£]')
plt.ylabel('Frequency')
plt.title('Distribution of Property Prices')
st.pyplot(st.histogram)
st.write(
    """
    - 50% of the property prices fall between approximately £100,000 and £200,000
- The median appears to be around £150,000, indicating that half of the properties are priced below this value and half are priced above.
- The majority of properties are priced between £100,000 and £200,000, with a few extending to £350,000 and above.
- The data seems to be slightly skewed to the right, as the whisker on the higher end is longer, indicating more variability in higher property prices.
- The presence of potential outliers suggests that there are some properties with prices that are significantly higher than the norm, which could be luxury properties or those in prime locations.
    """
)



# Create count plot for property types
st.subheader("Count of Property Types Sold")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(y='property_type', data=filtered_df,
              order=filtered_df['property_type'].value_counts().index, hue='property_type', ax=ax1)
ax1.set_title('Distribution of Property Types')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Property Type')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
# Display the plot in Streamlit
st.pyplot(fig1)


# Calculate the average price per property type
avg_price_per_type = filtered_df.groupby(
    'property_type')['price'].mean().reset_index()
# Create a vertical bar plot for average price per property type
st.subheader("Average Sale Price per Property Type")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x='property_type', y='price',
            data=avg_price_per_type, palette='viridis', ax=ax2)
# Annotate the bars with the average price values
for index, row in avg_price_per_type.iterrows():
    ax2.text(index, row.price, f'£{row.price:,.0f}',
             color='black', ha="center")
# Add title and labels
ax2.set_title('Average Sale Price per Property Type')
ax2.set_xlabel('Property Type')
ax2.set_ylabel('Average Sale Price (£)')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
# Adjust y-axis to give space for annotations
ax2.set_ylim(0, avg_price_per_type['price'].max() * 1.1)
# Display the plot in Streamlit
st.pyplot(fig2)


# Boxplot of Property Prices by Property Type
st.subheader("Distribution of Property Prices by Property Type")
st.boxplot = plt.figure(figsize=(10, 6))
sns.boxplot(x='property_type', y='price', data=filtered_df)
plt.xlabel('Property Type')
plt.ylabel('Price [£]')
plt.title('Property Prices by Property Type')
st.pyplot(st.boxplot)



# Create count plot for property tenure
st.subheader("Distribution of Property Tenure")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.countplot(y='is_old_or_new', data=filtered_df,
              order=filtered_df['is_old_or_new'].value_counts().index, hue='is_old_or_new', ax=ax1)
ax1.set_title('Distribution of Property Tenure')
ax1.set_xlabel('Count')
ax1.set_ylabel('Property Tenure')
# Display the plot in Streamlit
st.pyplot(fig1)


# Calculate the average price per property tenure
avg_price_per_tenure = filtered_df.groupby(
    'is_old_or_new')['price'].mean().reset_index()
# Create a vertical bar plot for average price per property tenure
st.subheader("Average Sale Price per Property Tenure")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.barplot(x='is_old_or_new', y='price',
            data=avg_price_per_tenure, palette='viridis', ax=ax2)
# Annotate the bars with the average price values
for index, row in avg_price_per_tenure.iterrows():
    ax2.text(index, row.price, f'£{row.price:,.0f}',
             color='black', ha="center")
# Add title and labels
ax2.set_title('Average Sale Price per Property Tenure')
ax2.set_xlabel('Property Tenure')
ax2.set_ylabel('Average Sale Price (£)')
# Adjust y-axis to give space for annotations
ax2.set_ylim(0, avg_price_per_tenure['price'].max() * 1.1)
# Display the plot in Streamlit
st.pyplot(fig2)




# Property Price Trends Over Time
filtered_df['transfer_date'] = pd.to_datetime(filtered_df['transfer_date'])
st.subheader("Property Price Trends Over Time")
st.write("Trend of property prices over time after applying filters:")
st.line_chart = plt.figure(figsize=(12, 6))
sns.lineplot(x=filtered_df['transfer_date'].dt.year,
             y='price', hue='property_type', data=filtered_df)
plt.xlabel('Year')
plt.ylabel('Price [£]')
plt.title('Property Price Trends Over Time by Property Type')
st.pyplot(st.line_chart)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
st.write("Correlation heatmap of features after applying filters:")
corr_matrix = filtered_df.select_dtypes('number').drop(columns='price').corr()
st.heatmap = plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True,
            fmt=".2f", linewidths=.5, annot_kws={'size': 7})
plt.title('Correlation Heatmap of Features')
st.pyplot(st.heatmap)




# Display the interactive map
st.subheader("Map of Property Prices by Location")
# Creating the scatter mapbox plot
fig = px.scatter_mapbox(
    filtered_df,
    lat='latitude',
    lon='longitude',
    color='price',
    hover_data=['town'],
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15,
    zoom=10,
    height=600
)
# Update layout with open-street-map style
fig.update_layout(mapbox_style="open-street-map", height=600, width=800)
# Show the figure in the Streamlit app
st.plotly_chart(fig)



