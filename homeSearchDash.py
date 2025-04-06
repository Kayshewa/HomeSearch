import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
import gspread
import matplotlib.pyplot as plt
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope of permissions we need for our script to access Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Authenticate and create a client using our credentials file
creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet by its URL
sheet_url = "https://docs.google.com/spreadsheets/d/1w4suUrGjIhfn_ufYhHJqgYE9V32GEvoCSBgcanjdwms/edit?gid=0#gid=0"
spreadsheet = client.open_by_url(sheet_url)

# Select the first worksheet in the spreadsheet
worksheet = spreadsheet.sheet1

# Get all values from the worksheet and store them in a pandas dataframe, pass expected_headers to avoid GSpreadException
#expected_headers = shortdf.columns.tolist()
existing_data = pd.DataFrame(worksheet.get_all_records())

# Title
st.title("Kayshewa and Tripi's Final House Explorer")


# Sidebar Filters
st.sidebar.header("🔍 Filter Listings")
# In the sidebar
st.sidebar.subheader("🏘️ Select a Property")
selected_address = st.sidebar.selectbox("Street Address:", existing_data["streetAddress"].sort_values())


# Numeric Range Filters
price_range = st.sidebar.slider("Price Range ($)", 
                                int(existing_data["price"].min()), 
                                int(existing_data["price"].max()), 
                                (int(existing_data["price"].min()), int(existing_data["price"].max())))

bedroom_filter = st.sidebar.slider("Minimum Bedrooms", 
                                   int(existing_data["bedrooms"].min()), 
                                   int(existing_data["bedrooms"].max()), 
                                   int(existing_data["bedrooms"].min()))

bathroom_filter = st.sidebar.slider("Minimum Bathrooms", 
                                    int(existing_data["bathrooms"].min()), 
                                    int(existing_data["bathrooms"].max()), 
                                    int(existing_data["bathrooms"].min()))

hoa_filter = st.sidebar.selectbox("Monthly HOA Fee", ["All", "No Fee", "With Fee"])
year_built_range = st.sidebar.slider("Year Built", 
                                     int(existing_data["yearBuilt"].min()), 
                                     int(existing_data["yearBuilt"].max()), 
                                     (int(existing_data["yearBuilt"].min()), int(existing_data["yearBuilt"].max())))

# Data Cleaning (Handle missing HOA fees)
existing_data['monthlyHoaFee'] = pd.to_numeric(existing_data['monthlyHoaFee'], errors='coerce')
existing_data['monthlyHoaFee'].fillna(0, inplace=True)

# Apply Filters
filtered_data = existing_data[
    (existing_data["price"] >= price_range[0]) &
    (existing_data["price"] <= price_range[1]) &
    (existing_data["bedrooms"] >= bedroom_filter) &
    (existing_data["bathrooms"] >= bathroom_filter) &
    (existing_data["yearBuilt"] >= year_built_range[0]) &
    (existing_data["yearBuilt"] <= year_built_range[1])
]

if hoa_filter == "No Fee":
    filtered_data = filtered_data[filtered_data["monthlyHoaFee"] == 0]
elif hoa_filter == "With Fee":
    filtered_data = filtered_data[filtered_data["monthlyHoaFee"] > 0]


# Map display
# MAP: Ensure lat/lon are numeric
filtered_data = filtered_data.dropna(subset=["latitude", "longitude"])
filtered_data["latitude"] = pd.to_numeric(filtered_data["latitude"])
filtered_data["longitude"] = pd.to_numeric(filtered_data["longitude"])

# MAP: Define the tooltip
tooltip = {
    "html": "<b>{streetAddress}</b><br>Price: ${price}<br>Bedrooms: {bedrooms}<br>Bathrooms: {bathrooms}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

# Show number of results
st.write(f"Showing {len(filtered_data)} matching properties")

#TABLE: Display Raw Data
st.dataframe(filtered_data)

selected_property = filtered_data[filtered_data["streetAddress"] == selected_address]


# Show the selected property in a transposed table format
if not selected_property.empty:
    st.subheader("Selected Property Details")

    # Create a DataFrame for the selected property
    property_details = selected_property[[
        'streetAddress', 'price', 'bedrooms', 'bathrooms', 
        'livingAreaValue', 'yearBuilt', 'monthlyHoaFee', 
        'climate_floodSources_primary_riskScore_label', 'zestimate', 'url'
    ]]

    # Rename columns for better readability
    property_details.columns = [
        'Street Address', 'Price', 'Bedrooms', 'Bathrooms', 
        'Living Area (sqft)', 'Year Built', 'HOA Fee', 
        'Flood Risk', 'Zestimate', 'url'
    ]


    # Transpose the data to have two columns: one for labels and one for values
    property_details_transposed = property_details.T.reset_index()
    
    # Rename the columns
    property_details_transposed.columns = ['Property Detail', 'Value']

    # Display the transposed table
    st.dataframe(property_details_transposed)
else:
    st.warning("No property selected or property not found.")


# Create a clickable link using Markdown
# Extract the URL from the selected property
if not selected_property.empty:
    # Extract the URL from the selected property
    property_url = selected_property["url"].values[0]  # Get the URL of the selected property
    
    # Concatenate 'https://www.zillow.com' to the beginning of the URL if the URL exists
    if pd.notnull(property_url):
        full_url = f"https://www.zillow.com{property_url}"
        # Create a clickable link using Markdown
        st.markdown(f"[View Property on Zillow]({full_url})")
    else:
        st.warning("No valid URL available for this property.")
else:
    st.warning("No property selected or no matching property found.")



# MAP: Tooltip config
tooltip = {
    "html": """
        <b>{streetAddress}</b><br>
        Price: ${price}<br>
        Bedrooms: {bedrooms}<br>
        Bathrooms: {bathrooms}<br>
        Year Built: {yearBuilt}
    """,
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

# MAP: Show all properties as points (red)
main_layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_data,
    get_position='[longitude, latitude]',
    get_radius=100,
    get_fill_color='[200, 30, 0, 160]',
    pickable=True,
)

# Optional highlight layer for selected property (blue)
highlight_layer = None
if not selected_property.empty:
    highlight_layer = pdk.Layer(
        "ScatterplotLayer",
        data=selected_property,
        get_position='[longitude, latitude]',
        get_radius=200,
        get_fill_color='[0, 150, 255, 200]',
        pickable=True,
    )

# MAP: View centered on selected property (or default view)
if not selected_property.empty:
    prop = selected_property.iloc[0]
    view_state = pdk.ViewState(
        latitude=prop["latitude"],
        longitude=prop["longitude"],
        zoom=14,
        pitch=0,
    )
else:
    view_state = pdk.ViewState(
        latitude=filtered_data["latitude"].mean(),
        longitude=filtered_data["longitude"].mean(),
        zoom=11,
        pitch=0,
    )

# Combine layers
layers = [main_layer]
if highlight_layer:
    layers.append(highlight_layer)

# MAP: Render the deck
st.subheader("📍 Property Map")
st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/light-v9"
))

# Optional: Scatter Plot
st.subheader("📊 Price vs Living Area")
st.scatter_chart(filtered_data[["livingAreaValue", "price"]].dropna().rename(columns={"livingAreaValue": "Living Area", "price": "Price"}))


# Ensure that price and zestimate are numeric (in case they're strings)
filtered_data['price'] = pd.to_numeric(filtered_data['price'], errors='coerce')
filtered_data['zestimate'] = pd.to_numeric(filtered_data['zestimate'], errors='coerce')

# Create a DataFrame with properties as the index and 'price' and 'zestimate' as columns
comparison_df = filtered_data[['streetAddress', 'price', 'zestimate']].set_index('streetAddress')

# Display a bar chart comparing Price and Zestimate for each property
st.bar_chart(comparison_df)
