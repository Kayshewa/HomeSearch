#Reminders for future Kayshewa - THIS IS THE LATEST 5/26
#git add .
#git commit -m "pls commit"
#git push origin master

#NEXT STEP: INTEGRATE TEMP WEALTH DASHBOARD > HOMESEARCH.IPYNB into UI (add ZPID)

#Function to call to add new zillow ZPID to Google Sheet:

import requests
import pandas as pd
import gspread
import toml
from oauth2client.service_account import ServiceAccountCredentials

def fetch_and_update_zillow_data(zpid: str, sheet_url: str):
    # 1. Fetch data from Zillow API
    url = "https://zillow-com1.p.rapidapi.com/property"
    querystring = {"zpid": zpid}
    headers = {
        "x-rapidapi-key": "287c0c69cbmsh41c226a35114322p1022a8jsn65498208884c",
        "x-rapidapi-host": "zillow-com1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()

    # 2. Flatten the JSON response
    def flatten_json(y):
        out = {}
        def flatten(x, name=''):
            if type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + '_')
            elif type(x) is list:
                for i, a in enumerate(x):
                    flatten(a, name + str(i) + '_')
            else:
                out[name[:-1]] = x
        flatten(y)
        return out

    flat_data = flatten_json(data)
    df = pd.DataFrame([flat_data])

    # 3. Save column names to CSV
    column_names = list(df.columns)
    pd.DataFrame(column_names).to_csv("zillowattributes.csv", index=False, header=["Column Name"])

    # 4. Extract a subset of relevant columns
    try:
        shortdf = df[[
            'zpid', 'streetAddress', 'monthlyHoaFee', 'yearBuilt', 'latitude', 'longitude', 'livingAreaValue',
            'climate_floodSources_primary_insuranceRecommendation', 'climate_floodSources_primary_riskScore_label',
            'climate_floodSources_primary_riskScore_value', 'rentZestimate', 'propertyTaxRate', 'timeOnZillow',
            'url', 'zestimate', 'bedrooms', 'bathrooms', 'zipcode', 'price', 'homeStatus', 'imgSrc',
            'annualHomeownersInsurance', 'priceHistory_0_date', 'priceHistory_0_event', 'priceHistory_0_price',
            'priceHistory_1_date', 'priceHistory_1_event', 'priceHistory_1_price', 'priceHistory_2_date',
            'priceHistory_2_event', 'priceHistory_2_price', 'priceHistory_3_date', 'priceHistory_3_event',
            'priceHistory_3_price', 'priceHistory_4_date', 'priceHistory_4_event', 'priceHistory_4_price',
            'priceHistory_5_date', 'priceHistory_5_event', 'priceHistory_5_price', 'schools_0_link',
            'schools_0_rating', 'schools_0_name', 'schools_1_link', 'schools_1_rating', 'schools_1_name',
            'homeType', 'resoFacts_taxAssessedValue'
        ]]
    except KeyError as e:
        print(f"Missing key in data: {e}")
        return

    print(shortdf)

    # 5.1. Load secrets from secrets.toml
    secrets = toml.load(".streamlit/secrets.toml")  # adjust the path if necessary

    # 5.2. Extract Google credentials
    google_creds = secrets["google"]

    # 5.3. Define the scope
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

    # 5.4. Authenticate using the credentials dictionary
    creds = ServiceAccountCredentials.from_json_keyfile_dict(google_creds, scope)

    # 5.5. Authorize the client
    client = gspread.authorize(creds)


    # 6. Open the target sheet and read existing data
    spreadsheet = client.open_by_url(sheet_url)
    worksheet = spreadsheet.sheet1
    expected_headers = shortdf.columns.tolist()

    existing_data = pd.DataFrame(worksheet.get_all_records(expected_headers=expected_headers))

    # 7. Append new data if zpid not already present
    shortdf['zpid'] = shortdf['zpid'].astype(str)
    if 'zpid' not in existing_data.columns:
        worksheet.append_rows([shortdf.iloc[0].astype(str).tolist()], value_input_option='USER_ENTERED')
    elif shortdf['zpid'].values[0] not in existing_data['zpid'].astype(str).values:
        worksheet.append_rows([shortdf.iloc[0].astype(str).tolist()], value_input_option='USER_ENTERED')

# Example usage:
# fetch_and_update_zillow_data("27760224", "https://docs.google.com/spreadsheets/d/1w4suUrGjIhfn_ufYhHJqgYE9V32GEvoCSBgcanjdwms/edit?gid=0#gid=0")




#The Streamlit UI


import requests
import pandas as pd
import streamlit as st
import pydeck as pdk
import gspread
import matplotlib.pyplot as plt
from oauth2client.service_account import ServiceAccountCredentials

# Access credentials from Streamlit secrets manager
client_email = st.secrets["google"]["client_email"]
private_key = st.secrets["google"]["private_key"]
project_id = st.secrets["google"]["project_id"]

# Define the scope of permissions we need for our script to access Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Authenticate and create a client using our credentials file
#creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
#client = gspread.authorize(creds)

# Construct the credentials dictionary (simulating the structure of a JSON key file)
creds_dict = {
    "type": "service_account",
    "project_id": project_id,
    "private_key_id": None,  # You can leave this out if you're not using a key file ID
    "private_key": private_key.replace('\\n', '\n'),  # Ensure the private key is formatted correctly
    "client_email": client_email,
    "client_id": None,  # You can leave this out if you're not using a client ID
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": None,  # You can leave this out if you're not using a certificate URL
}

# Create the credentials object from the dictionary
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
# Authenticate the client using the credentials
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
st.title("Kayshewa and Tripi's Final House Explorer v3")


st.subheader("🔄 Add New Zillow Property by ZPID")

# Input box to enter ZPID
zpid_input = st.text_input("Enter ZPID:")

# Button to trigger data fetch and update
if st.button("Fetch and Add Property"):
    if zpid_input.strip() == "":
        st.warning("Please enter a valid ZPID.")
    else:
        try:
            fetch_and_update_zillow_data(zpid_input.strip(), sheet_url)
            st.success(f"Data for ZPID {zpid_input} fetched and added (if not already present).")
            st.experimental_rerun()  # Refresh to show the new data
        except Exception as e:
            st.error(f"An error occurred while fetching data: {e}")


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
    
    # Concatenate 'https://www.zillow.com' to the beginning of the URL
    full_url = f"https://www.zillow.com{property_url}" if pd.notnull(property_url) else None
    
    # Create a clickable link using Markdown
    if full_url:
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
