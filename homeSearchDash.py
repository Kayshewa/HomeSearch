#Reminders for future Kayshewa - THIS IS THE LATEST 5/26
#git add .
#git commit -m "pls commit"
#git push origin master

#NEXT STEP: Figure out how when you click the map, it then repopulates the Selected_Address dropdown on the left menu

#Function to call to add new zillow ZPID to Google Sheet:

import requests
import pandas as pd
import gspread
import toml
import re
import plotly.express as px
import plotly.graph_objects as go
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

    desired_columns = [
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
    ]

    # Check which columns exist in the dataframe
    existing_columns = [col for col in desired_columns if col in df.columns]

    # Create a new DataFrame with existing columns
    shortdf = df[existing_columns].copy()

    # Add any missing columns with empty strings (or NaN if you prefer)
    missing_columns = [col for col in desired_columns if col not in df.columns]
    for col in missing_columns:
        shortdf[col] = ""  # or use pd.NA if you prefer

    # Optional: reorder columns to match desired_columns order
    shortdf = shortdf[desired_columns]

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
    #shortdf['zpid'] = shortdf['zpid'].astype(str)
    shortdf = shortdf.copy()
    shortdf['zpid'] = shortdf['zpid'].astype(str)
    if 'zpid' not in existing_data.columns:
        worksheet.append_rows([shortdf.iloc[0].astype(str).tolist()], value_input_option='USER_ENTERED')
    elif shortdf['zpid'].values[0] not in existing_data['zpid'].astype(str).values:
        worksheet.append_rows([shortdf.iloc[0].astype(str).tolist()], value_input_option='USER_ENTERED')

# Example usage:
# fetch_and_update_zillow_data("27760224", "https://docs.google.com/spreadsheets/d/1w4suUrGjIhfn_ufYhHJqgYE9V32GEvoCSBgcanjdwms/edit?gid=0#gid=0")


def sanitize_dataframe_for_streamlit(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Clean stringified 'None' or Python None
            df[col] = df[col].replace(['None', None], pd.NA)

            # Try coercing to numeric, otherwise convert to string
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                df[col] = df[col].astype(str)

    # Special handling for known problematic columns
    if 'priceHistory_1_price' in df.columns:
        # Coerce to string to match what Arrow expects if mixed types
        df['priceHistory_1_price'] = df['priceHistory_1_price'].astype(str)

    if 'rentZestimate' in df.columns:
        df['rentZestimate'] = pd.to_numeric(df['rentZestimate'], errors='coerce')

    if 'Value' in df.columns:
        df['Value'] = df['Value'].astype("string")

    # Convert all remaining columns to best types
    df = df.convert_dtypes()

    return df


def extract_zpid_from_url(url):
    """
    Extract ZPID from Zillow URL.
    Supports various Zillow URL formats.
    """
    # Common Zillow URL patterns
    patterns = [
        r'/(\d+)_zpid/?',  # Standard format: /12345_zpid/
        r'zpid=(\d+)',     # Query parameter: ?zpid=12345
        r'/(\d{8,})/?$'    # Direct ZPID at end of URL (8+ digits)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def validate_zpid(zpid_str):
    """
    Validate that the zpid is a numeric string of reasonable length.
    """
    if not zpid_str.isdigit():
        return False
    if len(zpid_str) < 6 or len(zpid_str) > 12:  # Reasonable ZPID length
        return False
    return True


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

#clense for mixed datatypes
existing_data = sanitize_dataframe_for_streamlit(existing_data)

# Title
st.title("Kayshewa and Tripi's Final House Explorer")


st.subheader("🔄 Add New Zillow Property by ZPID")

# Input box to enter ZPID or URL
zpid_input = st.text_input("Enter ZPID or Zillow URL:")

# Button to trigger data fetch and update
if st.button("Fetch and Add Property", key="fetch_property_btn"):
    if zpid_input.strip() == "":
        st.warning("Please enter a valid ZPID or Zillow URL.")
    else:
        input_value = zpid_input.strip()
        zpid = None
        
        # Check if input is a URL (contains 'zillow.com' or starts with http)
        if 'zillow.com' in input_value.lower() or input_value.startswith(('http://', 'https://')):
            # Extract ZPID from URL
            zpid = extract_zpid_from_url(input_value)
            if zpid:
                st.info(f"Extracted ZPID: {zpid}")
            else:
                st.error("Could not extract ZPID from the provided URL. Please check the URL format.")
        elif input_value.isdigit():
            # Direct ZPID input
            zpid = input_value
        else:
            st.error("Please enter either a valid ZPID (numbers only) or a Zillow URL.")
        
        # Proceed if we have a valid ZPID
        if zpid and validate_zpid(zpid):
            try:
                fetch_and_update_zillow_data(zpid, sheet_url)
                st.success(f"Data for ZPID {zpid} fetched and added (if not already present).")
                st.rerun()  # Refresh to show the new data
            except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")
        elif zpid:
            st.error(f"Invalid ZPID format: {zpid}. ZPID should be 6-12 digits.")


# Sidebar Filters
st.sidebar.header("🔍 Filter Listings")
st.sidebar.subheader("🏘️ Select a Property")

# Initialize session state for selected address if it doesn't exist
if 'selected_address' not in st.session_state:
    st.session_state.selected_address = existing_data["streetAddress"].sort_values().iloc[0]

# Function to update selected address and trigger rerun
def update_selected_address(new_address):
    if new_address != st.session_state.selected_address:
        st.session_state.selected_address = new_address
        st.rerun()

# Use session state for the selectbox
selected_address = st.sidebar.selectbox(
    "Street Address:", 
    existing_data["streetAddress"].sort_values(),
    index=existing_data["streetAddress"].sort_values().tolist().index(st.session_state.selected_address) 
    if st.session_state.selected_address in existing_data["streetAddress"].values else 0,
    key="address_selectbox"
)

# Update session state when selectbox changes
if selected_address != st.session_state.selected_address:
    update_selected_address(selected_address)

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
existing_data['monthlyHoaFee'] = existing_data['monthlyHoaFee'].fillna(0)

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


# ========== INTERACTIVE MAP ==========
st.subheader("📍 Property Map")

map_data = filtered_data.dropna(subset=["latitude", "longitude"]).copy()
map_data["latitude"] = pd.to_numeric(map_data["latitude"])
map_data["longitude"] = pd.to_numeric(map_data["longitude"])

# Add color and size columns for selected vs unselected properties
map_data['is_selected'] = map_data['streetAddress'] == st.session_state.selected_address
map_data['color'] = map_data['is_selected'].map({True: 'Selected Property', False: 'Available Properties'})
map_data['size'] = map_data['is_selected'].map({True: 15, False: 10})

# Create the map
fig_map = px.scatter_map(
    map_data,
    lat="latitude",
    lon="longitude",
    color="color",
    size="size",
    hover_name="streetAddress",
    hover_data={
        "price": ":$,.0f",
        "bedrooms": True,
        "bathrooms": True,
        "yearBuilt": True,
        "livingAreaValue": ":,.0f",
        "latitude": False,
        "longitude": False,
        "is_selected": False,
        "color": False,
        "size": False
    },
    color_discrete_map={
        'Selected Property': '#DC143C',
        'Available Properties': '#4682B4'
    },
    size_max=20,
    zoom=11,
    height=600,
    title="Click on any property to select it"
)

fig_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
)

# Center map on selected property
selected_data = map_data[map_data["streetAddress"] == st.session_state.selected_address]
if not selected_data.empty:
    prop = selected_data.iloc[0]
    fig_map.update_layout(mapbox=dict(center=dict(lat=prop["latitude"], lon=prop["longitude"]), zoom=14))

# Display map and handle clicks
clicked_data = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", selection_mode="points")

if clicked_data and 'selection' in clicked_data and clicked_data['selection']['points']:
    point = clicked_data['selection']['points'][0]
    point_index = point['point_index']
    clicked_address = map_data.iloc[point_index]['streetAddress']
    update_selected_address(clicked_address)

# Navigation buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬅️ Previous", key="map_prev"):
        addresses = filtered_data["streetAddress"].sort_values().tolist()
        if st.session_state.selected_address in addresses:
            current_idx = addresses.index(st.session_state.selected_address)
            if current_idx > 0:
                update_selected_address(addresses[current_idx - 1])

with col2:
    if st.button("🎯 Center Map", key="map_center"):
        st.rerun()

with col3:
    if st.button("➡️ Next", key="map_next"):
        addresses = filtered_data["streetAddress"].sort_values().tolist()
        if st.session_state.selected_address in addresses:
            current_idx = addresses.index(st.session_state.selected_address)
            if current_idx < len(addresses) - 1:
                update_selected_address(addresses[current_idx + 1])

st.info(f"📍 **Currently Selected:** {st.session_state.selected_address}")

# ========== INTERACTIVE SCATTER PLOT ==========
st.subheader("📊 Price vs Living Area")

scatter_data = filtered_data[["livingAreaValue", "price", "streetAddress", "zpid"]].dropna()

# Create scatter plot with highlighted selected property
fig_scatter = px.scatter(
    scatter_data, 
    x="livingAreaValue", 
    y="price",
    hover_data={'streetAddress': True, 'livingAreaValue': True, 'price': True, 'zpid': True},
    labels={
        'livingAreaValue': 'Living Area (sqft)',
        'price': 'Price ($)',
        'streetAddress': 'Street Address'
    },
    title="Click on any point to select that property"
)

# Update all points to be blue first
fig_scatter.update_traces(
    hovertemplate='<b>%{customdata[0]}</b><br>' +
                  'Living Area: %{x} sqft<br>' +
                  'Price: $%{y:,.0f}<br>' +
                  'ZPID: %{customdata[3]}<br>' +
                  '<extra></extra>',
    customdata=scatter_data[['streetAddress', 'livingAreaValue', 'price', 'zpid']],
    marker=dict(color='lightblue', size=8)
)

# Add red dot for selected property
selected_scatter_point = scatter_data[scatter_data['streetAddress'] == st.session_state.selected_address]
if not selected_scatter_point.empty:
    fig_scatter.add_trace(go.Scatter(
        x=selected_scatter_point['livingAreaValue'],
        y=selected_scatter_point['price'],
        mode='markers',
        marker=dict(color='red', size=12, symbol='diamond'),
        name='Selected Property',
        hovertemplate='<b>SELECTED: %{customdata[0]}</b><br>' +
                     'Living Area: %{x} sqft<br>' +
                     'Price: $%{y:,.0f}<br>' +
                     'ZPID: %{customdata[3]}<br>' +
                     '<extra></extra>',
        customdata=selected_scatter_point[['streetAddress', 'livingAreaValue', 'price', 'zpid']]
    ))

# Display scatter plot and handle clicks
clicked_scatter_data = st.plotly_chart(fig_scatter, use_container_width=True, on_select="rerun", selection_mode="points")

if clicked_scatter_data and 'selection' in clicked_scatter_data and clicked_scatter_data['selection']['points']:
    point = clicked_scatter_data['selection']['points'][0]
    # Handle clicks on the main scatter points (trace 0)
    if point.get('curve_number', 0) == 0:
        point_index = point['point_index']
        clicked_address = scatter_data.iloc[point_index]['streetAddress']
        update_selected_address(clicked_address)

# ========== INTERACTIVE BAR CHART ==========
st.subheader("💰 Price vs Zestimate Comparison")

# Prepare data for bar chart
filtered_data['price'] = pd.to_numeric(filtered_data['price'], errors='coerce')
filtered_data['zestimate'] = pd.to_numeric(filtered_data['zestimate'], errors='coerce')

bar_data = filtered_data[['streetAddress', 'price', 'zestimate']].dropna()

# Create interactive bar chart
fig_bar = go.Figure()

# Add price bars
fig_bar.add_trace(go.Bar(
    name='Listed Price',
    x=bar_data['streetAddress'],
    y=bar_data['price'],
    marker_color='lightblue',
    hovertemplate='<b>%{x}</b><br>Listed Price: $%{y:,.0f}<extra></extra>'
))

# Add zestimate bars
fig_bar.add_trace(go.Bar(
    name='Zestimate',
    x=bar_data['streetAddress'],
    y=bar_data['zestimate'],
    marker_color='lightcoral',
    hovertemplate='<b>%{x}</b><br>Zestimate: $%{y:,.0f}<extra></extra>'
))

# Highlight selected property
selected_bar_data = bar_data[bar_data['streetAddress'] == st.session_state.selected_address]
if not selected_bar_data.empty:
    # Add highlighted bars for selected property
    fig_bar.add_trace(go.Bar(
        name='Selected - Listed Price',
        x=[st.session_state.selected_address],
        y=[selected_bar_data['price'].iloc[0]],
        marker_color='darkblue',
        showlegend=False,
        hovertemplate='<b>SELECTED: %{x}</b><br>Listed Price: $%{y:,.0f}<extra></extra>'
    ))
    
    fig_bar.add_trace(go.Bar(
        name='Selected - Zestimate',
        x=[st.session_state.selected_address],
        y=[selected_bar_data['zestimate'].iloc[0]],
        marker_color='darkred',
        showlegend=False,
        hovertemplate='<b>SELECTED: %{x}</b><br>Zestimate: $%{y:,.0f}<extra></extra>'
    ))

fig_bar.update_layout(
    title="Click on any bar to select that property",
    xaxis_title="Property Address",
    yaxis_title="Price ($)",
    barmode='group',
    xaxis={'categoryorder': 'total descending'},
    height=500
)

# Make x-axis labels more readable
fig_bar.update_xaxes(tickangle=45)

# Display bar chart and handle clicks
clicked_bar_data = st.plotly_chart(fig_bar, use_container_width=True, on_select="rerun", selection_mode="points")

if clicked_bar_data and 'selection' in clicked_bar_data and clicked_bar_data['selection']['points']:
    point = clicked_bar_data['selection']['points'][0]
    # Get the address from the clicked point
    clicked_address = point['x']
    update_selected_address(clicked_address)

# ========== PROPERTY DETAILS SECTION ==========
st.subheader("🏠 Selected Property Details")

selected_property = existing_data[existing_data['streetAddress'] == st.session_state.selected_address]
if not selected_property.empty:
    prop = selected_property.iloc[0]
    
    # Create columns for property details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Listed Price", f"${prop['price']:,.0f}")
        st.metric("Bedrooms", f"{prop['bedrooms']}")
        st.metric("Year Built", f"{prop['yearBuilt']}")
    
    with col2:
        st.metric("Zestimate", f"${prop['zestimate']:,.0f}")
        st.metric("Bathrooms", f"{prop['bathrooms']}")
        st.metric("Living Area", f"{prop['livingAreaValue']:,.0f} sqft")
    
    with col3:
        price_diff = prop['price'] - prop['zestimate']
        st.metric("Price vs Zestimate", f"${price_diff:,.0f}", 
                 delta=f"{((price_diff/prop['zestimate'])*100):+.1f}%")
        st.metric("Monthly HOA", f"${prop['monthlyHoaFee']:,.0f}")
        if prop['url']:
            st.markdown(f"[View on Zillow]({prop['url']})")

# ========== SUMMARY STATISTICS ==========
st.subheader("📈 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Properties", len(filtered_data))
with col2:
    st.metric("Avg Price", f"${filtered_data['price'].mean():,.0f}")
with col3:
    st.metric("Avg Living Area", f"{filtered_data['livingAreaValue'].mean():,.0f} sqft")
with col4:
    st.metric("Avg Price/sqft", f"${(filtered_data['price']/filtered_data['livingAreaValue']).mean():.0f}")

# Add a reset button
if st.button("🔄 Reset Selection to First Property"):
    first_address = filtered_data["streetAddress"].sort_values().iloc[0]
    update_selected_address(first_address)