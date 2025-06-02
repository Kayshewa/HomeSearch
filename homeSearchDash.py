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


# ========== INTERACTIVE SPREADSHEET TABLE ==========
st.subheader("📋 Property Data Table")

# Prepare data for the table - select key columns for better display
table_columns = [
    'streetAddress', 'price', 'zestimate', 'bedrooms', 'bathrooms', 
    'livingAreaValue', 'yearBuilt', 'monthlyHoaFee', 'homeStatus', 'zipcode'
]

# Create display DataFrame with only existing columns
display_columns = [col for col in table_columns if col in filtered_data.columns]
table_data = filtered_data[display_columns].copy()

# Format the data for better display
if 'price' in table_data.columns:
    table_data['price'] = table_data['price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
if 'zestimate' in table_data.columns:
    table_data['zestimate'] = table_data['zestimate'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
if 'livingAreaValue' in table_data.columns:
    table_data['livingAreaValue'] = table_data['livingAreaValue'].apply(lambda x: f"{x:,.0f} sqft" if pd.notna(x) else "N/A")
if 'monthlyHoaFee' in table_data.columns:
    table_data['monthlyHoaFee'] = table_data['monthlyHoaFee'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "No Fee")

# Rename columns for better display
column_renames = {
    'streetAddress': 'Street Address',
    'price': 'Listed Price',
    'zestimate': 'Zestimate',
    'bedrooms': 'Beds',
    'bathrooms': 'Baths',
    'livingAreaValue': 'Living Area',
    'yearBuilt': 'Year Built',
    'monthlyHoaFee': 'Monthly HOA',
    'homeStatus': 'Status',
    'zipcode': 'ZIP Code'
}

table_data = table_data.rename(columns=column_renames)

# Find the index of the currently selected property
selected_index = None
if st.session_state.selected_address in filtered_data['streetAddress'].values:
    selected_index = filtered_data[filtered_data['streetAddress'] == st.session_state.selected_address].index[0]
    # Convert to position in filtered data
    selected_index = filtered_data.index.get_loc(selected_index)

# Create the interactive dataframe
st.write("**Click on any row to select that property:**")

# Use st.dataframe with selection
event = st.dataframe(
    table_data,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="property_table"
)

# Handle row selection
if event and len(event.selection.rows) > 0:
    selected_row_index = event.selection.rows[0]
    
    # Get the street address from the filtered data (not the display table)
    selected_street_address = filtered_data.iloc[selected_row_index]['streetAddress']
    
    # Update the selected address
    update_selected_address(selected_street_address)

# Add table navigation controls
st.write("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("⬆️ First Property", key="table_first"):
        first_address = filtered_data["streetAddress"].sort_values().iloc[0]
        update_selected_address(first_address)

with col2:
    if st.button("⬅️ Previous Property", key="table_prev"):
        addresses = filtered_data["streetAddress"].sort_values().tolist()
        if st.session_state.selected_address in addresses:
            current_idx = addresses.index(st.session_state.selected_address)
            if current_idx > 0:
                update_selected_address(addresses[current_idx - 1])

with col3:
    if st.button("➡️ Next Property", key="table_next"):
        addresses = filtered_data["streetAddress"].sort_values().tolist()
        if st.session_state.selected_address in addresses:
            current_idx = addresses.index(st.session_state.selected_address)
            if current_idx < len(addresses) - 1:
                update_selected_address(addresses[current_idx + 1])

with col4:
    if st.button("⬇️ Last Property", key="table_last"):
        last_address = filtered_data["streetAddress"].sort_values().iloc[-1]
        update_selected_address(last_address)

# Show current selection info
current_row = filtered_data[filtered_data['streetAddress'] == st.session_state.selected_address]
if not current_row.empty:
    row_position = filtered_data.index.get_loc(current_row.index[0]) + 1
    total_rows = len(filtered_data)
    st.info(f"📍 **Selected:** {st.session_state.selected_address} (Row {row_position} of {total_rows})")

# Add option to export filtered data
st.write("---")
col1, col2 = st.columns(2)

with col1:
    # Export to CSV
    csv_data = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_data,
        file_name=f"filtered_properties_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col2:
    # Show/hide all columns toggle
    if st.button("👁️ Toggle All Columns View", key="toggle_columns"):
        st.session_state.show_all_columns = not getattr(st.session_state, 'show_all_columns', False)

# Optional: Show all columns view
if getattr(st.session_state, 'show_all_columns', False):
    st.write("**Full Data View (All Columns):**")
    full_table_event = st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="full_property_table"
    )
    
    # Handle selection from full table
    if full_table_event and len(full_table_event.selection.rows) > 0:
        selected_row_index = full_table_event.selection.rows[0]
        selected_street_address = filtered_data.iloc[selected_row_index]['streetAddress']
        update_selected_address(selected_street_address)

# Function to save changes back to Google Sheets
def save_changes_to_sheet(updated_df, worksheet):
    """
    Save the updated dataframe back to Google Sheets
    """
    # Clear the existing data (except headers)
    worksheet.clear()
    
    # Create a clean copy of the dataframe for saving
    clean_df = updated_df.copy()
    
    # Handle ALL Int64 columns that might have NA values
    for col in clean_df.columns:
        if clean_df[col].dtype == 'Int64':
            # Convert Int64 columns to string, replacing <NA> with empty string
            clean_df[col] = clean_df[col].astype('string').fillna('')
        elif clean_df[col].dtype == 'Float64':
            # Handle Float64 columns similarly
            clean_df[col] = clean_df[col].astype('string').fillna('')
        else:
            # Handle other columns - convert to string and clean problematic values
            clean_df[col] = clean_df[col].astype(str).replace(['<NA>', 'nan', 'None', 'NaT'], '')
    
    # Convert dataframe to list of lists for upload
    # Include headers
    headers = clean_df.columns.tolist()
    values = [headers] + clean_df.values.tolist()
    
    # Update the sheet
    worksheet.update(values, value_input_option='USER_ENTERED')

# ========== EDITABLE SPREADSHEET TABLE ==========
st.subheader("📋 Editable Property Data Table")

# Define the specific columns we want to edit
edit_columns = ['Comments', 'tripiApprove', 'kayshewaApprove', 'siteVisit', 'offer']

# Create a copy of filtered data for editing and clean it
editable_data = filtered_data.copy()
editable_data = sanitize_dataframe_for_streamlit(editable_data)
editable_data = editable_data.reset_index(drop=True)

# Add the edit columns if they don't exist, with empty string defaults
for col in edit_columns:
    if col not in editable_data.columns:
        editable_data[col] = ""  # All columns default to empty string

# Keep streetAddress as identifier and the edit columns
display_columns = ['streetAddress'] + edit_columns
editable_subset = editable_data[display_columns].copy()

# Ensure all columns are treated as text
for col in edit_columns:
    if col in editable_subset.columns:
        editable_subset[col] = editable_subset[col].astype(str).fillna("")

# Define column configurations for text editing experience
column_config = {
    "streetAddress": st.column_config.TextColumn(
        "Street Address",
        help="Property street address",
        disabled=True  # Don't allow editing the address as it's our key
    ),
    "Comments": st.column_config.TextColumn(
        "Comments",
        help="Additional notes or comments about this property",
        max_chars=500
    ),
    "tripiApprove": st.column_config.TextColumn(
        "Tripi Approve",
        help="Tripi's approval status (free text)",
        max_chars=100
    ),
    "kayshewaApprove": st.column_config.TextColumn(
        "Kayshewa Approve", 
        help="Kayshewa's approval status (free text)",
        max_chars=100
    ),
    "siteVisit": st.column_config.TextColumn(
        "Site Visit",
        help="Site visit status or notes (free text)",
        max_chars=100
    ),
    "offer": st.column_config.TextColumn(
        "Offer Made",
        help="Offer status or details (free text)",
        max_chars=100
    )
}

# Add editing mode toggle
col1, col2 = st.columns([3, 1])
with col1:
    st.write("**Edit property approval status and comments:**")
with col2:
    editing_enabled = st.toggle("Enable Editing", key="enable_editing")

if editing_enabled:
    st.warning("⚠️ **Editing Mode Active** - Changes will be saved to Google Sheets when you click 'Save Changes'")
    
    try:
        # Show editable dataframe with only the specified columns
        edited_df = st.data_editor(
            editable_subset,
            column_config=column_config,
            use_container_width=True,
            hide_index=True,
            key="editable_property_table"
        )
        
        # Handle row selection through session state
        if hasattr(st.session_state, 'editable_property_table'):
            selection = getattr(st.session_state.editable_property_table, 'selection', None)
            if selection and hasattr(selection, 'rows') and selection.rows:
                try:
                    selected_row_index = selection.rows[0]
                    if selected_row_index < len(edited_df):
                        selected_street_address = edited_df.iloc[selected_row_index]['streetAddress']
                        update_selected_address(selected_street_address)
                except (IndexError, KeyError):
                    pass  # Ignore selection errors
        
    except Exception as e:
        st.error(f"Error with editable table: {str(e)}")
        st.write("Falling back to read-only mode...")
        edited_df = st.dataframe(
            editable_subset,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="fallback_table"
        )
    
    # Check for changes and provide save functionality
    changes_detected = not edited_df.equals(editable_subset)
    
    if changes_detected:
        st.success("✅ Changes detected in the data!")
        
        # Show what changed
        with st.expander("View Changes"):
            # Find changed rows
            for idx, (orig_row, edit_row) in enumerate(zip(editable_subset.itertuples(), edited_df.itertuples())):
                if orig_row != edit_row:
                    st.write(f"**Row {idx + 1} - {edit_row.streetAddress}:**")
                    for col in editable_subset.columns:
                        if col != 'streetAddress':  # Skip the address column
                            orig_val = getattr(orig_row, col)
                            edit_val = getattr(edit_row, col)
                            if orig_val != edit_val:
                                st.write(f"  • {col}: `{orig_val}` → `{edit_val}`")
        
        # Save changes button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Save Changes to Google Sheets", key="save_changes", type="primary"):
                try:
                    # Merge the edited columns back into the full dataset
                    full_updated_data = existing_data.copy()
                    
                    # CRITICAL: Convert Int64 columns to string FIRST before any other operations
                    for col in edit_columns:
                        if col in full_updated_data.columns:
                            # Check if it's an Int64 column and convert to string
                            if full_updated_data[col].dtype == 'Int64':
                                # Convert Int64 to string, handling <NA> properly
                                full_updated_data[col] = full_updated_data[col].astype('string').fillna('')
                            else:
                                # For other types, convert to string and clean
                                full_updated_data[col] = full_updated_data[col].astype(str).replace(['<NA>', 'nan', 'None', 'NaT'], '')
                        else:
                            # Add new columns as string type
                            full_updated_data[col] = ""
                    
                    # Now safely update the specific rows that were edited
                    for idx, row in edited_df.iterrows():
                        street_addr = row['streetAddress']
                        # Find matching row in full dataset
                        mask = full_updated_data['streetAddress'] == street_addr
                        if mask.any():
                            # Update the edit columns for this property
                            for col in edit_columns:
                                # Ensure the value is a clean string
                                value = str(row[col]).replace('<NA>', '').replace('nan', '').replace('None', '').replace('NaT', '')
                                full_updated_data.loc[mask, col] = value
                    
                    # Save to Google Sheets
                    save_changes_to_sheet(full_updated_data, worksheet)
                    st.success("✅ Changes saved successfully!")
                    st.rerun()  # Refresh the app to show updated data
                except Exception as e:
                    st.error(f"❌ Error saving changes: {str(e)}")
                    st.write(f"Error details: {type(e).__name__}: {str(e)}")
                    # Add debug info
                    st.write("**Debug info:**")
                    st.write(f"Edited dataframe dtypes: {edited_df.dtypes.to_dict()}")
                    if 'full_updated_data' in locals():
                        st.write(f"Full data dtypes: {full_updated_data.dtypes.to_dict()}")
                        # Show any problematic values
                        for col in edit_columns:
                            if col in full_updated_data.columns:
                                unique_vals = full_updated_data[col].astype(str).unique()[:10]  # Show first 10 unique values
                                st.write(f"{col} sample values: {unique_vals}")
    
else:
    # Read-only mode with selection
    st.write("**Read-only mode** - Toggle 'Enable Editing' to make changes")
    
    selected_event = st.dataframe(
        editable_subset,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="readonly_property_table"
    )
    
    # Handle row selection in read-only mode
    if selected_event and len(selected_event.selection.rows) > 0:
        selected_row_index = selected_event.selection.rows[0]
        selected_street_address = editable_subset.iloc[selected_row_index]['streetAddress']
        update_selected_address(selected_street_address)

# Show summary of text entries (count non-empty values)
st.write("---")
st.subheader("📊 Status Summary")

if len(editable_subset) > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        tripi_entries = editable_subset['tripiApprove'].astype(str).str.strip().ne('').sum() if 'tripiApprove' in editable_subset.columns else 0
        st.metric("Tripi Entries", f"{tripi_entries}/{len(editable_subset)}")
    
    with col2:
        kayshewa_entries = editable_subset['kayshewaApprove'].astype(str).str.strip().ne('').sum() if 'kayshewaApprove' in editable_subset.columns else 0
        st.metric("Kayshewa Entries", f"{kayshewa_entries}/{len(editable_subset)}")
    
    with col3:
        site_visit_entries = editable_subset['siteVisit'].astype(str).str.strip().ne('').sum() if 'siteVisit' in editable_subset.columns else 0
        st.metric("Site Visit Entries", f"{site_visit_entries}/{len(editable_subset)}")
    
    with col4:
        offer_entries = editable_subset['offer'].astype(str).str.strip().ne('').sum() if 'offer' in editable_subset.columns else 0
        st.metric("Offer Entries", f"{offer_entries}/{len(editable_subset)}")

# Show properties with specific text values (example: look for "approved" or "yes")
if 'tripiApprove' in editable_subset.columns and 'kayshewaApprove' in editable_subset.columns:
    # Look for common approval indicators
    approval_indicators = ['approved', 'approve', 'yes', 'y', 'good', 'ok', 'okay', '✓', 'check']
    
    tripi_approved = editable_subset[
        editable_subset['tripiApprove'].astype(str).str.lower().str.contains('|'.join(approval_indicators), na=False)
    ]
    
    kayshewa_approved = editable_subset[
        editable_subset['kayshewaApprove'].astype(str).str.lower().str.contains('|'.join(approval_indicators), na=False)
    ]
    
    # Find properties approved by both (intersection)
    both_approved_addresses = set(tripi_approved['streetAddress']) & set(kayshewa_approved['streetAddress'])
    both_approved = editable_subset[editable_subset['streetAddress'].isin(both_approved_addresses)]
    
    if len(both_approved) > 0:
        st.success(f"🎉 **{len(both_approved)} Properties with Approval Indicators from Both:**")
        for _, prop in both_approved.iterrows():
            st.write(f"• {prop['streetAddress']} (Tripi: '{prop['tripiApprove']}', Kayshewa: '{prop['kayshewaApprove']}')")
    else:
        st.info("No properties have approval indicators from both parties yet.")

# Add bulk operations
st.write("---")
st.subheader("🔧 Bulk Operations")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Approval Data", key="export_approval"):
        csv_data = (edited_df if editing_enabled and 'edited_df' in locals() else editable_subset).to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"property_approvals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔄 Refresh from Google Sheets", key="refresh_data"):
        st.rerun()

with col3:
    if st.button("↩️ Revert All Changes", key="revert_changes"):
        if editing_enabled:
            st.rerun()

# Show current selection info
current_selection = editable_subset[editable_subset['streetAddress'] == st.session_state.selected_address]
if not current_selection.empty:
    row_position = editable_subset.index.get_loc(current_selection.index[0]) + 1
    total_rows = len(editable_subset)
    st.info(f"📍 **Selected:** {st.session_state.selected_address} (Row {row_position} of {total_rows})")
    
    # Show current status for selected property
    prop = current_selection.iloc[0]
    st.write("**Current Status:**")
    status_cols = st.columns(5)
    
    with status_cols[0]:
        st.write(f"**Comments:** {prop.get('Comments', 'None')}")
    with status_cols[1]:
        st.write(f"**Tripi:** {prop.get('tripiApprove', 'Empty')}")
    with status_cols[2]:
        st.write(f"**Kayshewa:** {prop.get('kayshewaApprove', 'Empty')}")
    with status_cols[3]:
        st.write(f"**Site Visit:** {prop.get('siteVisit', 'Empty')}")
    with status_cols[4]:
        st.write(f"**Offer Made:** {prop.get('offer', 'Empty')}")