# -*- coding: utf-8 -*-
"""AI-Solutions Sales & Marketing Dashboard"""

# Absolute first line must be the page config
import streamlit as st
st.set_page_config(
    page_title="AI-Solutions Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now other imports can follow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import sys

# Handle auto-refresh import
AUTO_REFRESH_ENABLED = False
try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_ENABLED = True
except ImportError as e:
    st.sidebar.warning(f"Auto-refresh disabled: {str(e)}")

# Initialize data loading function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(uploaded_file=None):
    """Load and prepare the sales dataset"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=["OrderDate"])
        else:
            df = pd.read_csv("product_sales_web_data.csv", parse_dates=["OrderDate"])
        df['Month'] = df['OrderDate'].dt.month
        df['Month_Name'] = df['OrderDate'].dt.strftime('%b')
        df['OrderMonth'] = df['OrderDate'].dt.to_period('M')
        df['Year'] = df['OrderDate'].dt.year
        df['Date'] = df['OrderDate']
       
        # Add some marketing-specific columns for demo purposes
        if 'Campaign' not in df.columns:
            df['Campaign'] = df['Channel'].apply(lambda x: f"{x} Campaign")
        if 'Marketing_Spend' not in df.columns:
            df['Marketing_Spend'] = df['Revenue'] * 0.2  # Simulated marketing spend
        if 'Conversion_Rate' not in df.columns:
            df['Conversion_Rate'] = df['Revenue'] / df['Revenue'].max() * 100  # Simulated conversion rate
           
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe on error

# Initialize auto-refresh if available
if AUTO_REFRESH_ENABLED:
    st_autorefresh(interval=300000, key="data_refresh")

# Custom styling function
def apply_custom_styles():
    """Apply custom CSS styles to the dashboard"""
    st.markdown("""
<style>
        html, body, .stApp {
            height: 100vh;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            overflow: hidden;
        }
        .main, .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
        }
        footer { visibility: hidden; }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #4e73df;
            height: 80px;
            margin-bottom: 6px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .metric-title {
            font-size: 9px;
            color: #5a5c69;
            font-weight: 600;
            margin-bottom: 2px;
        }
        .metric-value {
            font-size: 14px;
            font-weight: 700;
            color: #2e59d9;
        }
        .metric-change {
            font-size: 8px;
            margin-top: 2px;
        }
        .positive { color: #1cc88a; }
        .negative { color: #e74a3b; }
        .metric-target {
            font-size: 8px;
            margin-top: 2px;
            color: #5a5c69;
        }
        .header {
            color: #2e59d9;
            font-size: 1.1rem;
            border-bottom: 1px solid #eee;
            padding-bottom: 6px;
            margin-bottom: 10px;
        }
        .stPlotlyChart {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background: white;
            padding: 10px;
            margin-bottom: 12px !important;
            height: 240px;
            width: 100% !important;
        }
        ::-webkit-scrollbar { display: none; }
        .sidebar .radio-group {
            margin-bottom: 20px;
        }
        .sidebar .radio-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
        }
</style>
    """, unsafe_allow_html=True)

     # --- Logout Button ---
    with st.sidebar:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

# Check if user is logged in
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login page function
# --- Login Page ---
def login_page():
    st.title("Login Page")
    # rest of the code
    """Display the login page"""
    apply_custom_styles()
    st.markdown("""
<div class="login-form">
<h2 style="text-align: center; color: #2e59d9;">🔐 SALES DASHBOARD LOGIN</h2>
<p style="text-align: center;">Please enter your credentials</p>
    """, unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "password":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Metric Display ---
def display_metric(title, current_value, prev_value=None, target=None, format_str="${:,.0f}", reverse_trend=False):
    if prev_value is not None:
        change = ((current_value - prev_value) / prev_value * 100) if prev_value != 0 else 0
        symbol = "▲" if change >= 0 else "▼"
        color_class = "positive" if change >= 0 else "negative"
        if reverse_trend:
            color_class = "negative" if change >= 0 else "positive"
            symbol = "▼" if change >= 0 else "▲"
        change_text = f"{symbol} {abs(change):.1f}% vs PY"
    else:
        change_text = ""
        color_class = ""
   
    if target is not None:
        target_status = current_value / target * 100
        icon = "✅" if target_status >= 100 else "⚠️" if target_status >= 90 else "❌"
        target_text = f"<div class='metric-target'>{icon} {target_status:.0f}% of target</div>"
    else:
        target_text = ""
   
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{format_str.format(current_value)}</div>
        {f"<div class='metric-change {color_class}'>{change_text}</div>" if prev_value is not None else ""}
        {target_text}
    </div>
    """, unsafe_allow_html=True)

# --- CHART FUNCTIONS ---
def create_sales_trend_chart(df):
    trend = df.groupby("OrderMonth")['Revenue'].sum().reset_index()
    trend['Target'] = 130000000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend['OrderMonth'].astype(str),
        y=trend['Revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#4e73df'),
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=trend['OrderMonth'].astype(str),
        y=trend['Target'],
        mode='lines',
        name='Target',
        line=dict(color='orange', dash='dash'),
        hovertemplate="<b>%{x}</b><br>Target: $%{y:,.0f}<extra></extra>"
    ))
   
    # Add range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),

            type="date"
        ),
        title="Monthly Revenue Trend vs Target",
        xaxis_title="Month",
        yaxis_title="Revenue ($)",
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    return fig

def create_sentiment_pie(df):
    sentiment = pd.cut(df['Satisfactory_Score'], bins=[0, 2.5, 4, 5], labels=['Negative', 'Neutral', 'Positive'])
    sentiment_counts = sentiment.value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.pie(
        sentiment_counts,
        names='Sentiment',
        values='Count',
        title='Customer Sentiment Distribution',
        color='Sentiment',
        color_discrete_map={'Negative':'#e74a3b', 'Neutral':'#f6c23e', 'Positive':'#1cc88a'},
        hole=0.3
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        uniformtext_minsize=12,
        uniformtext_mode='hide'
    )
    return fig

def create_campaign_chart(df):
    campaign_perf = df.groupby('Channel')['Revenue'].sum().reset_index()
    fig = px.bar(
        campaign_perf,
        x='Channel',
        y='Revenue',
        title='Revenue by Campaign Channel',
        color='Channel',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Channel",
        yaxis_title="Revenue ($)",
        height=240,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_leaderboard(df):
    leaderboard = df.groupby('Sales_Representative').agg({
        'Revenue': 'sum',
        'Product': 'count'
    }).rename(columns={'Product': 'Total_Orders'}).reset_index()
    leaderboard['Avg_Deal'] = leaderboard['Revenue'] / leaderboard['Total_Orders']
    leaderboard = leaderboard.sort_values(by='Revenue', ascending=False)
   
    fig = px.bar(
        leaderboard,
        x='Sales_Representative',
        y='Revenue',
        hover_data=['Total_Orders', 'Avg_Deal'],
        title='Top Performing Sales Representatives',
        color='Sales_Representative',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<br>Total Orders: %{customdata[0]}<br>Avg Deal: $%{customdata[1]:,.0f}<extra></extra>"
    )
    fig.update_layout(
        xaxis_title="Sales Representative",
        yaxis_title="Revenue ($)",
        height=240,
        margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_geographic_heatmap(df):
    region_map = df.groupby("Country")["Revenue"].sum().reset_index()
    fig = px.choropleth(
        region_map,
        locations="Country",
        locationmode="country names",
        color="Revenue",
        color_continuous_scale="Blues",
        title="🌍 Global Revenue by Country",
        hover_name="Country",
        hover_data={"Revenue": ":$,.0f"}
    )

    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",  # ✅ Ensures full-world map zoom
            lataxis_showgrid=True,
            lonaxis_showgrid=True
        )
    )
    return fig

def create_performance_gauge(value, title, target=100):
    if value >= target * 1.1:
        gauge_color = "#1cc88a"
        performance_text = "Exceeding Target"
    elif value >= target * 0.9:
        gauge_color = "#4e73df"
        performance_text = "Meeting Target"
    else:
        gauge_color = "#e74a3b"
        performance_text = "Below Target"
   
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        delta = {'reference': target, 'increasing': {'color': "#1cc88a"}, 'decreasing': {'color': "#e74a3b"}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"{title}<br><span style='font-size:0.6em;color:gray'>{performance_text}</span>",  
            'font': {'size': 12}  
        },
        gauge = {
            'axis': {'range': [None, target * 1.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color, 'thickness': 0.25},  
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, target*0.7], 'color': "#f8f9fa"},
                {'range': [target*0.7, target*0.9], 'color': "#f6c23e"},
                {'range': [target*0.9, target*1.1], 'color': "#4e73df"},
                {'range': [target*1.1, target*1.5], 'color': "#1cc88a"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},  
                'thickness': 0.7,
                'value': target
            }
        }
    ))
   
    fig.update_layout(
        height=180,  
        margin=dict(l=15, r=15, t=60, b=15),  
        font={'family': "Arial", 'color': "darkblue", 'size': 10}  
    )
    return fig

# Marketing-specific charts
def create_marketing_spend_chart(df):
    spend = df.groupby('Campaign').agg({
        'Marketing_Spend': 'sum',
        'Revenue': 'sum'
    }).reset_index()
    spend['ROI'] = (spend['Revenue'] - spend['Marketing_Spend']) / spend['Marketing_Spend'] * 100
   
    fig = px.bar(
        spend,
        x='Campaign',
        y='Marketing_Spend',
        title='Marketing Spend by Campaign',
        color='ROI',
        color_continuous_scale='Bluered',
        hover_data=['Revenue', 'ROI']
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.0f}<br>Revenue: $%{customdata[0]:,.0f}<br>ROI: %{customdata[1]:.1f}%<extra></extra>"
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_conversion_funnel(df):
    funnel = df.groupby('Campaign').agg({
        'Marketing_Spend': 'sum',
        'Revenue': 'sum',
        'OrderDate': 'count'
    }).rename(columns={'OrderDate': 'Conversions'}).reset_index()
    funnel['Conversion_Rate'] = funnel['Conversions'] / funnel['Conversions'].sum() * 100
   
    fig = px.funnel(
        funnel,
        x='Conversion_Rate',
        y='Campaign',
        title='Conversion Funnel by Campaign',
        color='Campaign',
        hover_data=['Marketing_Spend', 'Revenue']
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Conversion Rate: %{x:.1f}%<br>Spend: $%{customdata[0]:,.0f}<br>Revenue: $%{customdata[1]:,.0f}<extra></extra>"
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def create_roi_trend_chart(df):
    # Safely calculate Cost if missing
    if 'Cost' not in df.columns:
        df['Cost'] = df['Revenue'] - df['Profit']

    # Now calculate ROI
    df['ROI'] = df['Revenue'] / df['Cost'] * 100

    # Group ROI trend by month
    trend = df.groupby('OrderMonth')['ROI'].mean().reset_index()
    trend['OrderMonth'] = trend['OrderMonth'].astype(str)

    fig = px.line(
        trend,
        x='OrderMonth',
        y='ROI',
        title='📊 Monthly ROI Trend',
        markers=True
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis_title="ROI (%)",
        xaxis_title="Month"
    )
    return fig



#----FILTERS----
def apply_filters(df):
    with st.sidebar:
        # Section selector
        st.markdown("<div class='radio-group'>", unsafe_allow_html=True)
        section = st.radio("Select Section", ["Sales", "Marketing"], key="section_selector")
        st.markdown("</div>", unsafe_allow_html=True)
       
        st.header("🔍 Filters")
       
        # Data upload/download
        st.subheader("Data Management")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.success("File uploaded successfully!")
       
        if st.button("Download Current Data"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name="sales_data.csv",
                mime="text/csv"
            )

        # Date range picker
        st.subheader("Date Range")
        min_date = df['OrderDate'].min().date()
        max_date = df['OrderDate'].max().date()
        date_range = st.date_input(
            "Select date range:",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        # Year selection
        year_options = sorted(df['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Year", year_options)

        # Region selection
        region_options = df['Region'].unique()
        selected_regions = st.multiselect("Select Region(s)", region_options, default=region_options)

        # Category selection
        category_options = df['Category'].unique()
        selected_categories = st.multiselect("Select Product Category(s)", category_options, default=category_options)

    # Apply filters
    if len(date_range) == 2:
        filtered = df[
            (df['OrderDate'].dt.date >= date_range[0]) &
            (df['OrderDate'].dt.date <= date_range[1]) &
            (df['Year'] == selected_year) &
            (df['Region'].isin(selected_regions)) &
            (df['Category'].isin(selected_categories))
        ]
    else:
        filtered = df[
            (df['Year'] == selected_year) &
            (df['Region'].isin(selected_regions)) &
            (df['Category'].isin(selected_categories))
        ]

    return filtered, selected_year, section

# --- SALES DASHBOARD ---
def sales_dashboard(filtered_data, selected_year):
    st.markdown("<h1 style='text-align:center;'> AI-Solutions Sales Dashboard</h1>", unsafe_allow_html=True)
   
    current_month = filtered_data[filtered_data['OrderDate'].dt.month == datetime.now().month]
    monthly_revenue = current_month["Revenue"].sum()

    tab1, tab2 = st.tabs(["👥 Team View", "🙋 Individual View"])

    # -------------------- TEAM VIEW --------------------
    with tab1:
        # --- KPI Section ---
        prev_year = selected_year - 1
        previous_data = df[df['Year'] == prev_year]
        previous_month = previous_data[previous_data['OrderDate'].dt.month == datetime.now().month]

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        with col_kpi1:
            display_metric("MONTHLY REVENUE", current_month["Revenue"].sum(),
                         previous_month["Revenue"].sum(), target=100000000)

        with col_kpi2:
            display_metric("TOTAL ORDERS", len(current_month),
                         len(previous_month), target=1000)

        with col_kpi3:
            display_metric("AVG ORDER VALUE", current_month["Revenue"].mean(),
                         previous_month["Revenue"].mean())

        with col_kpi4:
            if not current_month.empty:
                top_product = current_month.groupby("Product")["Revenue"].sum().idxmax()
                top_value = current_month.groupby("Product")["Revenue"].sum().max()
                display_metric("TOP PRODUCT", top_value, None,
                             format_str=f"{top_product}: ${{:,.0f}}")
            else:
                display_metric("TOP PRODUCT", 0, None, format_str="No Data")

        # --- 2x2 Layout: 2 Gauges + 2 Graphs ---
        col1, col2 = st.columns(2)
        with col1:
            monthly_revenue = current_month["Revenue"].sum()
            st.plotly_chart(create_performance_gauge(
                monthly_revenue/5000000*100,
                "REVENUE PERFORMANCE",
                100),
                use_container_width=True
            )

        with col2:
            monthly_orders = len(current_month)
            st.plotly_chart(create_performance_gauge(
                monthly_orders/1000*100,
                "ORDERS PERFORMANCE",
                100),
                use_container_width=True
            )

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_sales_trend_chart(filtered_data),
                          use_container_width=True)

        with col4:
            st.plotly_chart(create_geographic_heatmap(filtered_data),
                          use_container_width=True)



# -------------------- INDIVIDUAL VIEW --------------------
    with tab2:
        reps = sorted(filtered_data["Sales_Representative"].dropna().unique())
        selected_rep = st.selectbox("Select Sales Representative", reps, key="rep_selector")

        rep_data = filtered_data[filtered_data["Sales_Representative"] == selected_rep]
        rep_month = rep_data[rep_data['OrderDate'].dt.month == datetime.now().month]

        # KPIs
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)

        with col_i1:
            display_metric("MONTHLY REVENUE", rep_month["Revenue"].sum())

        with col_i2:
            display_metric("TOTAL ORDERS", len(rep_month))

        with col_i3:
            avg_value = rep_month["Revenue"].mean() if not rep_month.empty else 0
            display_metric("AVG ORDER VALUE", avg_value)

        with col_i4:
            if not rep_month.empty:
                top_prod = rep_month.groupby("Product")["Revenue"].sum().idxmax()
                top_val = rep_month.groupby("Product")["Revenue"].sum().max()
                display_metric("TOP PRODUCT", top_val, None,
                              format_str=f"{top_prod}: ${{:,.0f}}")
            else:
                display_metric("TOP PRODUCT", 0, None, format_str="No Data")

        # Charts for Individual View
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sales_trend_chart(rep_data),
                          use_container_width=True)

        with col2:
            st.plotly_chart(create_sentiment_pie(rep_data),
                          use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            top_rep_products = rep_data.groupby("Product")["Revenue"].sum().nlargest(5).reset_index()
            fig_top = px.bar(
                top_rep_products,
                x="Product",
                y="Revenue",
                title="Top Products (Rep)",
                color='Product',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_top.update_traces(
                hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>"
            )
            fig_top.update_layout(
                xaxis_title="Product",
                yaxis_title="Revenue ($)",
                height=240,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_top, use_container_width=True)

        with col4:
            st.plotly_chart(create_campaign_chart(rep_data),
                          use_container_width=True)
           
 # NEW EXPANDABLE LEADERBOARD
        with st.expander("🏅 View Sales Leaderboard (All Reps)", expanded=False):
            st.plotly_chart(create_leaderboard(filtered_data), use_container_width=True)

# --- MARKETING DASHBOARD ---
def marketing_dashboard(filtered_data, selected_year):
    st.markdown("<h1 style='text-align:center;'> AI-Solutions Marketing Dashboard</h1>", unsafe_allow_html=True)
   
    current_month = filtered_data[filtered_data['OrderDate'].dt.month == datetime.now().month]
    prev_year = selected_year - 1
    previous_data = df[df['Year'] == prev_year]
    previous_month = previous_data[previous_data['OrderDate'].dt.month == datetime.now().month]

    tab1, = st.tabs(["👥 Campaign Overview"])

    # -------------------- CAMPAIGN OVERVIEW --------------------
    with tab1:
        # --- KPI Section ---
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

        with col_kpi1:
            display_metric("MARKETING SPEND", current_month["Marketing_Spend"].sum(),
                         previous_month["Marketing_Spend"].sum(), target=15000000)

        with col_kpi2:
            roi = (current_month["Revenue"].sum() - current_month["Marketing_Spend"].sum()) / current_month["Marketing_Spend"].sum() * 100
            prev_roi = (previous_month["Revenue"].sum() - previous_month["Marketing_Spend"].sum()) / previous_month["Marketing_Spend"].sum() * 100 if previous_month["Marketing_Spend"].sum() > 0 else 0
            display_metric("ROI", roi, prev_roi, target=100, format_str="{:.1f}%")

        with col_kpi3:
            display_metric("CONVERSION RATE", current_month["Conversion_Rate"].mean(),
                         previous_month["Conversion_Rate"].mean(), format_str="{:.1f}%")

        with col_kpi4:
            if not current_month.empty:
                top_campaign = current_month.groupby("Campaign")["Revenue"].sum().idxmax()
                top_value = current_month.groupby("Campaign")["Revenue"].sum().max()
                display_metric("TOP CAMPAIGN", top_value, None,
                             format_str=f"{top_campaign}: ${{:,.0f}}")
            else:
                display_metric("TOP CAMPAIGN", 0, None, format_str="No Data")

        # --- Charts ---
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_marketing_spend_chart(filtered_data),
                          use_container_width=True)

        with col2:
            st.plotly_chart(create_conversion_funnel(filtered_data),
                          use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_roi_trend_chart(filtered_data),
                          use_container_width=True)

        with col4:
            st.plotly_chart(create_campaign_chart(filtered_data),
                          use_container_width=True)



# --- Main Dashboard Layout ---
def main_dashboard():
    apply_custom_styles()
   
    # --- Apply filters ---
    filtered_data, selected_year, section = apply_filters(df)
   
    if section == "Sales":
        sales_dashboard(filtered_data, selected_year)
    else:
        marketing_dashboard(filtered_data, selected_year)

    # Last updated timestamp
    st.sidebar.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

df = load_data()

# --- Entry Point ---
if __name__ == "__main__":
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()