import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
from io import BytesIO
import json
from datetime import datetime

# Import our model classes
from models.cost_model import AISaaSCostModel
from models.growth_model import SaaSGrowthModel
from models.financial_model import SaaSFinancialModel

# Import the run functions from app.py
from app import (
    run_integrated_financial_model,
    run_with_s_curve_profile,
    run_with_acceleration_strategy,
    run_with_year_by_year_strategy,
    run_with_monthly_pattern,
    optimize_for_breakeven,
    optimize_for_series_b,
    run_enterprise_first_strategy,
    run_regulatory_impact_strategy
)

# Set page config
st.set_page_config(
    page_title="AI SaaS Financial Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        border-radius: 6px 6px 0 0;
        padding-top: 12px;
        padding-bottom: 12px;
        padding-left: 20px;
        padding-right: 20px;
        margin-right: 4px;
        background-color: #f0f2f6;
        font-weight: 600;
        letter-spacing: 0.5px;
        min-width: 160px;
        text-align: center;
        box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e2e6;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
        box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
        font-weight: 700;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stSlider {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .report-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .parameter-section {
        background-color: #f1f8ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .segment-tab {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to create download links for dataframes
def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Helper function to get download link for a matplotlib figure
def get_figure_download_link(fig, filename, text):
    """Generates a link allowing a matplotlib figure to be downloaded as PNG"""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to create prettier metrics
def display_metric(label, value, suffix="", prefix=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# Session state management
if 'financial_model' not in st.session_state:
    st.session_state.financial_model = None
if 'revenue_model' not in st.session_state:
    st.session_state.revenue_model = None
if 'cost_model' not in st.session_state:
    st.session_state.cost_model = None
if 'optimization_results' not in st.session_state:
    st.session_state.optimization_results = None
if 'run_button_clicked' not in st.session_state:
    st.session_state.run_button_clicked = False
if 'models_ready' not in st.session_state:
    st.session_state.models_ready = False

def run_models_callback():
    st.session_state.run_button_clicked = True

def save_config_to_file(config, filename):
    """Save configuration to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    st.success(f"Configuration saved to {filename}")

def load_config_from_file(filename):
    """Load configuration from a JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File {filename} not found")
        return None

# Main app layout
st.title("AI SaaS Financial Model with Growth Strategies")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌱   Growth Parameters   ", 
    "📈   Baseline S-Curves   ",
    "💰   Cost Parameters   ", 
    "📊   Growth Strategies   ", 
    "🔍   Results & Charts   ", 
    "📑   Data Tables   ", 
    "📝   VC Report   "
])

# Tab 1: Growth Parameters
with tab1:
    st.header("Growth Model Parameters")
    
    # Basic Parameters
    with st.expander("Basic Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.strptime("2025-01-01", "%Y-%m-%d"), format="YYYY-MM-DD")
        with col2:
            projection_months = st.slider("Projection Months", min_value=12, max_value=120, value=72, step=12)
        with col3:
            initial_investment = st.number_input("Initial Investment ($)", min_value=1000000, max_value=50000000, value=20000000, step=1000000, format="%d")
    
    # Segment parameters
    segments = ['Enterprise', 'Mid-Market', 'SMB']
    
    st.subheader("Segment Parameters")
    segment_tabs = st.tabs(segments)
    
    # Initialize dictionaries for parameter storage
    initial_arr = {}
    initial_customers = {}
    contract_length = {}
    churn_rates = {}
    annual_price_increases = {}
    s_curve_params = {}
    
    for i, segment in enumerate(segments):
        with segment_tabs[i]:
            st.markdown(f"<div class='segment-tab'><h3>{segment} Segment</h3></div>", unsafe_allow_html=True)
            
            # Initial parameters
            col1, col2 = st.columns(2)
            with col1:
                initial_arr[segment] = st.number_input(
                    f"Initial ARR per {segment} Customer ($)", 
                    min_value=1000, 
                    max_value=500000, 
                    value=150000 if segment == 'Enterprise' else (48000 if segment == 'Mid-Market' else 12000),
                    step=1000,
                    key=f"initial_arr_{segment}"
                )
                
                initial_customers[segment] = st.number_input(
                    f"Initial {segment} Customers", 
                    min_value=0, 
                    max_value=100, 
                    value=2 if segment == 'Enterprise' else (1 if segment == 'Mid-Market' else 2),
                    step=1,
                    key=f"initial_customers_{segment}"
                )
                
            with col2:
                contract_length[segment] = st.slider(
                    f"{segment} Contract Length (years)", 
                    min_value=0.25, 
                    max_value=3.0, 
                    value=2.0 if segment == 'Enterprise' else (1.5 if segment == 'Mid-Market' else 1.0),
                    step=0.25,
                    key=f"contract_length_{segment}"
                )
                
                churn_rates[segment] = st.slider(
                    f"{segment} Annual Churn Rate (%)", 
                    min_value=1.0, 
                    max_value=50.0, 
                    value=8.0 if segment == 'Enterprise' else (12.0 if segment == 'Mid-Market' else 20.0),
                    step=0.5,
                    key=f"churn_rate_{segment}"
                ) / 100  # Convert to decimal
                
                annual_price_increases[segment] = st.slider(
                    f"{segment} Annual Price Increase (%)", 
                    min_value=0.0, 
                    max_value=20.0, 
                    value=5.0 if segment == 'Enterprise' else (4.0 if segment == 'Mid-Market' else 3.0),
                    step=0.5,
                    key=f"price_increase_{segment}"
                ) / 100  # Convert to decimal
            
            # S-curve parameters by year
            st.markdown("##### S-Curve Parameters by Year")
            
            s_curve_params[segment] = {}
            
            with st.container():
                year_tabs = st.tabs([f"Year {y}" for y in range(1, 7)])
                
                # Default values per segment and year
                default_values = {
                    'Enterprise': [
                        {'midpoint': 6, 'steepness': 0.5, 'max_monthly': 3},    # Year 1
                        {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 5},    # Year 2
                        {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 7},    # Year 3
                        {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 10},   # Year 4
                        {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 12},   # Year 5
                        {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 15},   # Year 6
                    ],
                    'Mid-Market': [
                        {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 8},    # Year 1
                        {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 12},   # Year 2
                        {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 18},   # Year 3
                        {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 25},   # Year 4
                        {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 30},   # Year 5
                        {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 35},   # Year 6
                    ],
                    'SMB': [
                        {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 15},   # Year 1
                        {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 25},   # Year 2
                        {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 40},   # Year 3
                        {'midpoint': 6, 'steepness': 1.0, 'max_monthly': 60},   # Year 4
                        {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 80},   # Year 5
                        {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 100},  # Year 6
                    ]
                }
                
                for y in range(1, 7):
                    with year_tabs[y-1]:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            midpoint = st.slider(
                                f"Midpoint (Month) - Year {y}", 
                                min_value=1, 
                                max_value=12, 
                                value=default_values[segment][y-1]['midpoint'],
                                key=f"{segment}_midpoint_y{y}"
                            )
                        with col2:
                            steepness = st.slider(
                                f"Steepness - Year {y}", 
                                min_value=0.1, 
                                max_value=2.0, 
                                value=default_values[segment][y-1]['steepness'],
                                step=0.1,
                                key=f"{segment}_steepness_y{y}"
                            )
                        with col3:
                            max_monthly = st.slider(
                                f"Max Monthly Customers - Year {y}", 
                                min_value=0, 
                                max_value=250 if segment == 'SMB' else (100 if segment == 'Mid-Market' else 50), 
                                value=default_values[segment][y-1]['max_monthly'],
                                step=1,
                                key=f"{segment}_max_monthly_y{y}"
                            )
                        
                        s_curve_params[segment][y] = {
                            'midpoint': midpoint,
                            'steepness': steepness,
                            'max_monthly': max_monthly
                        }
    
    # Seasonality
    st.subheader("Seasonality Factors")
    
    seasonality = {}
    with st.expander("Monthly Seasonality (1.0 = average)"):
        cols = st.columns(4)
        for i, month in enumerate(range(1, 13)):
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            default_values = [0.85, 0.95, 1.05, 1.0, 1.0, 1.15, 0.9, 0.85, 1.05, 1.1, 1.0, 1.1]
            
            with cols[i % 4]:
                seasonality[month] = st.slider(
                    f"{month_names[i]} Factor", 
                    min_value=0.5, 
                    max_value=1.5, 
                    value=default_values[i],
                    step=0.05,
                    key=f"seasonality_{month}"
                )
    
    # Save & Load Configuration
    st.subheader("Save/Load Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        save_filename = st.text_input("Save configuration filename", "growth_config.json")
        if st.button("Save Configuration"):
            # Collect all growth parameters
            growth_config = {
                'start_date': start_date.strftime("%Y-%m-%d"),
                'projection_months': projection_months,
                'segments': segments,
                'initial_arr': initial_arr,
                'initial_customers': initial_customers,
                'contract_length': contract_length,
                'churn_rates': churn_rates,
                'annual_price_increases': annual_price_increases,
                's_curve': s_curve_params,
                'seasonality': seasonality
            }
            save_config_to_file(growth_config, save_filename)
    
    with col2:
        load_filename = st.text_input("Load configuration filename", "growth_config.json")
        if st.button("Load Configuration"):
            loaded_config = load_config_from_file(load_filename)
            if loaded_config:
                st.success(f"Configuration loaded from {load_filename}")
                st.info("Please refresh the page to apply the loaded configuration.")
                # In a more complex app, we would update all UI elements here

# Tab 2: Baseline S-Curves
with tab2:
    st.header("Baseline S-Curve Configuration")
    
    st.write("""
    The baseline S-curve parameters define the underlying growth pattern before applying any multipliers 
    from the growth strategies. These parameters determine the shape and pace of customer acquisition for each segment.
    """)
    
    # Define default baseline S-curve values from app.py
    default_baseline = {
        'Enterprise': {
            1: {'midpoint': 6, 'steepness': 0.5, 'max_monthly': 3},
            2: {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 5},
            3: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 7},
            4: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 10},
            5: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 12},
            6: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 15},
        },
        'Mid-Market': {
            1: {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 8},
            2: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 12},
            3: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 18},
            4: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 25},
            5: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 30},
            6: {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 35},
        },
        'SMB': {
            1: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 15},
            2: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 25},
            3: {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 40},
            4: {'midpoint': 6, 'steepness': 1.0, 'max_monthly': 60},
            5: {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 80},
            6: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 100},
        }
    }
    
    # Store user-modified baseline values in session state
    if 'baseline_scurve' not in st.session_state:
        st.session_state.baseline_scurve = default_baseline.copy()
    
    # Create a tabbed interface for each segment
    segment_tabs = st.tabs(['Enterprise', 'Mid-Market', 'SMB'])
    
    # For each segment, create year-by-year S-curve parameter controls
    for i, segment in enumerate(['Enterprise', 'Mid-Market', 'SMB']):
        with segment_tabs[i]:
            st.subheader(f"{segment} Baseline S-Curve Parameters")
            
            st.write("""
            Configure the S-curve parameters that control customer acquisition:
            - **Midpoint**: Month within the year when growth is at half the maximum rate
            - **Steepness**: How rapidly growth accelerates/decelerates (higher = steeper S-curve)
            - **Max Monthly**: Maximum number of new customers that can be acquired in a month
            """)
            
            # Show a table of the current values
            current_values = []
            for year in range(1, 7):
                params = st.session_state.baseline_scurve[segment][year]
                current_values.append({
                    'Year': year,
                    'Midpoint (Month)': params['midpoint'],
                    'Steepness': params['steepness'],
                    'Max Monthly': params['max_monthly']
                })
            
            df = pd.DataFrame(current_values)
            st.table(df)
            
            # Create year-by-year editing sliders in an expander
            with st.expander(f"Edit {segment} S-Curve Parameters"):
                for year in range(1, 7):
                    st.markdown(f"##### Year {year}")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        midpoint = st.slider(
                            f"Midpoint - Y{year}", 
                            min_value=1,
                            max_value=12,
                            value=st.session_state.baseline_scurve[segment][year]['midpoint'],
                            key=f"baseline_{segment}_y{year}_midpoint"
                        )
                        st.session_state.baseline_scurve[segment][year]['midpoint'] = midpoint
                    
                    with col2:
                        steepness = st.slider(
                            f"Steepness - Y{year}", 
                            min_value=0.1,
                            max_value=2.0,
                            value=st.session_state.baseline_scurve[segment][year]['steepness'],
                            step=0.1,
                            key=f"baseline_{segment}_y{year}_steepness"
                        )
                        st.session_state.baseline_scurve[segment][year]['steepness'] = steepness
                    
                    with col3:
                        max_value = 50 if segment == 'Enterprise' else (100 if segment == 'Mid-Market' else 150)
                        max_monthly = st.slider(
                            f"Max Monthly - Y{year}", 
                            min_value=1,
                            max_value=max_value,
                            value=st.session_state.baseline_scurve[segment][year]['max_monthly'],
                            key=f"baseline_{segment}_y{year}_max_monthly"
                        )
                        st.session_state.baseline_scurve[segment][year]['max_monthly'] = max_monthly
            
            # Visualize the S-curve for this segment
            if st.checkbox(f"Visualize {segment} S-Curves", value=True):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for year in range(1, 7):
                    # Get parameters
                    params = st.session_state.baseline_scurve[segment][year]
                    midpoint = params['midpoint'] - 1  # 0-indexed
                    steepness = params['steepness']
                    max_monthly = params['max_monthly']
                    
                    # Generate the S-curve for this year
                    months = np.arange(12)
                    s_curve_values = [max_monthly / (1 + np.exp(-steepness * (month - midpoint))) for month in months]
                    
                    # Plot
                    ax.plot(
                        months + 1,  # Convert back to 1-indexed for display
                        s_curve_values,
                        marker='o',
                        label=f'Year {year}'
                    )
                
                ax.set_xlabel('Month of Year')
                ax.set_ylabel('New Customers')
                ax.set_title(f'{segment} Baseline S-Curves by Year')
                ax.set_xticks(range(1, 13))
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
    
    # Show how to use the modified baseline
    st.markdown("### Using Your Modified Baseline")
    st.write("""
    When you modify the baseline S-curve parameters above, they will be applied to all subsequent model runs.
    
    The growth strategy multipliers will be applied on top of these baseline values.
    For example, if you set the Enterprise Year 1 Max Monthly to 5 and then apply a 2.0x multiplier
    from a growth strategy, the effective Max Monthly will be 10.
    """)
    
    # Option to reset to defaults
    if st.button("Reset to Default Baseline", type="secondary"):
        st.session_state.baseline_scurve = default_baseline.copy()
        st.success("Baseline S-curve parameters reset to defaults!")
        st.rerun()
    
    # Option to export and import (save/load) baseline
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Baseline Parameters", type="secondary"):
            # Convert dictionary to JSON string
            baseline_json = json.dumps(st.session_state.baseline_scurve, indent=2)
            # Create a downloadable link
            b64 = base64.b64encode(baseline_json.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="baseline_scurve_params.json">Download Baseline Parameters</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        uploaded_file = st.file_uploader("Import Baseline Parameters", type="json")
        if uploaded_file is not None:
            try:
                loaded_baseline = json.load(uploaded_file)
                # Validate the structure to ensure it matches expected format
                valid = True
                for segment in ['Enterprise', 'Mid-Market', 'SMB']:
                    if segment not in loaded_baseline:
                        valid = False
                        break
                    for year in range(1, 7):
                        if year not in loaded_baseline[segment]:
                            valid = False
                            break
                        if not all(k in loaded_baseline[segment][year] for k in ['midpoint', 'steepness', 'max_monthly']):
                            valid = False
                            break
                
                if valid:
                    st.session_state.baseline_scurve = loaded_baseline
                    st.success("Baseline parameters loaded successfully!")
                    st.rerun()
                else:
                    st.error("Invalid baseline parameter format!")
            except Exception as e:
                st.error(f"Error loading baseline parameters: {str(e)}")

# Tab 3: Cost Parameters
with tab3:
    st.header("Cost Model Parameters")
    
    # COGS Parameters
    with st.expander("COGS (Cost of Goods Sold)", expanded=True):
        st.markdown("##### COGS as percentage of ARR")
        col1, col2 = st.columns(2)
        
        with col1:
            cloud_hosting = st.slider("Cloud Hosting (%)", min_value=1.0, max_value=30.0, value=18.0, step=0.5) / 100
            customer_support = st.slider("Customer Support (%)", min_value=1.0, max_value=20.0, value=8.0, step=0.5) / 100
        
        with col2:
            third_party_apis = st.slider("Third-Party APIs (%)", min_value=1.0, max_value=20.0, value=6.0, step=0.5) / 100
            professional_services = st.slider("Professional Services (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.5) / 100
    
    # Headcount Parameters
    with st.expander("Headcount & Salaries", expanded=True):
        headcount_tabs = st.tabs([
            "Engineering", "Product", "Sales", "Marketing", 
            "Customer Success", "G&A", "Research"
        ])
        
        # Initialize headcount dictionaries
        headcount = {}
        
        # Engineering tab
        with headcount_tabs[0]:
            st.markdown("##### Engineering Team")
            col1, col2 = st.columns(2)
            
            with col1:
                eng_starting = st.number_input("Starting Count", min_value=1, max_value=50, value=10, step=1)
                eng_salary = st.number_input("Average Salary ($)", min_value=50000, max_value=300000, value=160000, step=10000)
            
            with col2:
                st.markdown("##### Growth Factors by Year")
                eng_growth = {}
                for year in range(1, 7):
                    default_values = [1.5, 1.8, 1.6, 1.4, 1.3, 1.2]
                    eng_growth[year] = st.slider(
                        f"Year {year} Growth", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=default_values[year-1], 
                        step=0.1,
                        key=f"eng_growth_y{year}"
                    )
            
            headcount['engineering'] = {
                'starting_count': eng_starting,
                'growth_type': 'step',
                'growth_factors': eng_growth,
                'avg_salary': eng_salary
            }
        
        # Product tab
        with headcount_tabs[1]:
            st.markdown("##### Product Team")
            col1, col2 = st.columns(2)
            
            with col1:
                prod_starting = st.number_input("Starting Count", min_value=1, max_value=30, value=3, step=1, key="prod_count")
                prod_salary = st.number_input("Average Salary ($)", min_value=50000, max_value=300000, value=170000, step=10000, key="prod_salary")
            
            with col2:
                st.markdown("##### Growth Factors by Year")
                prod_growth = {}
                for year in range(1, 7):
                    default_values = [1.5, 1.7, 1.5, 1.4, 1.3, 1.2]
                    prod_growth[year] = st.slider(
                        f"Year {year} Growth", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=default_values[year-1], 
                        step=0.1,
                        key=f"prod_growth_y{year}"
                    )
            
            headcount['product'] = {
                'starting_count': prod_starting,
                'growth_type': 'step',
                'growth_factors': prod_growth,
                'avg_salary': prod_salary
            }
        
        # Sales tab
        with headcount_tabs[2]:
            st.markdown("##### Sales Team")
            col1, col2 = st.columns(2)
            
            with col1:
                sales_starting = st.number_input("Starting Count", min_value=1, max_value=50, value=4, step=1, key="sales_count")
                sales_salary = st.number_input("Average Base Salary ($)", min_value=50000, max_value=300000, value=180000, step=10000, key="sales_salary")
            
            with col2:
                st.markdown("##### Growth Factors by Year")
                sales_growth = {}
                for year in range(1, 7):
                    default_values = [2.0, 1.8, 1.6, 1.4, 1.3, 1.2]
                    sales_growth[year] = st.slider(
                        f"Year {year} Growth", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=default_values[year-1], 
                        step=0.1,
                        key=f"sales_growth_y{year}"
                    )
            
            headcount['sales'] = {
                'starting_count': sales_starting,
                'growth_type': 'step',
                'growth_factors': sales_growth,
                'avg_salary': sales_salary
            }
        
        # Marketing tab
        with headcount_tabs[3]:
            st.markdown("##### Marketing Team")
            col1, col2 = st.columns(2)
            
            with col1:
                mktg_starting = st.number_input("Starting Count", min_value=1, max_value=40, value=3, step=1, key="mktg_count")
                mktg_salary = st.number_input("Average Salary ($)", min_value=50000, max_value=250000, value=120000, step=10000, key="mktg_salary")
            
            with col2:
                st.markdown("##### Growth Factors by Year")
                mktg_growth = {}
                for year in range(1, 7):
                    default_values = [1.8, 1.7, 1.5, 1.4, 1.3, 1.2]
                    mktg_growth[year] = st.slider(
                        f"Year {year} Growth", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=default_values[year-1], 
                        step=0.1,
                        key=f"mktg_growth_y{year}"
                    )
            
            headcount['marketing'] = {
                'starting_count': mktg_starting,
                'growth_type': 'step',
                'growth_factors': mktg_growth,
                'avg_salary': mktg_salary
            }
        
        # Customer Success tab
        with headcount_tabs[4]:
            st.markdown("##### Customer Success Team")
            col1, col2 = st.columns(2)
            
            with col1:
                cs_starting = st.number_input("Starting Count", min_value=1, max_value=40, value=2, step=1, key="cs_count")
                cs_salary = st.number_input("Average Salary ($)", min_value=50000, max_value=200000, value=110000, step=10000, key="cs_salary")
            
            with col2:
                st.markdown("##### Growth Factors by Year")
                cs_growth = {}
                for year in range(1, 7):
                    default_values = [1.5, 1.7, 1.6, 1.4, 1.3, 1.2]
                    cs_growth[year] = st.slider(
                        f"Year {year} Growth", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=default_values[year-1], 
                        step=0.1,
                        key=f"cs_growth_y{year}"
                    )
            
            headcount['customer_success'] = {
                'starting_count': cs_starting,
                'growth_type': 'step',
                'growth_factors': cs_growth,
                'avg_salary': cs_salary
            }
        
        # G&A tab
        with headcount_tabs[5]:
            st.markdown("##### G&A Team (General & Administrative)")
            col1, col2 = st.columns(2)
            
            with col1:
                ga_starting = st.number_input("Starting Count", min_value=1, max_value=30, value=4, step=1, key="ga_count")
                ga_salary = st.number_input("Average Salary ($)", min_value=50000, max_value=200000, value=120000, step=10000, key="ga_salary")
            
            with col2:
                st.markdown("##### Growth Factors by Year")
                ga_growth = {}
                for year in range(1, 7):
                    default_values = [1.4, 1.6, 1.4, 1.3, 1.2, 1.1]
                    ga_growth[year] = st.slider(
                        f"Year {year} Growth", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=default_values[year-1], 
                        step=0.1,
                        key=f"ga_growth_y{year}"
                    )
            
            headcount['g_and_a'] = {
                'starting_count': ga_starting,
                'growth_type': 'step',
                'growth_factors': ga_growth,
                'avg_salary': ga_salary
            }
        
        # Research tab
        with headcount_tabs[6]:
            st.markdown("##### Research Team")
            col1, col2 = st.columns(2)
            
            with col1:
                research_starting = st.number_input("Starting Count", min_value=1, max_value=30, value=4, step=1, key="research_count")
                research_salary = st.number_input("Average Salary ($)", min_value=50000, max_value=350000, value=200000, step=10000, key="research_salary")
            
            with col2:
                st.markdown("##### Growth Factors by Year")
                research_growth = {}
                for year in range(1, 7):
                    default_values = [1.5, 1.7, 1.5, 1.4, 1.3, 1.2]
                    research_growth[year] = st.slider(
                        f"Year {year} Growth", 
                        min_value=1.0, 
                        max_value=3.0, 
                        value=default_values[year-1], 
                        step=0.1,
                        key=f"research_growth_y{year}"
                    )
            
            headcount['research'] = {
                'starting_count': research_starting,
                'growth_type': 'step',
                'growth_factors': research_growth,
                'avg_salary': research_salary
            }
    
    # Salary & Benefits
    with st.expander("Salary & Benefits", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            annual_increase = st.slider("Annual Salary Increase (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.5) / 100
            benefits_multiplier = st.slider("Benefits Multiplier", min_value=1.0, max_value=1.5, value=1.28, step=0.01)
        
        with col2:
            payroll_tax_rate = st.slider("Payroll Tax Rate (%)", min_value=5.0, max_value=15.0, value=9.0, step=0.5) / 100
            bonus_rate = st.slider("Annual Bonus Rate (%)", min_value=0.0, max_value=30.0, value=15.0, step=1.0) / 100
            equity_compensation = st.slider("Equity Compensation (% of Salary)", min_value=0.0, max_value=40.0, value=20.0, step=1.0) / 100
    
    # Marketing Expenses
    with st.expander("Marketing Expenses", expanded=True):
        st.markdown("##### Non-headcount marketing expenses (% of ARR)")
        col1, col2 = st.columns(2)
        
        with col1:
            paid_advertising = st.slider("Paid Advertising (%)", min_value=5.0, max_value=50.0, value=25.0, step=1.0) / 100
            content_creation = st.slider("Content Creation (%)", min_value=1.0, max_value=30.0, value=10.0, step=1.0) / 100
        
        with col2:
            events_and_pr = st.slider("Events & PR (%)", min_value=1.0, max_value=20.0, value=8.0, step=1.0) / 100
            partner_marketing = st.slider("Partner Marketing (%)", min_value=1.0, max_value=20.0, value=7.0, step=1.0) / 100
        
        st.markdown("##### Marketing Efficiency by Year (Lower values = more efficient)")
        cols = st.columns(6)
        marketing_efficiency = {}
        
        for i, year in enumerate(range(1, 7)):
            default_values = [1.0, 0.92, 0.85, 0.8, 0.75, 0.7]
            with cols[i]:
                marketing_efficiency[year] = st.slider(
                    f"Year {year}", 
                    min_value=0.5, 
                    max_value=1.5, 
                    value=default_values[i],
                    step=0.05,
                    key=f"mktg_eff_y{year}"
                )
    
    # Sales Expenses
    with st.expander("Sales Expenses", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            commission_rate = st.slider("Commission Rate (% of New ARR)", min_value=5.0, max_value=30.0, value=15.0, step=1.0) / 100
        
        with col2:
            tools_and_enablement = st.slider("Tools & Enablement (% of ARR)", min_value=1.0, max_value=10.0, value=5.0, step=0.5) / 100
    
    # R&D Expenses
    with st.expander("R&D Expenses", expanded=True):
        st.markdown("##### Non-headcount R&D expenses (% of ARR)")
        col1, col2 = st.columns(2)
        
        with col1:
            cloud_compute = st.slider("Cloud Compute for Training (%)", min_value=5.0, max_value=30.0, value=18.0, step=1.0) / 100
        
        with col2:
            research_tools = st.slider("Research Tools & Data (%)", min_value=5.0, max_value=20.0, value=12.0, step=1.0) / 100
            third_party_research = st.slider("Third-Party Research (%)", min_value=1.0, max_value=15.0, value=8.0, step=1.0) / 100
    
    # G&A Expenses
    with st.expander("G&A Expenses", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            office_facilities = st.number_input("Office & Facilities ($/month)", min_value=10000, max_value=200000, value=50000, step=5000)
            per_employee_office = st.number_input("Per Employee Office Cost ($/month)", min_value=500, max_value=5000, value=1500, step=100)
        
        with col2:
            software_tools = st.number_input("Software & Tools ($/employee/month)", min_value=200, max_value=3000, value=1000, step=100)
            legal_accounting = st.number_input("Legal & Accounting ($/month)", min_value=5000, max_value=100000, value=25000, step=5000)
            insurance = st.number_input("Insurance ($/month)", min_value=5000, max_value=50000, value=15000, step=1000)
    
    # One-time Expenses
    with st.expander("One-time Expenses", expanded=True):
        st.markdown("##### Major one-time and periodic expenses")
        
        one_time_expenses = []
        for i in range(10):  # Display 10 possible one-time expenses
            col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
            
            default_months = [3, 9, 15, 17, 21, 24, 27, 36, 41, 48]
            default_categories = ['office', 'marketing', 'software', 'research', 'marketing', 'office', 'legal', 'office', 'research', 'infrastructure']
            default_amounts = [750000, 500000, 350000, 1200000, 600000, 400000, 300000, 800000, 1500000, 1000000]
            default_descriptions = [
                'Office setup and expansion',
                'Major product launch campaign',
                'Enterprise software licenses',
                'Major AI model training run',
                'Industry conference sponsorship',
                'Office expansion',
                'IP protection and legal work',
                'New office location setup',
                'Advanced AI model development',
                'Major infrastructure upgrade'
            ]
            
            with col1:
                month = st.number_input(
                    f"Month #{i+1}", 
                    min_value=1, 
                    max_value=72, 
                    value=default_months[i] if i < len(default_months) else 6*(i+1),
                    step=1,
                    key=f"onetime_month_{i}"
                )
            
            with col2:
                category = st.selectbox(
                    f"Category #{i+1}",
                    ['office', 'marketing', 'software', 'research', 'legal', 'infrastructure', 'other'],
                    index=0 if i >= len(default_categories) else ['office', 'marketing', 'software', 'research', 'legal', 'infrastructure', 'other'].index(default_categories[i]),
                    key=f"onetime_cat_{i}"
                )
            
            with col3:
                amount = st.number_input(
                    f"Amount #{i+1} ($)", 
                    min_value=0, 
                    max_value=3000000,
                    value=default_amounts[i] if i < len(default_amounts) else 500000,
                    step=50000,
                    key=f"onetime_amount_{i}"
                )
            
            with col4:
                description = st.text_input(
                    f"Description #{i+1}",
                    value=default_descriptions[i] if i < len(default_descriptions) else f"One-time expense in month {6*(i+1)}",
                    key=f"onetime_desc_{i}"
                )
            
            # Only add non-zero expenses
            if amount > 0:
                one_time_expenses.append([month, category, amount, description])
    
    # Save & Load Configuration
    st.subheader("Save/Load Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        save_cost_filename = st.text_input("Save configuration filename", "cost_config.json")
        if st.button("Save Cost Configuration"):
            # Collect all cost parameters
            cost_config = {
                'start_date': start_date.strftime("%Y-%m-%d"),
                'projection_months': projection_months,
                'cogs': {
                    'cloud_hosting': cloud_hosting,
                    'customer_support': customer_support,
                    'third_party_apis': third_party_apis,
                    'professional_services': professional_services
                },
                'headcount': headcount,
                'salary': {
                    'annual_increase': annual_increase,
                    'benefits_multiplier': benefits_multiplier,
                    'payroll_tax_rate': payroll_tax_rate,
                    'bonus_rate': bonus_rate,
                    'equity_compensation': equity_compensation
                },
                'marketing_expenses': {
                    'paid_advertising': paid_advertising,
                    'content_creation': content_creation,
                    'events_and_pr': events_and_pr,
                    'partner_marketing': partner_marketing
                },
                'marketing_efficiency': marketing_efficiency,
                'sales_expenses': {
                    'commission_rate': commission_rate,
                    'tools_and_enablement': tools_and_enablement
                },
                'r_and_d_expenses': {
                    'cloud_compute_for_training': cloud_compute,
                    'research_tools_and_data': research_tools,
                    'third_party_research': third_party_research
                },
                'g_and_a_expenses': {
                    'office_and_facilities': office_facilities,
                    'per_employee_office_cost': per_employee_office,
                    'software_and_tools': software_tools,
                    'legal_and_accounting': legal_accounting,
                    'insurance': insurance
                },
                'one_time_expenses': {
                    'items': one_time_expenses
                }
            }
            
            save_config_to_file(cost_config, save_cost_filename)
    
    with col2:
        load_cost_filename = st.text_input("Load configuration filename", "cost_config.json")
        if st.button("Load Cost Configuration"):
            loaded_config = load_config_from_file(load_cost_filename)
            if loaded_config:
                st.success(f"Configuration loaded from {load_cost_filename}")
                st.info("Please refresh the page to apply the loaded configuration.")

# Tab 3: Growth Strategies
with tab3:
    st.header("Growth Strategy Selection")
    
    strategy_type = st.radio(
        "Select Growth Strategy Type",
        [
            "Standard S-Curve Profiles",
            "Acceleration/Deceleration Strategy",
            "Year-by-Year Custom Strategy",
            "Enterprise-First Strategy",
            "AI Regulation Impact Strategy",
            "Optimization for Breakeven",
            "Optimization for Series B"
        ]
    )
    
    if strategy_type == "Standard S-Curve Profiles":
        growth_profile = st.selectbox(
            "Select Growth Profile",
            ["baseline", "conservative", "aggressive", "hypergrowth"],
            index=0,
            format_func=lambda x: x.capitalize()
        )
        
        st.markdown("""
        - **Baseline**: Standard growth trajectory
        - **Conservative**: 30% slower growth than baseline (0.7x multiplier)
        - **Aggressive**: 50% faster growth than baseline (1.5x multiplier)
        - **Hypergrowth**: 150% faster growth than baseline (2.5x multiplier)
        """)
    
    elif strategy_type == "Acceleration/Deceleration Strategy":
        st.subheader("Acceleration/Deceleration Strategy")
        st.markdown("This strategy allows you to accelerate growth in early years and decelerate in later years.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_segments = st.multiselect(
                "Target Segments for Strategy",
                segments,
                default=segments
            )
            
            acceleration_years = st.multiselect(
                "Years to Accelerate Growth",
                list(range(1, 7)),
                default=[1, 2]
            )
            
            deceleration_years = st.multiselect(
                "Years to Decelerate Growth",
                list(range(1, 7)),
                default=[5, 6]
            )
        
        with col2:
            accel_multiplier = st.slider(
                "Acceleration Multiplier",
                min_value=1.1,
                max_value=5.0,
                value=2.0,
                step=0.1
            )
            
            decel_multiplier = st.slider(
                "Deceleration Multiplier",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1
            )
    
    elif strategy_type == "Year-by-Year Custom Strategy":
        st.subheader("Year-by-Year Custom Strategy")
        st.markdown("Set specific growth multipliers for each segment by year")
        
        # Initialize the segment year multipliers
        segment_year_multipliers = {}
        
        segment_tabs = st.tabs(segments)
        
        for i, segment in enumerate(segments):
            with segment_tabs[i]:
                st.markdown(f"##### {segment} Growth Multipliers")
                
                segment_year_multipliers[segment] = {}
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    for year in range(1, 3):
                        segment_year_multipliers[segment][year] = st.slider(
                            f"Year {year} Multiplier",
                            min_value=0.1,
                            max_value=5.0,
                            value=2.0 if segment == 'Enterprise' and year <= 2 else 1.0,
                            step=0.1,
                            key=f"{segment}_y{year}_multiplier"
                        )
                
                with col2:
                    for year in range(3, 5):
                        segment_year_multipliers[segment][year] = st.slider(
                            f"Year {year} Multiplier",
                            min_value=0.1,
                            max_value=5.0,
                            value=1.5 if segment == 'Mid-Market' and 3 <= year <= 4 else 1.0,
                            step=0.1,
                            key=f"{segment}_y{year}_multiplier"
                        )
                
                with col3:
                    for year in range(5, 7):
                        segment_year_multipliers[segment][year] = st.slider(
                            f"Year {year} Multiplier",
                            min_value=0.1,
                            max_value=5.0,
                            value=2.0 if segment == 'SMB' and year >= 5 else 0.8 if segment == 'Enterprise' and year >= 5 else 1.0,
                            step=0.1,
                            key=f"{segment}_y{year}_multiplier"
                        )
                
                # Display the strategy visually
                if st.checkbox(f"Show {segment} Strategy Visualization", value=False):
                    years = list(range(1, 7))
                    multipliers = [segment_year_multipliers[segment][y] for y in years]
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(years, multipliers, color='steelblue')
                    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
                    
                    for i, v in enumerate(multipliers):
                        ax.text(i+1, v+0.1, f"{v:.1f}x", ha='center')
                    
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Growth Multiplier')
                    ax.set_title(f'{segment} Growth Strategy')
                    ax.set_xticks(years)
                    ax.set_ylim(0, max(multipliers) + 0.5)
                    
                    st.pyplot(fig)
    
    elif strategy_type == "Enterprise-First Strategy":
        st.subheader("Enterprise-First Strategy")
        st.markdown("""
        This is a predefined strategy that:
        - Focuses on enterprise and mid-market in years 1-2
        - Shifts to mid-market and SMB in years 3-4
        - Takes a more balanced approach in years 5-6
        """)
        
        # Show the strategy data
        enterprise_first_strategy = {
            'Enterprise': {
                1: 2.0,  # Strong focus in year 1
                2: 1.8,  # Strong focus in year 2
                3: 1.4,  # Moderate focus in year 3
                4: 1.2,  # Less focus in year 4
                5: 1.0,  # Back to baseline in year 5
                6: 0.9,  # Slight deceleration in year 6
            },
            'Mid-Market': {
                1: 1.5,  # Good focus in year 1
                2: 1.7,  # Increased focus in year 2
                3: 1.8,  # Peak focus in year 3
                4: 1.6,  # Still strong in year 4
                5: 1.3,  # Moderate focus in year 5
                6: 1.1,  # Slight focus in year 6
            },
            'SMB': {
                1: 0.7,  # Low focus initially
                2: 0.8,  # Still low focus
                3: 1.3,  # Increased focus in year 3
                4: 1.7,  # Strong focus in year 4
                5: 2.0,  # Peak focus in year 5
                6: 2.2,  # Continued strong focus in year 6
            }
        }
        
        # Display as a table first
        strategy_df = pd.DataFrame({
            'Year': list(range(1, 7)),
            'Enterprise': [enterprise_first_strategy['Enterprise'][y] for y in range(1, 7)],
            'Mid-Market': [enterprise_first_strategy['Mid-Market'][y] for y in range(1, 7)],
            'SMB': [enterprise_first_strategy['SMB'][y] for y in range(1, 7)]
        })
        
        st.table(strategy_df.style.format('{:.1f}x'))
        
        # Display as a chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(1, 7))
        
        for segment in segments:
            y = [enterprise_first_strategy[segment][year] for year in range(1, 7)]
            ax.plot(x, y, marker='o', linewidth=2, label=segment)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xlabel('Year')
        ax.set_ylabel('Growth Multiplier')
        ax.set_title('Enterprise-First Growth Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
    elif strategy_type == "AI Regulation Impact Strategy":
        st.subheader("AI Regulation Impact Strategy")
        st.markdown("""
        ### Regulatory Impact on AI Adoption
        
        This strategy models the impact of AI regulations starting in 2026:
        
        - **Enterprise** customers lead adoption (years 1-3) due to:
          - Greater resources for regulatory compliance
          - Established legal teams to navigate regulations
          - Ability to invest in compliance frameworks
        
        - **Mid-Market** follows with moderate delay (years 2-4) as:
          - Compliance frameworks become more established
          - Third-party compliance solutions emerge
        
        - **SMB** adoption is significantly delayed (years 5-6) until:
          - Regulations stabilize and become predictable
          - Turnkey compliance solutions become affordable
          - Industry standards simplify implementation
        """)
        
        # Show the strategy data
        regulatory_impact_strategy = {
            'Enterprise': {
                1: 2.2,  # Very strong focus in year 1 (early adopters with resources for compliance)
                2: 2.0,  # Strong focus in year 2
                3: 1.6,  # Continued strong focus as regulations solidify
                4: 1.3,  # Moderate focus in year 4
                5: 1.1,  # Slightly above baseline in year 5
                6: 1.0,  # Baseline in year 6
            },
            'Mid-Market': {
                1: 1.7,  # Good focus in year 1
                2: 1.9,  # Increased focus in year 2 as mid-market follows enterprise
                3: 2.0,  # Peak focus in year 3 as mid-market adoption accelerates
                4: 1.8,  # Still strong in year 4
                5: 1.5,  # Moderate focus in year 5
                6: 1.3,  # Continued moderate focus in year 6
            },
            'SMB': {
                1: 0.3,  # Minimal focus initially due to regulatory barriers
                2: 0.4,  # Still very low focus due to compliance costs
                3: 0.6,  # Gradual increase as compliance frameworks become accessible
                4: 1.0,  # Reaching baseline as regulations stabilize
                5: 1.8,  # Accelerating as compliance becomes more standardized
                6: 2.5,  # Strong catch-up growth as barriers lower and solutions become turnkey
            }
        }
        
        # Display as a table first
        strategy_df = pd.DataFrame({
            'Year': list(range(1, 7)),
            'Enterprise': [regulatory_impact_strategy['Enterprise'][y] for y in range(1, 7)],
            'Mid-Market': [regulatory_impact_strategy['Mid-Market'][y] for y in range(1, 7)],
            'SMB': [regulatory_impact_strategy['SMB'][y] for y in range(1, 7)]
        })
        
        st.table(strategy_df.style.format('{:.1f}x'))
        
        # Display as a chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = list(range(1, 7))
        
        for segment in segments:
            y = [regulatory_impact_strategy[segment][year] for year in range(1, 7)]
            ax.plot(x, y, marker='o', linewidth=2, label=segment)
        
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xlabel('Year')
        ax.set_ylabel('Growth Multiplier')
        ax.set_title('AI Regulatory Impact Growth Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line showing when regulations begin
        ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.7)
        ax.text(1.5, ax.get_ylim()[1]*0.95, 'AI Regulations\nBegin', ha='center', va='top', 
                color='red', backgroundcolor='white', alpha=0.8)
        
        st.pyplot(fig)
    
    elif strategy_type == "Optimization for Breakeven":
        st.subheader("Optimize for Breakeven")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_month = st.slider(
                "Target Month to Achieve Breakeven",
                min_value=12,
                max_value=60,
                value=24,
                step=1
            )
            
            min_multiplier = st.slider(
                "Minimum Growth Multiplier",
                min_value=0.1,
                max_value=1.0,
                value=0.4,
                step=0.1
            )
        
        with col2:
            base_profile = st.selectbox(
                "Base Growth Profile",
                ["baseline", "conservative", "aggressive", "hypergrowth"],
                index=0,
                format_func=lambda x: x.capitalize()
            )
            
            max_multiplier = st.slider(
                "Maximum Growth Multiplier",
                min_value=1.1,
                max_value=5.0,
                value=3.0,
                step=0.1
            )
    
    elif strategy_type == "Optimization for Series B":
        st.subheader("Optimize for Series B Qualification")
        st.markdown("""
        Series B qualification typically requires:
        - $10M+ ARR
        - 100%+ YoY growth rate
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_month_sb = st.slider(
                "Target Month to Achieve Series B Criteria",
                min_value=18,
                max_value=60,
                value=36,
                step=1
            )
            
            min_multiplier_sb = st.slider(
                "Minimum Growth Multiplier",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                key="sb_min_mult"
            )
        
        with col2:
            target_arr = st.number_input(
                "Target ARR ($)",
                min_value=5000000,
                max_value=20000000,
                value=10000000,
                step=1000000
            )
            
            target_growth_rate = st.slider(
                "Target YoY Growth Rate (%)",
                min_value=50,
                max_value=200,
                value=100,
                step=10
            ) / 100
            
            max_multiplier_sb = st.slider(
                "Maximum Growth Multiplier",
                min_value=1.1,
                max_value=5.0,
                value=4.0,
                step=0.1,
                key="sb_max_mult"
            )

    # Run model button
    st.markdown("---")
    run_col1, run_col2 = st.columns([3, 1])
    
    with run_col1:
        st.markdown("### Ready to Run the Model?")
        st.markdown("Click the button to run the financial model with your selected parameters and growth strategy.")
    
    with run_col2:
        run_button = st.button("Run Financial Model", type="primary", on_click=run_models_callback)
    
    if st.session_state.run_button_clicked:
        with st.spinner("Running the financial model..."):
            # Construct the revenue configuration
            revenue_config = {
                'start_date': start_date.strftime("%Y-%m-%d"),
                'projection_months': projection_months,
                'segments': segments,
                'initial_arr': initial_arr,
                'initial_customers': initial_customers,
                'contract_length': contract_length,
                'churn_rates': churn_rates,
                'annual_price_increases': annual_price_increases,
                # Use customized baseline S-curve parameters if available, otherwise use the ones from the Growth Parameters tab
                's_curve': st.session_state.baseline_scurve if 'baseline_scurve' in st.session_state else s_curve_params,
                'seasonality': seasonality
            }
            
            # Construct the cost configuration
            cost_config = {
                'start_date': start_date.strftime("%Y-%m-%d"),
                'projection_months': projection_months,
                'cogs': {
                    'cloud_hosting': cloud_hosting,
                    'customer_support': customer_support,
                    'third_party_apis': third_party_apis,
                    'professional_services': professional_services
                },
                'headcount': headcount,
                'salary': {
                    'annual_increase': annual_increase,
                    'benefits_multiplier': benefits_multiplier,
                    'payroll_tax_rate': payroll_tax_rate,
                    'bonus_rate': bonus_rate,
                    'equity_compensation': equity_compensation
                },
                'marketing_expenses': {
                    'paid_advertising': paid_advertising,
                    'content_creation': content_creation,
                    'events_and_pr': events_and_pr,
                    'partner_marketing': partner_marketing
                },
                'marketing_efficiency': marketing_efficiency,
                'sales_expenses': {
                    'commission_rate': commission_rate,
                    'tools_and_enablement': tools_and_enablement
                },
                'r_and_d_expenses': {
                    'cloud_compute_for_training': cloud_compute,
                    'research_tools_and_data': research_tools,
                    'third_party_research': third_party_research
                },
                'g_and_a_expenses': {
                    'office_and_facilities': office_facilities,
                    'per_employee_office_cost': per_employee_office,
                    'software_and_tools': software_tools,
                    'legal_and_accounting': legal_accounting,
                    'insurance': insurance
                },
                'one_time_expenses': {
                    'items': one_time_expenses
                }
            }
            
            # Run the appropriate model based on the selected strategy
            try:
                if strategy_type == "Standard S-Curve Profiles":
                    st.session_state.financial_model, st.session_state.revenue_model, st.session_state.cost_model, st.session_state.optimization_results = run_with_s_curve_profile(
                        growth_profile=growth_profile,
                        initial_investment=initial_investment
                    )
                
                elif strategy_type == "Acceleration/Deceleration Strategy":
                    st.session_state.financial_model, st.session_state.revenue_model, st.session_state.cost_model, st.session_state.optimization_results = run_with_acceleration_strategy(
                        target_segments=target_segments,
                        acceleration_years=acceleration_years,
                        deceleration_years=deceleration_years,
                        accel_multiplier=accel_multiplier,
                        decel_multiplier=decel_multiplier,
                        initial_investment=initial_investment
                    )
                
                elif strategy_type == "Year-by-Year Custom Strategy":
                    st.session_state.financial_model, st.session_state.revenue_model, st.session_state.cost_model, st.session_state.optimization_results = run_with_year_by_year_strategy(
                        segment_year_multipliers=segment_year_multipliers,
                        initial_investment=initial_investment
                    )
                
                elif strategy_type == "Enterprise-First Strategy":
                    st.session_state.financial_model, st.session_state.revenue_model, st.session_state.cost_model, st.session_state.optimization_results = run_enterprise_first_strategy(
                        initial_investment=initial_investment
                    )
                
                elif strategy_type == "AI Regulation Impact Strategy":
                    st.session_state.financial_model, st.session_state.revenue_model, st.session_state.cost_model, st.session_state.optimization_results = run_regulatory_impact_strategy(
                        initial_investment=initial_investment
                    )
                
                elif strategy_type == "Optimization for Breakeven":
                    st.session_state.financial_model, st.session_state.revenue_model, st.session_state.cost_model, st.session_state.optimization_results = optimize_for_breakeven(
                        target_month=target_month,
                        growth_profile=base_profile,
                        initial_investment=initial_investment
                    )
                
                elif strategy_type == "Optimization for Series B":
                    st.session_state.financial_model, st.session_state.revenue_model, st.session_state.cost_model, st.session_state.optimization_results = optimize_for_series_b(
                        target_month=target_month_sb,
                        initial_investment=initial_investment
                    )
                
                st.session_state.models_ready = True
                st.success("Model run complete! Check the Results & Charts tab to view the results.")
                
            except Exception as e:
                st.error(f"Error running the model: {str(e)}")
                st.session_state.models_ready = False

# Tab 4: Results & Charts
with tab4:
    st.header("Results & Charts")
    
    if not st.session_state.models_ready:
        st.warning("Please run the model first from the Growth Strategies tab.")
    else:
        # Show key metrics
        st.subheader("Key Metrics")
        key_metrics = st.session_state.financial_model.get_key_metrics_table()
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics directly from model data instead of using the key_metrics table
        # This avoids potential formatting and indexing issues
        
        # Calculate additional metrics
        monthly_data = st.session_state.financial_model.get_monthly_data()
        annual_data = st.session_state.financial_model.get_annual_data()
        
        # Find breakeven month
        profitable_months = monthly_data[monthly_data['ebitda'] > 0]
        breakeven_month = profitable_months['month_number'].min() if len(profitable_months) > 0 else "Not reached"
        
        # Calculate max burn rate
        max_burn_rate = monthly_data['burn_rate'].max() / 1000000  # Convert to millions
        
        # Calculate terminal valuation (5x revenue)
        terminal_revenue = annual_data.iloc[-1]['annual_revenue']
        terminal_valuation_5x = terminal_revenue * 5 / 1000000  # Convert to millions
        
        # Calculate Year 6 growth rate
        if len(annual_data) >= 2:
            prev_revenue = annual_data.iloc[-2]['annual_revenue']
            current_revenue = annual_data.iloc[-1]['annual_revenue']
            if prev_revenue > 0:
                year6_growth = ((current_revenue / prev_revenue) - 1) * 100  # Convert to percentage
            else:
                year6_growth = 0
        else:
            year6_growth = 0
        
        # Get year 6 metrics directly from model data
        year6_arr = annual_data.iloc[-1]['year_end_arr'] / 1000000
        year6_ebitda = annual_data.iloc[-1]['annual_ebitda'] / 1000000
        year6_customers = int(annual_data.iloc[-1]['year_end_customers'])
        year6_ltv_cac = annual_data.iloc[-1]['annual_ltv_cac_ratio']
        
        with col1:
            display_metric("Terminal ARR", f"{year6_arr:.1f}", suffix="M", prefix="$")
            display_metric("Breakeven Month", f"{breakeven_month}")
        
        with col2:
            display_metric("ARR Growth Y6", f"{year6_growth:.1f}", suffix="%")
            display_metric("Terminal Valuation", f"{terminal_valuation_5x:.1f}", suffix="M", prefix="$")
        
        with col3:
            display_metric("Year 6 Customers", f"{year6_customers:,}")
            display_metric("LTV/CAC Ratio Y6", f"{year6_ltv_cac:.1f}")
        
        with col4:
            display_metric("Max Burn Rate", f"{max_burn_rate:.1f}", suffix="M/mo", prefix="$")
            display_metric("Year 6 EBITDA", f"{year6_ebitda:.1f}", suffix="M", prefix="$")
        
        # Show optimization results if available
        if st.session_state.optimization_results:
            st.subheader("Optimization Results")
            opt_results = st.session_state.optimization_results
            
            opt_col1, opt_col2 = st.columns(2)
            
            with opt_col1:
                st.markdown(f"**Target:** {opt_results['target'].capitalize()}")
                st.markdown(f"**Target Month:** {opt_results['target_month']}")
            
            with opt_col2:
                st.markdown(f"**Achieved Month:** {opt_results['achieved_month'] if opt_results['achieved_month'] else 'Not achieved'}")
                st.markdown(f"**Growth Multiplier:** {opt_results['growth_multiplier']:.2f}x")
        
        # Chart selection
        st.subheader("Charts")
        
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Financial Summary",
                "Break Even Analysis",
                "Runway and Capital",
                "Unit Economics",
                "Growth Curves",
                "Annual Metrics",
                "Customer Segment Shares",
                "Expense Breakdown",
                "Headcount Growth"
            ]
        )
        
        def display_chart_with_download(fig, filename):
            st.pyplot(fig)
            st.markdown(get_figure_download_link(fig, filename, "Download Chart"), unsafe_allow_html=True)
        
        if chart_type == "Financial Summary":
            fig = st.session_state.financial_model.plot_financial_summary(figsize=(12, 8))
            display_chart_with_download(fig, "financial_summary.png")
            
        elif chart_type == "Break Even Analysis":
            fig = st.session_state.financial_model.plot_break_even_analysis(figsize=(12, 8))
            display_chart_with_download(fig, "break_even_analysis.png")
            
        elif chart_type == "Runway and Capital":
            fig = st.session_state.financial_model.plot_runway_and_capital(figsize=(12, 8))
            display_chart_with_download(fig, "runway_and_capital.png")
            
        elif chart_type == "Unit Economics":
            fig = st.session_state.financial_model.plot_unit_economics(figsize=(12, 8))
            display_chart_with_download(fig, "unit_economics.png")
            
        elif chart_type == "Growth Curves":
            fig = st.session_state.revenue_model.plot_growth_curves(figsize=(12, 10), highlight_customizations=True)
            display_chart_with_download(fig, "growth_curves.png")
            
        elif chart_type == "Annual Metrics":
            fig = st.session_state.revenue_model.plot_annual_metrics(figsize=(12, 8))
            display_chart_with_download(fig, "annual_metrics.png")
            
        elif chart_type == "Customer Segment Shares":
            fig = st.session_state.revenue_model.plot_customer_segment_shares(figsize=(12, 6))
            display_chart_with_download(fig, "segment_shares.png")
            
        elif chart_type == "Expense Breakdown":
            fig = st.session_state.cost_model.plot_expense_breakdown(figsize=(12, 8))
            display_chart_with_download(fig, "expense_breakdown.png")
            
        elif chart_type == "Headcount Growth":
            fig = st.session_state.cost_model.plot_headcount_growth(figsize=(12, 8))
            display_chart_with_download(fig, "headcount_growth.png")

# Tab 5: Data Tables
with tab5:
    st.header("Financial Data Tables")
    
    if not st.session_state.models_ready:
        st.warning("Please run the model first from the Growth Strategies tab.")
    else:
        # Data table selection
        data_type = st.selectbox(
            "Select Data Table",
            [
                "Key Metrics",
                "Monthly Data",
                "Annual Data",
                "Growth Monthly Data",
                "Growth Annual Data",
                "Cost Monthly Data",
                "Cost Annual Data"
            ]
        )
        
        def display_table_with_download(df, filename, description=None):
            if description:
                st.markdown(f"**{description}**")
            
            st.dataframe(df)
            st.markdown(get_table_download_link(df, filename, "Download CSV"), unsafe_allow_html=True)
        
        if data_type == "Key Metrics":
            key_metrics = st.session_state.financial_model.get_key_metrics_table()
            display_table_with_download(key_metrics, "key_metrics.csv", "Key Financial Metrics")
            
        elif data_type == "Monthly Data":
            monthly_data = st.session_state.financial_model.get_monthly_data()
            display_table_with_download(monthly_data, "monthly_data.csv", "Monthly Financial Data")
            
        elif data_type == "Annual Data":
            annual_data = st.session_state.financial_model.get_annual_data()
            display_table_with_download(annual_data, "annual_data.csv", "Annual Financial Data")
            
        elif data_type == "Growth Monthly Data":
            growth_monthly = st.session_state.revenue_model.get_monthly_data()
            display_table_with_download(growth_monthly, "growth_monthly_data.csv", "Growth Model Monthly Data")
            
        elif data_type == "Growth Annual Data":
            growth_annual = st.session_state.revenue_model.get_annual_data()
            display_table_with_download(growth_annual, "growth_annual_data.csv", "Growth Model Annual Data")
            
        elif data_type == "Cost Monthly Data":
            cost_monthly = st.session_state.cost_model.get_monthly_data()
            display_table_with_download(cost_monthly, "cost_monthly_data.csv", "Cost Model Monthly Data")
            
        elif data_type == "Cost Annual Data":
            cost_annual = st.session_state.cost_model.get_annual_data()
            display_table_with_download(cost_annual, "cost_annual_data.csv", "Cost Model Annual Data")

# Tab 6: VC Report
with tab6:
    st.header("VC Investment Report")
    
    if not st.session_state.models_ready:
        st.warning("Please run the model first from the Growth Strategies tab.")
    else:
        try:
            # Try to load the existing VC report
            vc_report_path = "reports/vc_investment_report.md"
            
            if os.path.exists(vc_report_path):
                with open(vc_report_path, "r") as f:
                    vc_report_content = f.read()
                
                st.markdown(vc_report_content)
            else:
                # If no report exists, generate one
                st.subheader("Generate VC Investment Report")
                
                if st.button("Generate Report"):
                    with st.spinner("Generating VC Investment Report..."):
                        # Get key data for the report
                        fm = st.session_state.financial_model
                        rm = st.session_state.revenue_model
                        cm = st.session_state.cost_model
                        
                        monthly_data = fm.get_monthly_data()
                        annual_data = fm.get_annual_data()
                        
                        # Calculate key metrics directly instead of using the key_metrics table
                        
                        # Find breakeven month
                        profitable_months = monthly_data[monthly_data['ebitda'] > 0]
                        breakeven_month = profitable_months['month_number'].min() if len(profitable_months) > 0 else "Not reached"
                        
                        # Year 6 metrics
                        year6_arr = annual_data.iloc[-1]['year_end_arr'] / 1000000
                        year6_ebitda = annual_data.iloc[-1]['annual_ebitda'] / 1000000
                        year6_customers = int(annual_data.iloc[-1]['year_end_customers'])
                        year6_ltv_cac = annual_data.iloc[-1]['annual_ltv_cac_ratio']
                        
                        # Calculate terminal valuation (5x revenue)
                        terminal_revenue = annual_data.iloc[-1]['annual_revenue']
                        terminal_valuation_5x = terminal_revenue * 5 / 1000000
                        
                        # Calculate burn rate metrics
                        max_burn_rate = monthly_data['burn_rate'].max() / 1000000
                        total_burn = -monthly_data[monthly_data['cash_flow'] < 0]['cash_flow'].sum() / 1000000
                        min_cash_position = monthly_data['capital'].min() / 1000000
                        
                        # Capital efficiency ratio
                        year3_revenue = annual_data[annual_data['year'] == 3]['annual_revenue'].values[0] / 1000000 if len(annual_data) >= 3 else 0
                        capital_efficiency = year3_revenue / total_burn if total_burn > 0 else 0
                        
                        # Create directory if needed
                        os.makedirs(os.path.dirname(vc_report_path), exist_ok=True)
                        
                        # Generate the report content
                        vc_report = f"""# AI SaaS Financial Model - VC Investment Report

## Executive Summary

This report presents a comprehensive financial model for an AI SaaS business, with projections over a 6-year period. The model demonstrates a compelling investment opportunity with strong growth potential and attractive unit economics.

**Key Highlights:**
- Terminal ARR (Year 6): ${year6_arr:.1f}M
- Breakeven achieved in Month {breakeven_month}
- Year 6 EBITDA: ${year6_ebitda:.1f}M
- Terminal valuation (5x Revenue): ${terminal_valuation_5x:.1f}M
- Year 6 LTV:CAC Ratio: {year6_ltv_cac:.1f}
- Total customers by Year 6: {year6_customers:,}

## Growth Metrics

The business demonstrates strong and sustainable growth across all key metrics:

| Year | ARR ($M) | YoY Growth | Customers | EBITDA ($M) | EBITDA Margin |
|------|----------|------------|-----------|-------------|---------------|
"""
                        
                        # Add annual data rows
                        for i, year in enumerate(annual_data['year']):
                            year_data = annual_data.iloc[i]
                            growth_pct = year_data.get('revenue_growth_rate', 0) * 100 if i > 0 else 0
                            vc_report += f"| {year} | {year_data['year_end_arr']/1000000:.1f} | {growth_pct:.1f}% | {int(year_data['year_end_customers']):,} | {year_data['annual_ebitda']/1000000:.1f} | {year_data['annual_ebitda_margin']*100:.1f}% |\n"
                        
                        # Continue building the report
                        vc_report += f"""
## Investment Thesis

1. **Large Addressable Market**: The AI SaaS market is projected to grow significantly, providing a substantial opportunity for growth.

2. **Differentiated Technology**: The company's AI technology provides a competitive advantage that allows for premium pricing and high retention.

3. **Scalable Business Model**: The model demonstrates strong unit economics with improving margins as the business scales.

4. **Multi-Segment Strategy**: The company targets Enterprise, Mid-Market, and SMB segments with tailored approaches for each.

5. **Efficient Growth**: The company achieves breakeven in Month {breakeven_month} while maintaining strong growth.

## Capital Requirements & Deployment

Initial Investment: $20M

The investment will be deployed across the following areas:
- Engineering & Product Development: 35%
- Sales & Marketing: 40%
- Operations & G&A: 15%
- Research & IP Development: 10%

The model projects that the initial investment provides sufficient runway to reach breakeven, after which the company can fund growth from operations.

## Market Strategy by Segment

The company employs a targeted go-to-market strategy across three key segments:

### Enterprise Segment
- Initial ARR: ${rm.config['initial_arr']['Enterprise']/1000:,.0f}K per customer
- Contract Length: {rm.config['contract_length']['Enterprise']} years
- Annual Churn Rate: {rm.config['churn_rates']['Enterprise']*100:.1f}%
- Year 6 Customer Count: {int(annual_data.iloc[-1]['year_end_customers'] * 0.2):,} (est.)

### Mid-Market Segment
- Initial ARR: ${rm.config['initial_arr']['Mid-Market']/1000:,.0f}K per customer
- Contract Length: {rm.config['contract_length']['Mid-Market']} years
- Annual Churn Rate: {rm.config['churn_rates']['Mid-Market']*100:.1f}%
- Year 6 Customer Count: {int(annual_data.iloc[-1]['year_end_customers'] * 0.3):,} (est.)

### SMB Segment
- Initial ARR: ${rm.config['initial_arr']['SMB']/1000:,.0f}K per customer
- Contract Length: {rm.config['contract_length']['SMB']} years
- Annual Churn Rate: {rm.config['churn_rates']['SMB']*100:.1f}%
- Year 6 Customer Count: {int(annual_data.iloc[-1]['year_end_customers'] * 0.5):,} (est.)

## Burn Rate & Capital Efficiency Analysis

- Maximum Monthly Burn Rate: ${max_burn_rate:.1f}M
- Total Burn Before Breakeven: ${total_burn:.1f}M
- Minimum Cash Position: ${min_cash_position:.1f}M
- Capital Efficiency Ratio (ARR / Burn): {capital_efficiency:.2f}x (Year 3)

The company is projected to achieve breakeven in Month {breakeven_month}, with sufficient capital from the initial investment.

## Dynamic Growth Strategy

The model employs a dynamic growth strategy that prioritizes different segments at different stages:
- Years 1-2: Focus on Enterprise and Mid-Market to establish credibility and stable revenue
- Years 3-4: Expand Mid-Market presence while accelerating SMB growth
- Years 5-6: Scale SMB segment while maintaining Enterprise and Mid-Market

This approach maximizes capital efficiency and minimizes risk by building a foundation of stable customers before scaling to higher-volume, higher-CAC segments.

## Unit Economics

| Year | CAC ($) | LTV ($) | LTV:CAC | Gross Margin |
|------|---------|---------|---------|--------------|
"""
                        
                        # Add unit economics data
                        for i, year in enumerate(annual_data['year']):
                            year_data = annual_data.iloc[i]
                            avg_cac = year_data.get('annual_avg_cac', 0)
                            avg_ltv = year_data.get('annual_avg_ltv', 0)
                            ltv_cac = year_data.get('annual_ltv_cac_ratio', 0)
                            gross_margin = year_data['annual_gross_margin'] * 100
                            vc_report += f"| {year} | ${avg_cac:,.0f} | ${avg_ltv:,.0f} | {ltv_cac:.1f} | {gross_margin:.1f}% |\n"
                        
                        # Complete the report
                        vc_report += f"""
## Risk Factors & Mitigations

1. **Competition Risk**: The market for AI SaaS solutions is competitive and rapidly evolving.
   - *Mitigation*: Continuous innovation and focus on targeted segments with specific needs.

2. **Execution Risk**: Achieving the projected growth requires excellent execution.
   - *Mitigation*: Experienced management team and staged growth strategy.

3. **Technology Risk**: AI technology is evolving rapidly.
   - *Mitigation*: Dedicated research team and ongoing investment in R&D.

4. **Market Adoption Risk**: Enterprise sales cycles can be longer than anticipated.
   - *Mitigation*: Multi-segment approach with diverse customer base.

## Conclusion

This financial model demonstrates a compelling investment opportunity with strong returns potential. The combination of high growth, healthy unit economics, and a clear path to profitability positions the company for success. The staged approach to growth across segments provides both stability and upside, while the focus on capital efficiency ensures responsible use of investment funds.

*Generated on {datetime.now().strftime("%Y-%m-%d")}*
"""
                        
                        # Save the report
                        with open(vc_report_path, "w") as f:
                            f.write(vc_report)
                        
                        st.success("VC Investment Report generated successfully!")
                        st.markdown(vc_report)
        
        except Exception as e:
            st.error(f"Error generating or displaying VC report: {str(e)}")
            st.markdown("Please run the model from the Growth Strategies tab, then generate a VC report.")

if __name__ == "__main__":
    # This will be executed when the Streamlit app is run
    pass