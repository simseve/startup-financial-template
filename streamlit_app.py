import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import io
import base64
from PIL import Image

from models.growth_model import SaaSGrowthModel
from models.cost_model import AISaaSCostModel
from models.financial_model import SaaSFinancialModel

# Set page config
st.set_page_config(
    page_title="2025 Financial Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f6f6f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Function to load configuration files
@st.cache_data
def load_configs():
    """Load revenue and cost configuration files"""
    with open(os.path.join('configs', 'revenue_config.json'), 'r') as f:
        revenue_config = json.load(f)
    
    with open(os.path.join('configs', 'cost_config.json'), 'r') as f:
        cost_config = json.load(f)
    
    # Convert string keys to integers in various config sections
    # Revenue config
    for segment in revenue_config['segments']:
        revenue_config['s_curve'][segment] = {
            int(year): params for year, params in revenue_config['s_curve'][segment].items()
        }
    
    revenue_config['seasonality'] = {
        int(month): factor for month, factor in revenue_config['seasonality'].items()
    }
    
    # Cost config
    for dept in cost_config['headcount']:
        cost_config['headcount'][dept]['growth_factors'] = {
            int(year): factor for year, factor in cost_config['headcount'][dept]['growth_factors'].items()
        }
    
    cost_config['marketing_efficiency'] = {
        int(year): factor for year, factor in cost_config['marketing_efficiency'].items()
    }
    
    return revenue_config, cost_config

# Function to save configurations
def save_configs(revenue_config, cost_config):
    """Save revenue and cost configurations to files"""
    # Convert integer keys back to strings for JSON serialization
    rev_config_copy = copy.deepcopy(revenue_config)
    cost_config_copy = copy.deepcopy(cost_config)
    
    # Revenue config
    for segment in rev_config_copy['segments']:
        rev_config_copy['s_curve'][segment] = {
            str(year): params for year, params in rev_config_copy['s_curve'][segment].items()
        }
    
    rev_config_copy['seasonality'] = {
        str(month): factor for month, factor in rev_config_copy['seasonality'].items()
    }
    
    # Cost config
    for dept in cost_config_copy['headcount']:
        cost_config_copy['headcount'][dept]['growth_factors'] = {
            str(year): factor for year, factor in cost_config_copy['headcount'][dept]['growth_factors'].items()
        }
    
    cost_config_copy['marketing_efficiency'] = {
        str(year): factor for year, factor in cost_config_copy['marketing_efficiency'].items()
    }
    
    # Save to files
    with open(os.path.join('configs', 'revenue_config.json'), 'w') as f:
        json.dump(rev_config_copy, f, indent=2)
    
    with open(os.path.join('configs', 'cost_config.json'), 'w') as f:
        json.dump(cost_config_copy, f, indent=2)
    
    st.success("✅ Configurations saved successfully!")

# Function to get JSON download button
def get_download_link(config, filename, button_text):
    """Generate a download link for a JSON file"""
    config_copy = copy.deepcopy(config)
    
    # Convert integer keys back to strings for JSON serialization
    if 's_curve' in config_copy:
        for segment in config_copy['segments']:
            config_copy['s_curve'][segment] = {
                str(year): params for year, params in config_copy['s_curve'][segment].items()
            }
    
    if 'seasonality' in config_copy:
        config_copy['seasonality'] = {
            str(month): factor for month, factor in config_copy['seasonality'].items()
        }
    
    if 'headcount' in config_copy:
        for dept in config_copy['headcount']:
            config_copy['headcount'][dept]['growth_factors'] = {
                str(year): factor for year, factor in config_copy['headcount'][dept]['growth_factors'].items()
            }
    
    if 'marketing_efficiency' in config_copy:
        config_copy['marketing_efficiency'] = {
            str(year): factor for year, factor in config_copy['marketing_efficiency'].items()
        }
    
    json_str = json.dumps(config_copy, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'<a href="data:file/json;base64,{b64}" download="{filename}" class="download-button">{button_text}</a>'

# Get figure data as base64 string
def get_figure_as_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_str

# Function to run models and get results
@st.cache_resource
def run_models(revenue_config, cost_config, _initial_investment=5000000):
    """Run growth, cost, and financial models with the given configurations"""
    # Initialize models
    growth_model = SaaSGrowthModel(revenue_config)
    cost_model = AISaaSCostModel(cost_config)
    
    # Run growth model
    growth_model.run_model()
    
    # Run cost model with growth model
    cost_model.run_model(growth_model)
    
    # Initialize and run financial model
    financial_model = SaaSFinancialModel(
        revenue_model=growth_model,
        cost_model=cost_model,
        initial_investment=_initial_investment
    )
    financial_model.run_model()
    
    return growth_model, cost_model, financial_model

# Main app
def main():
    # Sidebar header
    st.sidebar.title("2025 Financial Model")
    st.sidebar.markdown("#### Interactive Growth & Financial Planning")
    
    # Load configurations
    revenue_config, cost_config = load_configs()
    
    # Sidebar tabs
    sidebar_tab = st.sidebar.radio(
        "Configuration",
        ["Main Settings", "S-Curve Editor", "Cost Settings", "Advanced Settings", "Load/Save"]
    )
    
    # Create a session state to store modified configs
    if 'revenue_config' not in st.session_state:
        st.session_state.revenue_config = copy.deepcopy(revenue_config)
    
    if 'cost_config' not in st.session_state:
        st.session_state.cost_config = copy.deepcopy(cost_config)
    
    if 'initial_investment' not in st.session_state:
        st.session_state.initial_investment = 5000000
    
    if 'global_multiplier' not in st.session_state:
        st.session_state.global_multiplier = 1.0
    
    # Flag to check if we need to rerun the models
    rerun_models = False
    
    # Sidebar content based on selected tab
    if sidebar_tab == "Main Settings":
        st.sidebar.header("Growth Profile")
        profile_options = {
            "Custom": 1.0,
            "Conservative": 0.7,
            "Baseline": 1.0,
            "Aggressive": 1.5,
            "Hypergrowth": 2.5
        }
        
        # Initialize selected_profile in session state if not present
        if 'selected_profile' not in st.session_state:
            st.session_state.selected_profile = "Custom"
            
        selected_profile = st.sidebar.selectbox(
            "Select Growth Profile", 
            list(profile_options.keys()),
            index=list(profile_options.keys()).index(st.session_state.selected_profile)
        )
        
        # Store selected profile in session state
        st.session_state.selected_profile = selected_profile
        
        if selected_profile != "Custom":
            multiplier = profile_options[selected_profile]
            if st.session_state.global_multiplier != multiplier:
                st.session_state.global_multiplier = multiplier
                # Apply the multiplier to all segments and years
                for segment in st.session_state.revenue_config['segments']:
                    for year in range(1, 7):
                        # Get the original value from the base config
                        base_max = revenue_config['s_curve'][segment][year]['max_monthly']
                        # Apply the multiplier
                        st.session_state.revenue_config['s_curve'][segment][year]['max_monthly'] = int(round(base_max * multiplier))
                rerun_models = True
        
        st.sidebar.header("Initial Parameters")
        
        # Initial investment
        new_investment = st.sidebar.slider(
            "Initial Investment", 
            min_value=1000000, 
            max_value=50000000, 
            value=st.session_state.initial_investment,
            step=1000000,
            format="$%d"  # Format is supported for slider
        )
        
        if new_investment != st.session_state.initial_investment:
            st.session_state.initial_investment = new_investment
            rerun_models = True
        
        # Initial customers
        st.sidebar.subheader("Initial Customers")
        for segment in st.session_state.revenue_config['segments']:
            new_customers = st.sidebar.number_input(
                f"{segment} Customers",
                min_value=0,
                max_value=100,
                value=st.session_state.revenue_config['initial_customers'][segment],
                step=1
            )
            
            if new_customers != st.session_state.revenue_config['initial_customers'][segment]:
                st.session_state.revenue_config['initial_customers'][segment] = new_customers
                rerun_models = True
        
        # ARR per customer
        st.sidebar.subheader("Annual Recurring Revenue")
        for segment in st.session_state.revenue_config['segments']:
            new_arr = st.sidebar.number_input(
                f"{segment} ARR",
                min_value=1000,
                max_value=1000000,
                value=st.session_state.revenue_config['initial_arr'][segment],
                step=1000
            )
            
            if new_arr != st.session_state.revenue_config['initial_arr'][segment]:
                st.session_state.revenue_config['initial_arr'][segment] = new_arr
                rerun_models = True
    
    elif sidebar_tab == "S-Curve Editor":
        # S-curve parameters
        st.sidebar.header("S-Curve Parameters")
        
        # Get the selected profile from session state
        if 'selected_profile' not in st.session_state:
            st.session_state.selected_profile = "Custom"
        
        # Only show global multiplier if using Custom profile
        if st.session_state.selected_profile == "Custom":
            # Global multiplier slider
            new_global_multiplier = st.sidebar.slider(
                "Global Growth Multiplier",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.global_multiplier,
                step=0.1
            )
            
            if new_global_multiplier != st.session_state.global_multiplier:
                st.session_state.global_multiplier = new_global_multiplier
                # Apply the multiplier to all segments and years
                for segment in st.session_state.revenue_config['segments']:
                    for year in range(1, 7):
                        # Get the original value from the base config
                        base_max = revenue_config['s_curve'][segment][year]['max_monthly']
                        # Apply the multiplier
                        st.session_state.revenue_config['s_curve'][segment][year]['max_monthly'] = int(round(base_max * new_global_multiplier))
                rerun_models = True
        
        # Show segment tabs for S-curve editor
        segment_tab = st.sidebar.radio("Segment", st.session_state.revenue_config['segments'])
        year_tab = st.sidebar.slider("Year", min_value=1, max_value=6, value=1)
        
        # Get S-curve parameters for selected segment and year
        s_params = st.session_state.revenue_config['s_curve'][segment_tab][year_tab]
        
        # Sliders for S-curve parameters
        new_midpoint = st.sidebar.slider(
            "Month Peak (Midpoint)",
            min_value=1,
            max_value=12,
            value=s_params['midpoint'],
            help="Month within the year when acquisition peaks"
        )
        
        new_steepness = st.sidebar.slider(
            "Steepness",
            min_value=0.1,
            max_value=2.0,
            value=s_params['steepness'],
            step=0.1,
            help="How steep the S-curve is (higher = steeper)"
        )
        
        new_max_monthly = st.sidebar.slider(
            "Max Monthly Customers",
            min_value=0,
            max_value=50,
            value=s_params['max_monthly'],
            help="Maximum number of new customers per month at peak"
        )
        
        # Update parameters if changed
        if (new_midpoint != s_params['midpoint'] or 
            new_steepness != s_params['steepness'] or 
            new_max_monthly != s_params['max_monthly']):
            
            st.session_state.revenue_config['s_curve'][segment_tab][year_tab]['midpoint'] = new_midpoint
            st.session_state.revenue_config['s_curve'][segment_tab][year_tab]['steepness'] = new_steepness
            st.session_state.revenue_config['s_curve'][segment_tab][year_tab]['max_monthly'] = new_max_monthly
            rerun_models = True
        
        # Add seasonality editor
        st.sidebar.header("Seasonality")
        
        # Show months in a 3x4 grid using columns
        col1, col2, col3, col4 = st.sidebar.columns(4)
        
        with col1:
            for month in [1, 5, 9]:
                new_factor = st.number_input(
                    f"Month {month}",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.revenue_config['seasonality'][month],
                    step=0.1,
                    key=f"season_{month}"
                )
                
                if new_factor != st.session_state.revenue_config['seasonality'][month]:
                    st.session_state.revenue_config['seasonality'][month] = new_factor
                    rerun_models = True
        
        with col2:
            for month in [2, 6, 10]:
                new_factor = st.number_input(
                    f"Month {month}",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.revenue_config['seasonality'][month],
                    step=0.1,
                    key=f"season_{month}"
                )
                
                if new_factor != st.session_state.revenue_config['seasonality'][month]:
                    st.session_state.revenue_config['seasonality'][month] = new_factor
                    rerun_models = True
        
        with col3:
            for month in [3, 7, 11]:
                new_factor = st.number_input(
                    f"Month {month}",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.revenue_config['seasonality'][month],
                    step=0.1,
                    key=f"season_{month}"
                )
                
                if new_factor != st.session_state.revenue_config['seasonality'][month]:
                    st.session_state.revenue_config['seasonality'][month] = new_factor
                    rerun_models = True
        
        with col4:
            for month in [4, 8, 12]:
                new_factor = st.number_input(
                    f"Month {month}",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.revenue_config['seasonality'][month],
                    step=0.1,
                    key=f"season_{month}"
                )
                
                if new_factor != st.session_state.revenue_config['seasonality'][month]:
                    st.session_state.revenue_config['seasonality'][month] = new_factor
                    rerun_models = True
    
    elif sidebar_tab == "Cost Settings":
        # Cost settings
        st.sidebar.header("Cost Settings")
        
        # Headcount settings
        st.sidebar.subheader("Headcount")
        
        department = st.sidebar.selectbox("Department", list(st.session_state.cost_config['headcount'].keys()))
        
        new_starting_count = st.sidebar.number_input(
            f"{department} Starting Headcount",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.cost_config['headcount'][department]['starting_count']),
            step=0.5
        )
        
        if new_starting_count != st.session_state.cost_config['headcount'][department]['starting_count']:
            st.session_state.cost_config['headcount'][department]['starting_count'] = new_starting_count
            rerun_models = True
        
        new_avg_salary = st.sidebar.number_input(
            f"{department} Avg Salary",
            min_value=50000,
            max_value=300000,
            value=st.session_state.cost_config['headcount'][department]['avg_salary'],
            step=10000
        )
        
        if new_avg_salary != st.session_state.cost_config['headcount'][department]['avg_salary']:
            st.session_state.cost_config['headcount'][department]['avg_salary'] = new_avg_salary
            rerun_models = True
        
        # Headcount growth factors by year
        st.sidebar.write("Headcount Growth Factors by Year")
        
        for year in range(1, 7):
            new_factor = st.sidebar.number_input(
                f"Year {year} Growth",
                min_value=0.1,
                max_value=5.0,
                value=st.session_state.cost_config['headcount'][department]['growth_factors'][year],
                step=0.1,
                key=f"hc_growth_{department}_{year}"
            )
            
            if new_factor != st.session_state.cost_config['headcount'][department]['growth_factors'][year]:
                st.session_state.cost_config['headcount'][department]['growth_factors'][year] = new_factor
                rerun_models = True
        
        # COGS settings
        st.sidebar.subheader("COGS (% of Revenue)")
        
        for cogs_item in st.session_state.cost_config['cogs']:
            new_value = st.sidebar.slider(
                f"{cogs_item.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.cost_config['cogs'][cogs_item],
                step=0.01,
                format="%.2f"
            )
            
            if new_value != st.session_state.cost_config['cogs'][cogs_item]:
                st.session_state.cost_config['cogs'][cogs_item] = new_value
                rerun_models = True
    
    elif sidebar_tab == "Advanced Settings":
        # Churn and contract settings
        st.sidebar.header("Customer Retention")
        
        for segment in st.session_state.revenue_config['segments']:
            new_churn = st.sidebar.slider(
                f"{segment} Annual Churn",
                min_value=0.01,
                max_value=0.5,
                value=st.session_state.revenue_config['churn_rates'][segment],
                step=0.01,
                format="%.2f"
            )
            
            if new_churn != st.session_state.revenue_config['churn_rates'][segment]:
                st.session_state.revenue_config['churn_rates'][segment] = new_churn
                rerun_models = True
            
            new_contract = st.sidebar.slider(
                f"{segment} Contract Length (Years)",
                min_value=0.5,
                max_value=5.0,
                value=st.session_state.revenue_config['contract_length'][segment],
                step=0.5,
                format="%.1f"
            )
            
            if new_contract != st.session_state.revenue_config['contract_length'][segment]:
                st.session_state.revenue_config['contract_length'][segment] = new_contract
                rerun_models = True
        
        # Price increase settings
        st.sidebar.header("Annual Price Increases")
        
        for segment in st.session_state.revenue_config['segments']:
            new_increase = st.sidebar.slider(
                f"{segment} Annual Increase",
                min_value=0.0,
                max_value=0.2,
                value=st.session_state.revenue_config['annual_price_increases'][segment],
                step=0.01,
                format="%.2f"
            )
            
            if new_increase != st.session_state.revenue_config['annual_price_increases'][segment]:
                st.session_state.revenue_config['annual_price_increases'][segment] = new_increase
                rerun_models = True
    
    elif sidebar_tab == "Load/Save":
        st.sidebar.header("Save & Export Configuration")
        
        # Save current configuration
        if st.sidebar.button("💾 Save Current Configuration"):
            save_configs(st.session_state.revenue_config, st.session_state.cost_config)
        
        # Export configurations
        st.sidebar.markdown("### Export Configurations")
        st.sidebar.markdown(
            get_download_link(
                st.session_state.revenue_config, 
                "revenue_config.json", 
                "📥 Export Revenue Config"
            ), 
            unsafe_allow_html=True
        )
        
        st.sidebar.markdown(
            get_download_link(
                st.session_state.cost_config, 
                "cost_config.json", 
                "📥 Export Cost Config"
            ), 
            unsafe_allow_html=True
        )
        
        # Import configurations
        st.sidebar.header("Import Configuration")
        
        rev_config_file = st.sidebar.file_uploader("Import Revenue Config", type="json")
        if rev_config_file is not None:
            try:
                imported_rev_config = json.load(rev_config_file)
                # Convert string keys to integers
                for segment in imported_rev_config['segments']:
                    imported_rev_config['s_curve'][segment] = {
                        int(year): params for year, params in imported_rev_config['s_curve'][segment].items()
                    }
                
                imported_rev_config['seasonality'] = {
                    int(month): factor for month, factor in imported_rev_config['seasonality'].items()
                }
                
                st.session_state.revenue_config = imported_rev_config
                st.sidebar.success("✅ Revenue config imported!")
                rerun_models = True
            except Exception as e:
                st.sidebar.error(f"Error importing revenue config: {e}")
        
        cost_config_file = st.sidebar.file_uploader("Import Cost Config", type="json")
        if cost_config_file is not None:
            try:
                imported_cost_config = json.load(cost_config_file)
                # Convert string keys to integers
                for dept in imported_cost_config['headcount']:
                    imported_cost_config['headcount'][dept]['growth_factors'] = {
                        int(year): factor for year, factor in imported_cost_config['headcount'][dept]['growth_factors'].items()
                    }
                
                imported_cost_config['marketing_efficiency'] = {
                    int(year): factor for year, factor in imported_cost_config['marketing_efficiency'].items()
                }
                
                st.session_state.cost_config = imported_cost_config
                st.sidebar.success("✅ Cost config imported!")
                rerun_models = True
            except Exception as e:
                st.sidebar.error(f"Error importing cost config: {e}")
        
        # Reset to default configurations
        st.sidebar.header("Reset Configuration")
        
        if st.sidebar.button("♻️ Reset to Default Configuration"):
            st.session_state.revenue_config = copy.deepcopy(revenue_config)
            st.session_state.cost_config = copy.deepcopy(cost_config)
            st.session_state.initial_investment = 5000000
            st.session_state.global_multiplier = 1.0
            st.sidebar.success("✅ Reset to default configuration!")
            rerun_models = True
    
    # Run models with current configuration
    if rerun_models or 'growth_model' not in st.session_state:
        with st.spinner("Running financial model..."):
            growth_model, cost_model, financial_model = run_models(
                st.session_state.revenue_config, 
                st.session_state.cost_config,
                st.session_state.initial_investment
            )
            st.session_state.growth_model = growth_model
            st.session_state.cost_model = cost_model
            st.session_state.financial_model = financial_model
    else:
        growth_model = st.session_state.growth_model
        cost_model = st.session_state.cost_model
        financial_model = st.session_state.financial_model
    
    # Main content
    st.title("AI GRC Financial Model Dashboard")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Financial Overview", 
        "Growth & Customers", 
        "Expenses & Headcount",
        "Configuration"
    ])
    
    with tab1:
        # Financial Overview Tab
        st.header("Financial Overview")
        
        # Key metrics at the top
        key_metrics = financial_model.get_key_metrics_table()
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Find break-even point
        monthly_data = financial_model.get_monthly_data()
        breakeven_msg = "Not achieved within 6 years"
        if 'profitable_month' in monthly_data.columns:
            profitable_months = monthly_data[monthly_data['profitable_month'] == True]
            if len(profitable_months) > 0:
                first_profitable = profitable_months.iloc[0]
                month_number = first_profitable['month_number']
                year = first_profitable['year']
                breakeven_msg = f"Month {month_number} (Year {year})"
        
        with col1:
            st.metric(
                "Year 6 ARR", 
                key_metrics.iloc[-1]['ARR ($M)'], 
                delta=None
            )
            st.metric(
                "Break-Even Point", 
                breakeven_msg,
                delta=None
            )
        
        with col2:
            st.metric(
                "Year 6 Customers", 
                key_metrics.iloc[-1]['Customers'],
                delta=None
            )
            st.metric(
                "Year 6 EBITDA Margin", 
                key_metrics.iloc[-1]['EBITDA Margin (%)'],
                delta=None
            )
        
        with col3:
            st.metric(
                "Year 6 Headcount", 
                key_metrics.iloc[-1]['Headcount'],
                delta=None
            )
            st.metric(
                "Year 6 LTV/CAC", 
                key_metrics.iloc[-1]['LTV/CAC Ratio'],
                delta=None
            )
        
        with col4:
            st.metric(
                "Year 6 Capital Position", 
                key_metrics.iloc[-1]['Capital Position ($M)'],
                delta=None
            )
            st.metric(
                "Year 6 Rule of 40", 
                key_metrics.iloc[-1]['Rule of 40 Score'],
                delta=None
            )
        
        # Financial plots
        col1, col2 = st.columns(2)
        
        with col1:
            financial_summary_fig = financial_model.plot_financial_summary()
            st.pyplot(financial_summary_fig)
        
        with col2:
            break_even_fig = financial_model.plot_break_even_analysis()
            st.pyplot(break_even_fig)
        
        col1, col2 = st.columns(2)
        
        with col1:
            unit_economics_fig = cost_model.plot_unit_economics(
                cost_model.calculate_unit_economics(growth_model)
            )
            st.pyplot(unit_economics_fig)
        
        with col2:
            runway_fig = financial_model.plot_runway_and_capital()
            st.pyplot(runway_fig)
        
        # Key metrics table
        st.subheader("Annual Key Metrics")
        st.dataframe(key_metrics, use_container_width=True)
        
        # Unit economics table
        st.subheader("Unit Economics")
        unit_economics = cost_model.calculate_unit_economics(growth_model)
        unit_economics_table = cost_model.display_unit_economics_table(unit_economics)
        st.dataframe(unit_economics_table, use_container_width=True)
    
    with tab2:
        # Growth & Customers Tab
        st.header("Growth & Customer Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            growth_curves_fig = growth_model.plot_growth_curves()
            st.pyplot(growth_curves_fig)
        
        with col2:
            annual_metrics_fig = growth_model.plot_annual_metrics()
            st.pyplot(annual_metrics_fig)
            
            segment_shares_fig = growth_model.plot_customer_segment_shares()
            st.pyplot(segment_shares_fig)
        
        # Growth metrics table
        st.subheader("Growth Summary")
        growth_summary = growth_model.display_summary_metrics()
        st.dataframe(growth_summary, use_container_width=True)
    
    with tab3:
        # Expenses & Headcount Tab
        st.header("Expenses & Headcount")
        
        col1, col2 = st.columns(2)
        
        with col1:
            expense_breakdown_fig = cost_model.plot_expense_breakdown()
            st.pyplot(expense_breakdown_fig)
        
        with col2:
            headcount_growth_fig = cost_model.plot_headcount_growth()
            st.pyplot(headcount_growth_fig)
        
        # Cost summary table
        st.subheader("Cost Summary")
        cost_summary = cost_model.display_summary_metrics()
        st.dataframe(cost_summary, use_container_width=True)
    
    with tab4:
        # Configuration Tab
        st.header("Model Configuration")
        
        # S-curve parameters
        st.subheader("S-Curve Parameters")
        
        s_curve_params = growth_model.get_s_curve_parameters()
        st.dataframe(s_curve_params, use_container_width=True)
        
        # Revenue configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue Configuration")
            
            # Create a formatted display of revenue configuration
            rev_config_display = pd.DataFrame(index=growth_model.config['segments'])
            
            rev_config_display['Initial ARR'] = [
                f"${growth_model.config['initial_arr'][segment]:,}" 
                for segment in growth_model.config['segments']
            ]
            
            rev_config_display['Initial Customers'] = [
                growth_model.config['initial_customers'][segment] 
                for segment in growth_model.config['segments']
            ]
            
            rev_config_display['Contract Length'] = [
                f"{growth_model.config['contract_length'][segment]:.1f} years" 
                for segment in growth_model.config['segments']
            ]
            
            rev_config_display['Churn Rate'] = [
                f"{growth_model.config['churn_rates'][segment]*100:.1f}%" 
                for segment in growth_model.config['segments']
            ]
            
            rev_config_display['Annual Price Increase'] = [
                f"{growth_model.config['annual_price_increases'].get(segment, 0)*100:.1f}%" 
                for segment in growth_model.config['segments']
            ]
            
            st.dataframe(rev_config_display, use_container_width=True)
            
            # Seasonality
            st.subheader("Seasonality Factors")
            
            seasonality_data = {
                'Month': list(range(1, 13)),
                'Factor': [growth_model.config['seasonality'].get(month, 1.0) for month in range(1, 13)]
            }
            seasonality_df = pd.DataFrame(seasonality_data)
            
            st.bar_chart(seasonality_df.set_index('Month')['Factor'], use_container_width=True)
        
        with col2:
            st.subheader("Cost Configuration")
            
            # Headcount display
            headcount_data = []
            for dept, details in cost_model.config['headcount'].items():
                headcount_data.append({
                    'Department': dept.replace('_', ' ').title(),
                    'Starting Count': f"{details['starting_count']:.1f}",
                    'Avg Salary': f"${details['avg_salary']:,}"
                })
            
            headcount_df = pd.DataFrame(headcount_data)
            st.dataframe(headcount_df, use_container_width=True)
            
            # COGS display
            st.subheader("COGS (% of Revenue)")
            
            cogs_data = [{
                'Category': category.replace('_', ' ').title(),
                'Percentage': f"{value*100:.1f}%"
            } for category, value in cost_model.config['cogs'].items()]
            
            cogs_df = pd.DataFrame(cogs_data)
            st.dataframe(cogs_df, use_container_width=True)
            
            # Compensation factors
            st.subheader("Compensation Factors")
            
            comp_data = [{
                'Factor': factor.replace('_', ' ').title(),
                'Value': f"{value*100:.1f}%" if isinstance(value, float) else value
            } for factor, value in cost_model.config['salary'].items()]
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True)

if __name__ == "__main__":
    main()