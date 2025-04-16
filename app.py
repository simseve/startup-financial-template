import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import io
import base64
import os
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="AI Governance Startup Financial Model",
    page_icon="📊",
    layout="wide"
)

# Config file path
CONFIG_FILE = "model_config.csv"

# Function to load config from CSV or create default config


def load_config():
    if os.path.exists(CONFIG_FILE):
        config_df = pd.read_csv(CONFIG_FILE)
        config = {}
        for _, row in config_df.iterrows():
            key = row['parameter']
            value = row['value']

            # Try to convert to appropriate type
            try:
                # Try as number first
                if '.' in str(value):
                    config[key] = float(value)
                else:
                    config[key] = int(value)
            except (ValueError, TypeError):
                # If conversion fails, keep as string
                config[key] = value

        return config
    else:
        # Default values
        return {
            'initial_funding': 2000000,
            'start_year': 2024,
            'initial_acv': 57000,
            'acv_growth_rate': 0.20,
            'initial_customers': 5,
            'cust_growth_y1': 0.10,
            'cust_growth_y2_y3': 0.075,
            'cust_growth_y4_plus': 0.04,
            'churn_y1': 0.20,
            'churn_y2': 0.15,
            'churn_y3_plus': 0.10,
            'cogs_y1': 0.30,
            'cogs_y2': 0.28,
            'cogs_y3': 0.25,
            'cogs_y4_plus': 0.22,
            'hc_dev_initial': 8,
            'hc_sales_initial': 5,
            'hc_ops_initial': 3,
            'hc_ga_initial': 2,
            'hc_growth_dev_y1': 0.5,
            'hc_growth_dev_y2': 0.6,
            'hc_growth_dev_y3': 0.5,
            'hc_growth_dev_y4_plus': 0.35,
            'hc_growth_sales_y1': 0.6,
            'hc_growth_sales_y2': 0.7,
            'hc_growth_sales_y3': 0.5,
            'hc_growth_sales_y4_plus': 0.4,
            'hc_growth_ops_y1': 0.33,
            'hc_growth_ops_y2': 0.5,
            'hc_growth_ops_y3': 0.4,
            'hc_growth_ops_y4_plus': 0.3,
            'hc_growth_ga_y1': 0.5,
            'hc_growth_ga_y2': 0.33,
            'hc_growth_ga_y3': 0.25,
            'hc_growth_ga_y4_plus': 0.2,
            'salary_dev': 145000,
            'salary_sales': 120000,
            'salary_ops': 100000,
            'salary_ga': 130000,
            'benefits_multiplier': 0.30,
            'annual_salary_increase': 0.05,
            'marketing_percent_y1': 0.5,
            'marketing_percent_y2': 0.4,
            'marketing_percent_y3': 0.35,
            'marketing_percent_y4_plus': 0.3,
            'dev_tools_per_dev': 5000,
            'cloud_infra_per_customer': 40,
            'cloud_fixed_monthly': 2500,
            'office_per_employee': 400,
            'ga_monthly_base': 15000,
            'ga_percent_revenue': 0.08,
            'efficiency_y3': 0.05,
            'efficiency_y4_plus': 0.08
        }

# Function to save config to CSV


def save_config(config):
    config_df = pd.DataFrame({
        'parameter': list(config.keys()),
        'value': list(config.values())
    })
    config_df.to_csv(CONFIG_FILE, index=False)
    return config_df

# Function to update a single config value and save


def update_config_value(key, value):
    config = load_config()
    config[key] = value
    save_config(config)


# Create sidebar for inputs
st.sidebar.title("Model Parameters")

# Function to create a downloadable link for dataframes


def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href


# Main content
st.title("AI Governance Startup Financial Model")
st.markdown("Interactive financial model for your AI governance startup. Adjust parameters in the sidebar to see how they affect your projections.")

# Load the current configuration
config = load_config()

# Initialize parameters with default values

# Basic parameters
with st.sidebar.expander("Business Model & Funding", expanded=True):
    initial_funding = st.number_input(
        "Initial Funding ($)",
        value=int(config['initial_funding']),
        step=100000,
        format="%d",
        key="initial_funding"
    )
    if st.session_state.initial_funding != int(config['initial_funding']):
        update_config_value('initial_funding',
                            st.session_state.initial_funding)

    start_year = st.number_input(
        "Start Year",
        value=int(config['start_year']),
        step=1,
        format="%d",
        key="start_year"
    )
    if st.session_state.start_year != int(config['start_year']):
        update_config_value('start_year', st.session_state.start_year)

# Revenue parameters
with st.sidebar.expander("Revenue Assumptions", expanded=True):
    initial_acv = st.number_input(
        "Initial Average Contract Value ($)",
        value=int(config['initial_acv']),
        step=1000,
        format="%d",
        key="initial_acv"
    )
    if st.session_state.initial_acv != int(config['initial_acv']):
        update_config_value('initial_acv', st.session_state.initial_acv)

    acv_growth_rate = st.slider(
        "Annual ACV Growth Rate (%)",
        min_value=0.05,
        max_value=0.30,
        value=float(config['acv_growth_rate']),
        step=0.01,
        format="%.2f",
        key="acv_growth_rate"
    )
    if st.session_state.acv_growth_rate != float(config['acv_growth_rate']):
        update_config_value('acv_growth_rate',
                            st.session_state.acv_growth_rate)

    initial_customers = st.number_input(
        "Initial Customer Count",
        value=int(config['initial_customers']),
        step=1,
        format="%d",
        key="initial_customers"
    )
    if st.session_state.initial_customers != int(config['initial_customers']):
        update_config_value('initial_customers',
                            st.session_state.initial_customers)

    # Customer growth rates
    cust_growth_y1 = st.slider(
        "Monthly Customer Growth - Year 1 (%)",
        min_value=0.03,
        max_value=0.15,
        value=float(config['cust_growth_y1']),
        step=0.01,
        format="%.2f",
        key="cust_growth_y1"
    )
    if st.session_state.cust_growth_y1 != float(config['cust_growth_y1']):
        update_config_value('cust_growth_y1', st.session_state.cust_growth_y1)

    cust_growth_y2_y3 = st.slider(
        "Monthly Customer Growth - Years 2-3 (%)",
        min_value=0.03,
        max_value=0.12,
        value=float(config['cust_growth_y2_y3']),
        step=0.005,
        format="%.3f",
        key="cust_growth_y2_y3"
    )
    if st.session_state.cust_growth_y2_y3 != float(config['cust_growth_y2_y3']):
        update_config_value('cust_growth_y2_y3',
                            st.session_state.cust_growth_y2_y3)

    cust_growth_y4_plus = st.slider(
        "Monthly Customer Growth - Years 4+ (%)",
        min_value=0.01,
        max_value=0.08,
        value=float(config['cust_growth_y4_plus']),
        step=0.005,
        format="%.3f",
        key="cust_growth_y4_plus"
    )
    if st.session_state.cust_growth_y4_plus != float(config['cust_growth_y4_plus']):
        update_config_value('cust_growth_y4_plus',
                            st.session_state.cust_growth_y4_plus)

    # Churn rates
    churn_y1 = st.slider(
        "Annual Churn Rate - Year 1 (%)",
        min_value=0.10,
        max_value=0.40,
        value=float(config['churn_y1']),
        step=0.05,
        format="%.2f",
        key="churn_y1"
    )
    if st.session_state.churn_y1 != float(config['churn_y1']):
        update_config_value('churn_y1', st.session_state.churn_y1)

    churn_y2 = st.slider(
        "Annual Churn Rate - Year 2 (%)",
        min_value=0.05,
        max_value=0.30,
        value=float(config['churn_y2']),
        step=0.05,
        format="%.2f",
        key="churn_y2"
    )
    if st.session_state.churn_y2 != float(config['churn_y2']):
        update_config_value('churn_y2', st.session_state.churn_y2)

    churn_y3_plus = st.slider(
        "Annual Churn Rate - Years 3+ (%)",
        min_value=0.03,
        max_value=0.20,
        value=float(config['churn_y3_plus']),
        step=0.01,
        format="%.2f",
        key="churn_y3_plus"
    )
    if st.session_state.churn_y3_plus != float(config['churn_y3_plus']):
        update_config_value('churn_y3_plus', st.session_state.churn_y3_plus)

# COGS parameters
with st.sidebar.expander("COGS Assumptions", expanded=False):
    cogs_y1 = st.slider(
        "COGS as % of Revenue - Year 1",
        min_value=0.20,
        max_value=0.40,
        value=float(config['cogs_y1']),
        step=0.01,
        format="%.2f",
        key="cogs_y1"
    )
    if st.session_state.cogs_y1 != float(config['cogs_y1']):
        update_config_value('cogs_y1', st.session_state.cogs_y1)

    cogs_y2 = st.slider(
        "COGS as % of Revenue - Year 2",
        min_value=0.15,
        max_value=0.35,
        value=float(config['cogs_y2']),
        step=0.01,
        format="%.2f",
        key="cogs_y2"
    )
    if st.session_state.cogs_y2 != float(config['cogs_y2']):
        update_config_value('cogs_y2', st.session_state.cogs_y2)

    cogs_y3 = st.slider(
        "COGS as % of Revenue - Year 3",
        min_value=0.15,
        max_value=0.30,
        value=float(config['cogs_y3']),
        step=0.01,
        format="%.2f",
        key="cogs_y3"
    )
    if st.session_state.cogs_y3 != float(config['cogs_y3']):
        update_config_value('cogs_y3', st.session_state.cogs_y3)

    cogs_y4_plus = st.slider(
        "COGS as % of Revenue - Years 4+",
        min_value=0.15,
        max_value=0.25,
        value=float(config['cogs_y4_plus']),
        step=0.01,
        format="%.2f",
        key="cogs_y4_plus"
    )
    if st.session_state.cogs_y4_plus != float(config['cogs_y4_plus']):
        update_config_value('cogs_y4_plus', st.session_state.cogs_y4_plus)

# Headcount parameters
with st.sidebar.expander("Headcount Assumptions", expanded=True):
    # Initial headcount by department
    st.subheader("Initial Headcount")
    hc_dev_initial = st.number_input("Development", value=int(
        config['hc_dev_initial']), step=1, format="%d")
    hc_sales_initial = st.number_input("Sales & Marketing", value=int(
        config['hc_sales_initial']), step=1, format="%d")
    hc_ops_initial = st.number_input("Operations", value=int(
        config['hc_ops_initial']), step=1, format="%d")
    hc_ga_initial = st.number_input("G&A", value=int(
        config['hc_ga_initial']), step=1, format="%d")

    # Calculate and display total
    hc_total_initial = hc_dev_initial + \
        hc_sales_initial + hc_ops_initial + hc_ga_initial
    st.markdown(f"**Total Headcount: {hc_total_initial}**")

    # Headcount growth rates
    st.subheader("Annual Headcount Growth Rates")

    # Development
    st.markdown("Development Team:")
    hc_growth_dev_y1 = st.slider("Year 1", min_value=0.1, max_value=1.0, value=float(
        config['hc_growth_dev_y1']), step=0.05, format="%.2f", key="dev_y1")
    hc_growth_dev_y2 = st.slider("Year 2", min_value=0.1, max_value=1.0, value=float(
        config['hc_growth_dev_y2']), step=0.05, format="%.2f", key="dev_y2")
    hc_growth_dev_y3 = st.slider("Year 3", min_value=0.1, max_value=0.8, value=float(
        config['hc_growth_dev_y3']), step=0.05, format="%.2f", key="dev_y3")
    hc_growth_dev_y4_plus = st.slider("Years 4+", min_value=0.1, max_value=0.6, value=float(
        config['hc_growth_dev_y4_plus']), step=0.05, format="%.2f", key="dev_y4")

    # Sales
    st.markdown("Sales & Marketing Team:")
    hc_growth_sales_y1 = st.slider("Year 1", min_value=0.1, max_value=1.0, value=float(
        config['hc_growth_sales_y1']), step=0.05, format="%.2f", key="sales_y1")
    hc_growth_sales_y2 = st.slider("Year 2", min_value=0.1, max_value=1.0, value=float(
        config['hc_growth_sales_y2']), step=0.05, format="%.2f", key="sales_y2")
    hc_growth_sales_y3 = st.slider("Year 3", min_value=0.1, max_value=0.8, value=float(
        config['hc_growth_sales_y3']), step=0.05, format="%.2f", key="sales_y3")
    hc_growth_sales_y4_plus = st.slider("Years 4+", min_value=0.1, max_value=0.6, value=float(
        config['hc_growth_sales_y4_plus']), step=0.05, format="%.2f", key="sales_y4")

    # Operations
    st.markdown("Operations Team:")
    hc_growth_ops_y1 = st.slider("Year 1", min_value=0.1, max_value=1.0, value=float(
        config['hc_growth_ops_y1']), step=0.05, format="%.2f", key="ops_y1")
    hc_growth_ops_y2 = st.slider("Year 2", min_value=0.1, max_value=1.0, value=float(
        config['hc_growth_ops_y2']), step=0.05, format="%.2f", key="ops_y2")
    hc_growth_ops_y3 = st.slider("Year 3", min_value=0.1, max_value=0.8, value=float(
        config['hc_growth_ops_y3']), step=0.05, format="%.2f", key="ops_y3")
    hc_growth_ops_y4_plus = st.slider("Years 4+", min_value=0.1, max_value=0.6, value=float(
        config['hc_growth_ops_y4_plus']), step=0.05, format="%.2f", key="ops_y4")

    # G&A
    st.markdown("G&A Team:")
    hc_growth_ga_y1 = st.slider("Year 1", min_value=0.0, max_value=0.8, value=float(
        config['hc_growth_ga_y1']), step=0.05, format="%.2f", key="ga_y1")
    hc_growth_ga_y2 = st.slider("Year 2", min_value=0.0, max_value=0.6, value=float(
        config['hc_growth_ga_y2']), step=0.05, format="%.2f", key="ga_y2")
    hc_growth_ga_y3 = st.slider("Year 3", min_value=0.0, max_value=0.5, value=float(
        config['hc_growth_ga_y3']), step=0.05, format="%.2f", key="ga_y3")
    hc_growth_ga_y4_plus = st.slider("Years 4+", min_value=0.0, max_value=0.3, value=float(
        config['hc_growth_ga_y4_plus']), step=0.05, format="%.2f", key="ga_y4")

# Salary and benefits parameters
with st.sidebar.expander("Salary & Benefits", expanded=False):
    salary_dev = st.number_input("Avg. Developer Salary ($)", value=int(
        config['salary_dev']), step=5000, format="%d")
    salary_sales = st.number_input("Avg. Sales Salary ($)", value=int(
        config['salary_sales']), step=5000, format="%d")
    salary_ops = st.number_input("Avg. Operations Salary ($)", value=int(
        config['salary_ops']), step=5000, format="%d")
    salary_ga = st.number_input("Avg. G&A Salary ($)", value=int(
        config['salary_ga']), step=5000, format="%d")
    benefits_multiplier = st.slider("Benefits Multiplier", min_value=0.15, max_value=0.40, value=float(
        config['benefits_multiplier']), step=0.01, format="%.2f")
    annual_salary_increase = st.slider("Annual Salary Increase (%)", min_value=0.0, max_value=0.10, value=float(
        config['annual_salary_increase']), step=0.01, format="%.2f")

# Marketing and other expenses
with st.sidebar.expander("Marketing & Other Expenses", expanded=False):
    marketing_percent_y1 = st.slider("Marketing Spend (% of Revenue) - Year 1", min_value=0.3,
                                     max_value=0.7, value=float(config['marketing_percent_y1']), step=0.05, format="%.2f")
    marketing_percent_y2 = st.slider("Marketing Spend (% of Revenue) - Year 2", min_value=0.2,
                                     max_value=0.6, value=float(config['marketing_percent_y2']), step=0.05, format="%.2f")
    marketing_percent_y3 = st.slider("Marketing Spend (% of Revenue) - Year 3", min_value=0.2,
                                     max_value=0.5, value=float(config['marketing_percent_y3']), step=0.05, format="%.2f")
    marketing_percent_y4_plus = st.slider("Marketing Spend (% of Revenue) - Years 4+", min_value=0.1,
                                          max_value=0.4, value=float(config['marketing_percent_y4_plus']), step=0.05, format="%.2f")

    dev_tools_per_dev = st.number_input("Annual Dev Tools Cost per Developer ($)", value=int(
        config['dev_tools_per_dev']), step=500, format="%d")
    cloud_infra_per_customer = st.number_input("Monthly Cloud Cost per Customer ($)", value=int(
        config['cloud_infra_per_customer']), step=5, format="%d")
    cloud_fixed_monthly = st.number_input("Fixed Monthly Cloud Costs ($)", value=int(
        config['cloud_fixed_monthly']), step=500, format="%d")
    office_per_employee = st.number_input("Monthly Office Cost per Employee ($)", value=int(
        config['office_per_employee']), step=50, format="%d")

    ga_monthly_base = st.number_input("Base Monthly G&A Expenses ($)", value=int(
        config['ga_monthly_base']), step=1000, format="%d")
    ga_percent_revenue = st.slider("G&A as % of Revenue (when higher than base)", min_value=0.03,
                                   max_value=0.15, value=float(config['ga_percent_revenue']), step=0.01, format="%.2f")

# Efficiency gains
with st.sidebar.expander("Efficiency Gains", expanded=False):
    efficiency_y3 = st.slider("Cost Efficiency Gains - Year 3", min_value=0.0,
                              max_value=0.15, value=float(config['efficiency_y3']), step=0.01, format="%.2f")
    efficiency_y4_plus = st.slider("Cost Efficiency Gains - Years 4+", min_value=0.0,
                                   max_value=0.20, value=float(config['efficiency_y4_plus']), step=0.01, format="%.2f")

# Financial model calculation


def generate_financial_model():
    """Generate financial projections based on input parameters."""
    # Initialize model structure
    months = 72  # 6 years
    start_date = datetime(start_year, 1, 1)

    # Initialize arrays for tracking
    dates = []
    year_num = []
    customer_count = []
    acv_values = []
    monthly_revenue = []

    # Headcount by department
    headcount_dev = []
    headcount_sales = []
    headcount_ops = []
    headcount_ga = []
    headcount_total = []

    # Expense breakdowns
    cogs = []
    salary_dev = []
    salary_sales = []
    salary_ops = []
    salary_ga = []
    marketing_spend = []
    dev_tools = []
    cloud_costs = []
    office_costs = []
    ga_expenses = []

    # Summary financials
    total_expenses = []
    gross_profit = []
    ebitda = []
    cash_balance = []

    # Set initial values
    current_customers = initial_customers
    current_acv = initial_acv

    # Initialize department headcounts
    current_headcount_dev = int(hc_dev_initial)
    current_headcount_sales = int(hc_sales_initial)
    current_headcount_ops = int(hc_ops_initial)
    current_headcount_ga = int(hc_ga_initial)

    # Initialize salaries with safe defaults
    def safe_float(value, default=0.0):
        """Helper function to safely convert values to float with default fallback."""
        if isinstance(value, list):
            if len(value) > 0:
                return float(value[0])
            else:
                return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # Safe initialization with better error handling
    current_salary_dev = safe_float(salary_dev, 145000.0)
    current_salary_sales = safe_float(salary_sales, 120000.0)
    current_salary_ops = safe_float(salary_ops, 100000.0)
    current_salary_ga = safe_float(salary_ga, 130000.0)

    # Initialize cash balance
    current_cash = initial_funding

    # Run monthly projections
    for month in range(months):
        # Calculate date and year
        current_date = start_date + timedelta(days=30*month)
        year = math.ceil((month + 1) / 12)
        month_of_year = month % 12 + 1

        dates.append(current_date)
        year_num.append(year)

        # Update ACV and salaries annually (at the beginning of each year)
        if month > 0 and month_of_year == 1:
            current_acv *= (1 + acv_growth_rate)

            # Update salaries with annual increases
            current_salary_dev *= (1 + annual_salary_increase)
            current_salary_sales *= (1 + annual_salary_increase)
            current_salary_ops *= (1 + annual_salary_increase)
            current_salary_ga *= (1 + annual_salary_increase)

        # Determine growth rates based on year
        if year == 1:
            customer_growth_rate = cust_growth_y1
            churn_rate = churn_y1
            cogs_percent = cogs_y1
            marketing_percent = marketing_percent_y1

            # Headcount growth rates by department
            dev_growth_rate = hc_growth_dev_y1
            sales_growth_rate = hc_growth_sales_y1
            ops_growth_rate = hc_growth_ops_y1
            ga_growth_rate = hc_growth_ga_y1

        elif year <= 3:
            customer_growth_rate = cust_growth_y2_y3
            churn_rate = churn_y2 if year == 2 else churn_y3_plus
            cogs_percent = cogs_y2 if year == 2 else cogs_y3
            marketing_percent = marketing_percent_y2 if year == 2 else marketing_percent_y3

            # Headcount growth rates by department
            if year == 2:
                dev_growth_rate = hc_growth_dev_y2
                sales_growth_rate = hc_growth_sales_y2
                ops_growth_rate = hc_growth_ops_y2
                ga_growth_rate = hc_growth_ga_y2
            else:  # year == 3
                dev_growth_rate = hc_growth_dev_y3
                sales_growth_rate = hc_growth_sales_y3
                ops_growth_rate = hc_growth_ops_y3
                ga_growth_rate = hc_growth_ga_y3

        else:
            customer_growth_rate = cust_growth_y4_plus
            churn_rate = churn_y3_plus
            cogs_percent = cogs_y4_plus
            marketing_percent = marketing_percent_y4_plus

            # Headcount growth rates by department
            dev_growth_rate = hc_growth_dev_y4_plus
            sales_growth_rate = hc_growth_sales_y4_plus
            ops_growth_rate = hc_growth_ops_y4_plus
            ga_growth_rate = hc_growth_ga_y4_plus

        # Calculate customer movement
        monthly_churn_rate = churn_rate / 12
        new_customers = round(current_customers * customer_growth_rate)
        churned_customers = round(current_customers * monthly_churn_rate)
        current_customers = current_customers + new_customers - churned_customers

        # Calculate revenue
        monthly_rev = (current_customers * current_acv) / 12
        annual_run_rate = monthly_rev * 12

        # Update headcount quarterly
        if month % 3 == 0 and month > 0:
            current_headcount_dev = math.ceil(
                current_headcount_dev * (1 + dev_growth_rate / 4))
            current_headcount_sales = math.ceil(
                current_headcount_sales * (1 + sales_growth_rate / 4))
            current_headcount_ops = math.ceil(
                current_headcount_ops * (1 + ops_growth_rate / 4))
            current_headcount_ga = math.ceil(
                current_headcount_ga * (1 + ga_growth_rate / 4))

        # Calculate total headcount
        total_headcount = current_headcount_dev + current_headcount_sales + \
            current_headcount_ops + current_headcount_ga

        # Apply efficiency gains in later years
        efficiency_multiplier = 1.0
        if year >= 4:
            efficiency_multiplier = 1.0 - efficiency_y4_plus
        elif year == 3:
            efficiency_multiplier = 1.0 - efficiency_y3

        # Calculate detailed expenses
        # 1. COGS (now including development costs)
        base_monthly_cogs = monthly_rev * cogs_percent

        # Development costs (now part of COGS)
        monthly_salary_dev = (
            current_headcount_dev * current_salary_dev * (1 + benefits_multiplier)) / 12
        monthly_dev_tools = (current_headcount_dev * dev_tools_per_dev) / 12

        # Apply efficiency to all dev costs within COGS
        monthly_salary_dev *= efficiency_multiplier
        monthly_dev_tools *= efficiency_multiplier

        # Total COGS including development costs
        monthly_cogs = base_monthly_cogs + monthly_salary_dev + monthly_dev_tools


        # 2. Remaining employee costs by department (excluding development)
        monthly_salary_sales = (
            current_headcount_sales * current_salary_sales * (1 + benefits_multiplier)) / 12
        monthly_salary_ops = (
            current_headcount_ops * current_salary_ops * (1 + benefits_multiplier)) / 12
        monthly_salary_ga = (current_headcount_ga *
                             current_salary_ga * (1 + benefits_multiplier)) / 12

        # Apply efficiency to salary costs
        monthly_salary_sales *= efficiency_multiplier
        monthly_salary_ops *= efficiency_multiplier
        monthly_salary_ga *= efficiency_multiplier

        # 3. Non-salary expenses
        # Marketing (non-salary)
        monthly_marketing = monthly_rev * marketing_percent

        # Cloud infrastructure
        monthly_cloud = cloud_fixed_monthly + \
            (current_customers * cloud_infra_per_customer)

        # Office costs
        monthly_office = total_headcount * office_per_employee

        # G&A expenses (non-salary)
        monthly_ga = max(ga_monthly_base, monthly_rev * ga_percent_revenue)

        # Apply efficiency to non-salary costs
        monthly_marketing *= efficiency_multiplier
        monthly_cloud *= efficiency_multiplier
        monthly_office *= efficiency_multiplier
        monthly_ga *= efficiency_multiplier

        # Total expenses
        monthly_total_expenses = (
            monthly_cogs +
            monthly_salary_sales + monthly_salary_ops + monthly_salary_ga +
            monthly_marketing +
            monthly_cloud + monthly_office + monthly_ga
        )

        # Calculate gross profit
        monthly_gross_profit = monthly_rev - monthly_cogs

        # EBITDA
        monthly_ebitda = monthly_rev - monthly_total_expenses

        # Cash balance
        current_cash += monthly_ebitda

        # Append values to arrays
        customer_count.append(current_customers)
        acv_values.append(current_acv)
        monthly_revenue.append(monthly_rev)

        headcount_dev.append(current_headcount_dev)
        headcount_sales.append(current_headcount_sales)
        headcount_ops.append(current_headcount_ops)
        headcount_ga.append(current_headcount_ga)
        headcount_total.append(total_headcount)

        cogs.append(monthly_cogs)
        salary_dev.append(monthly_salary_dev)
        salary_sales.append(monthly_salary_sales)
        salary_ops.append(monthly_salary_ops)
        salary_ga.append(monthly_salary_ga)
        marketing_spend.append(monthly_marketing)
        dev_tools.append(monthly_dev_tools)
        cloud_costs.append(monthly_cloud)
        office_costs.append(monthly_office)
        ga_expenses.append(monthly_ga)

        total_expenses.append(monthly_total_expenses)
        gross_profit.append(monthly_gross_profit)
        ebitda.append(monthly_ebitda)
        cash_balance.append(current_cash)

    # Create monthly dataframe
    monthly_df = pd.DataFrame({
        'date': dates,
        'year': year_num,
        'month': [d.month for d in dates],
        'customers': customer_count,
        'acv': acv_values,
        'monthly_revenue': monthly_revenue,
        'headcount_dev': headcount_dev,
        'headcount_sales': headcount_sales,
        'headcount_ops': headcount_ops,
        'headcount_ga': headcount_ga,
        'headcount_total': headcount_total,
        'cogs': cogs,
        'salary_dev': salary_dev,
        'salary_sales': salary_sales,
        'salary_ops': salary_ops,
        'salary_ga': salary_ga,
        'marketing_spend': marketing_spend,
        'dev_tools': dev_tools,
        'cloud_costs': cloud_costs,
        'office_costs': office_costs,
        'ga_expenses': ga_expenses,
        'total_expenses': total_expenses,
        'gross_profit': gross_profit,
        'ebitda': ebitda,
        'cash_balance': cash_balance
    })

    # Create annual summary
    annual_summary = []

    for year in range(1, 7):
        year_data = monthly_df[monthly_df['year'] == year]

        # Year-end values
        year_end = year_data.iloc[-1]

        # Annual aggregates
        annual_revenue = year_data['monthly_revenue'].sum()
        annual_expenses = year_data['total_expenses'].sum()
        annual_ebitda = annual_revenue - annual_expenses
        annual_gross_profit = year_data['gross_profit'].sum()

        # Expense breakdowns
        annual_cogs = year_data['cogs'].sum()
        annual_salary_dev = year_data['salary_dev'].sum()
        annual_salary_sales = year_data['salary_sales'].sum()
        annual_salary_ops = year_data['salary_ops'].sum()
        annual_salary_ga = year_data['salary_ga'].sum()
        annual_marketing = year_data['marketing_spend'].sum()
        annual_dev_tools = year_data['dev_tools'].sum()
        annual_cloud = year_data['cloud_costs'].sum()
        annual_office = year_data['office_costs'].sum()
        annual_ga = year_data['ga_expenses'].sum()

        # Calculate department totals
        annual_dev_total = annual_salary_dev + annual_dev_tools
        annual_sales_total = annual_salary_sales + annual_marketing
        annual_ops_total = annual_salary_ops + annual_cloud
        annual_ga_total = annual_salary_ga + annual_ga + annual_office

        annual_summary.append({
            'year': start_year + year - 1,
            'annual_revenue': annual_revenue,
            'annual_expenses': annual_expenses,
            'annual_ebitda': annual_ebitda,
            'annual_gross_profit': annual_gross_profit,
            'annual_cogs': annual_cogs,
            'annual_dev_total': annual_dev_total,
            'annual_sales_total': annual_sales_total,
            'annual_ops_total': annual_ops_total,
            'annual_ga_total': annual_ga_total,
            'year_end_customers': year_end['customers'],
            'year_end_headcount_total': year_end['headcount_total'],
            'year_end_headcount_dev': year_end['headcount_dev'],
            'year_end_headcount_sales': year_end['headcount_sales'],
            'year_end_headcount_ops': year_end['headcount_ops'],
            'year_end_headcount_ga': year_end['headcount_ga'],
            'year_end_cash_balance': year_end['cash_balance'],
            'gross_margin_percent': (annual_gross_profit / annual_revenue) * 100 if annual_revenue > 0 else 0,
            'ebitda_margin_percent': (annual_ebitda / annual_revenue) * 100 if annual_revenue > 0 else 0
        })

    annual_df = pd.DataFrame(annual_summary)

    # Format for chart display (millions)
    chart_data = []

    for _, row in annual_df.iterrows():
        revenue_millions = round(row['annual_revenue'] / 1000000)
        ebitda_millions = round(row['annual_ebitda'] / 1000000)

        revenue_label = f"${revenue_millions}M"
        if ebitda_millions >= 0:
            ebitda_label = f"${ebitda_millions}M"
        else:
            ebitda_label = f"-${abs(ebitda_millions)}M"

        chart_data.append({
            'year': row['year'],
            'revenue_millions': revenue_millions,
            'ebitda_millions': ebitda_millions,
            'revenue_label': revenue_label,
            'ebitda_label': ebitda_label,
            'customers': int(row['year_end_customers'])
        })

    chart_df = pd.DataFrame(chart_data)

    # Create department breakdown data - with development costs included in COGS
    department_data = []

    for _, row in annual_df.iterrows():
        # Note: Development costs are now part of COGS
        department_data.append({
            'year': row['year'],
            # COGS now includes development costs
            'cogs': round((row['annual_cogs']) / 1000000, 1),
            'sales_marketing': round(row['annual_sales_total'] / 1000000, 1),
            'operations': round(row['annual_ops_total'] / 1000000, 1),
            'g_and_a': round(row['annual_ga_total'] / 1000000, 1),
            'headcount_dev': int(row['year_end_headcount_dev']),
            'headcount_sales': int(row['year_end_headcount_sales']),
            'headcount_ops': int(row['year_end_headcount_ops']),
            'headcount_ga': int(row['year_end_headcount_ga']),
            'headcount_total': int(row['year_end_headcount_total'])
        })

    department_df = pd.DataFrame(department_data)

    return monthly_df, annual_df, chart_df, department_df


# Run the model with current parameters
monthly_df, annual_df, chart_df, department_df = generate_financial_model()

# Display tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(
    ["Key Metrics & Chart", "Annual Summary", "Department Breakdown", "Export Data"])

with tab1:
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate the financial chart
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Plot the data
        x = chart_df['year']
        width = 0.35

        # Revenue bars (dark blue)
        revenue_bars = ax1.bar(
            x - width/2, chart_df['revenue_millions'], width, color='#0047AB', label='Total revenue')

        # EBITDA bars (light blue)
        ebitda_bars = ax1.bar(
            x + width/2, chart_df['ebitda_millions'], width, color='#00BFFF', label='EBITDA')

        # Customers line (red)
        customer_line = ax2.plot(
            x, chart_df['customers'], 'o-', color='#FF4500', linewidth=2, markersize=8, label='Customers')

        # Add data labels to the bars
        for i, bar in enumerate(revenue_bars):
            height = bar.get_height()
            if height >= 0:
                y_pos = height + 1
            else:
                y_pos = height - 3
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                     chart_df['revenue_label'].iloc[i],
                     ha='center', va='bottom', fontweight='bold')

        for i, bar in enumerate(ebitda_bars):
            height = bar.get_height()
            if height >= 0:
                y_pos = height + 1
            else:
                y_pos = height - 3
            ax1.text(bar.get_x() + bar.get_width()/2, y_pos,
                     chart_df['ebitda_label'].iloc[i],
                     ha='center', va='bottom', fontweight='bold')

        # Add customer count labels
        for i, customer in enumerate(chart_df['customers']):
            ax2.text(x[i], customer + 10, str(customer), ha='center',
                     va='bottom', color='#FF4500', fontweight='bold')

        # Set labels and title
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Revenue/EBITDA ($M)', fontsize=12)
        ax2.set_ylabel('Number of customers', fontsize=12)

        plt.title('Key Financial Results and Projections\nIllustrative example',
                  fontsize=16, fontweight='bold')

        # Set the y-axis limits
        max_revenue = max(chart_df['revenue_millions'].max(
        ), chart_df['ebitda_millions'].max())
        min_ebitda = min(0, chart_df['ebitda_millions'].min())

        ax1.set_ylim(min_ebitda - 5, max_revenue + 10)
        ax2.set_ylim(0, chart_df['customers'].max() * 1.2)

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.tight_layout()

        st.pyplot(fig)

        # Add footer text
        st.caption(
            "EBITDA = earnings before interest, taxes, depreciation and amortization; M = millions;")

    with col2:
        st.subheader("Key Metrics")

        # Find profitability month
        profit_month_idx = next((i for i, ebitda in enumerate(
            monthly_df['ebitda']) if ebitda > 0), None)

        if profit_month_idx is not None:
            profit_month = monthly_df.iloc[profit_month_idx]
            profit_date = profit_month['date']
            profit_month_str = f"{profit_date.strftime('%B %Y')} (Month {profit_month_idx + 1})"
        else:
            profit_month_str = "Not within 6-year projection"

        # Find lowest cash point
        lowest_cash_idx = monthly_df['cash_balance'].idxmin()
        lowest_cash = monthly_df.iloc[lowest_cash_idx]
        lowest_cash_date = lowest_cash['date']
        lowest_cash_amount = lowest_cash['cash_balance']

        # Calculate burn rate if negative
        if lowest_cash['ebitda'] < 0:
            burn_rate = abs(lowest_cash['ebitda'])
            runway_months = math.floor(
                lowest_cash['cash_balance'] / burn_rate) if burn_rate > 0 else "N/A"
        else:
            burn_rate = 0
            runway_months = "Cash flow positive"

        # Final year metrics
        final_year = annual_df.iloc[-1]
        final_customers = int(final_year['year_end_customers'])
        final_headcount = int(final_year['year_end_headcount_total'])
        final_revenue = final_year['annual_revenue'] / 1000000
        final_ebitda = final_year['annual_ebitda'] / 1000000
        final_margin = final_year['ebitda_margin_percent']

        # Customer metrics for year 5
        year5 = annual_df.iloc[4] if len(annual_df) > 4 else None

        if year5 is not None:
            # LTV calculation
            avg_annual_revenue = year5['annual_revenue'] / \
                year5['year_end_customers']
            customer_lifetime = 1 / \
                (churn_y3_plus / 100) if churn_y3_plus > 0 else 10
            gross_margin = year5['gross_margin_percent'] / 100
            ltv = avg_annual_revenue * customer_lifetime * gross_margin

            # CAC approximation (simplified)
            y5_customers = year5['year_end_customers']
            y4_customers = annual_df.iloc[3]['year_end_customers'] if len(
                annual_df) > 3 else 0
            new_customers_y5 = y5_customers - y4_customers
            sm_spend_y5 = year5['annual_sales_total']

            cac = sm_spend_y5 / new_customers_y5 if new_customers_y5 > 0 else 0

            ltv_cac_ratio = ltv / cac if cac > 0 else "N/A"
        else:
            ltv = 0
            cac = 0
            ltv_cac_ratio = "N/A"

        # Create metrics display
        st.metric("Profitability Reached", profit_month_str)
        st.metric("Lowest Cash Balance",
                  f"${lowest_cash_amount/1000000:.1f}M in {lowest_cash_date.strftime('%B %Y')}")

        if isinstance(runway_months, str):
            st.metric("Runway at Lowest Point", runway_months)
        else:
            st.metric("Runway at Lowest Point", f"{runway_months} months")

        st.metric("Final Year Revenue", f"${final_revenue:.1f}M")
        st.metric("Final Year EBITDA",
                  f"${final_ebitda:.1f}M ({final_margin:.1f}%)")
        st.metric("Final Year Customers", final_customers)
        st.metric("Final Year Headcount", final_headcount)

        if isinstance(ltv_cac_ratio, str):
            st.metric("Year 5 LTV:CAC Ratio", ltv_cac_ratio)
        else:
            st.metric("Year 5 LTV:CAC Ratio", f"{ltv_cac_ratio:.1f}")

with tab2:
    st.subheader("Annual Financial Summary")

    # Format annual summary for display
    display_annual = annual_df.copy()

    # Convert monetary values to millions
    for col in ['annual_revenue', 'annual_expenses', 'annual_ebitda', 'annual_gross_profit',
                'annual_cogs', 'annual_dev_total', 'annual_sales_total', 'annual_ops_total',
                'annual_ga_total', 'year_end_cash_balance']:
        display_annual[col] = display_annual[col] / 1000000

    # Rename columns for better display
    display_annual = display_annual.rename(columns={
        'year': 'Year',
        'annual_revenue': 'Revenue ($M)',
        'annual_expenses': 'Expenses ($M)',
        'annual_ebitda': 'EBITDA ($M)',
        'annual_gross_profit': 'Gross Profit ($M)',
        'annual_cogs': 'COGS ($M)',
        'annual_dev_total': 'Development ($M)',
        'annual_sales_total': 'Sales & Marketing ($M)',
        'annual_ops_total': 'Operations ($M)',
        'annual_ga_total': 'G&A ($M)',
        'year_end_customers': 'Customers',
        'year_end_headcount_total': 'Total Headcount',
        'year_end_headcount_dev': 'Dev Headcount',
        'year_end_headcount_sales': 'Sales Headcount',
        'year_end_headcount_ops': 'Ops Headcount',
        'year_end_headcount_ga': 'G&A Headcount',
        'year_end_cash_balance': 'Cash Balance ($M)',
        'gross_margin_percent': 'Gross Margin (%)',
        'ebitda_margin_percent': 'EBITDA Margin (%)'
    })

    # Format number of decimal places
    float_cols = display_annual.select_dtypes(include=['float64']).columns
    for col in float_cols:
        display_annual[col] = display_annual[col].round(1)

    # Display as a styled table
    st.dataframe(display_annual, use_container_width=True)

    # Create annual charts
    col1, col2 = st.columns(2)

    with col1:
        # Revenue and EBITDA chart
        fig, ax = plt.subplots(figsize=(10, 6))

        x = display_annual['Year']
        ax.bar(x, display_annual['Revenue ($M)'],
               label='Revenue', color='#0047AB', alpha=0.7)
        ax.plot(x, display_annual['EBITDA ($M)'], marker='o',
                linewidth=2, color='red', label='EBITDA')

        for i, v in enumerate(display_annual['Revenue ($M)']):
            ax.text(x[i], v + 0.5, f"${v}M", ha='center')

        for i, v in enumerate(display_annual['EBITDA ($M)']):
            ax.text(x[i], v + 0.5 if v >= 0 else v - 2, f"${v}M", ha='center')

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel('$ Millions')
        ax.set_title('Annual Revenue and EBITDA')
        ax.legend()

        st.pyplot(fig)

    with col2:
        # Margin percentages
        fig, ax = plt.subplots(figsize=(10, 6))

        x = display_annual['Year']
        ax.plot(x, display_annual['Gross Margin (%)'], marker='s',
                linewidth=2, color='green', label='Gross Margin')
        ax.plot(x, display_annual['EBITDA Margin (%)'], marker='o',
                linewidth=2, color='purple', label='EBITDA Margin')

        for i, v in enumerate(display_annual['Gross Margin (%)']):
            ax.text(x[i], v + 1, f"{v}%", ha='center')

        for i, v in enumerate(display_annual['EBITDA Margin (%)']):
            ax.text(x[i], v + 1 if v >= 0 else v - 5, f"{v}%", ha='center')

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage')
        ax.set_title('Margin Percentages')
        ax.legend()

        st.pyplot(fig)

with tab3:
    st.subheader("Department Expense Breakdown")

    # Format department breakdown for display
    display_dept = department_df.copy()

    # Rename columns for better display
    display_dept = display_dept.rename(columns={
        'year': 'Year',
        'cogs': 'COGS ($M)',
        'sales_marketing': 'Sales & Marketing ($M)',
        'operations': 'Operations ($M)',
        'g_and_a': 'G&A ($M)',
        'headcount_dev': 'Dev HC',
        'headcount_sales': 'Sales HC',
        'headcount_ops': 'Ops HC',
        'headcount_ga': 'G&A HC',
        'headcount_total': 'Total HC'
    })

    # Display as a styled table
    st.dataframe(display_dept, use_container_width=True)

    # Create department visualization with development costs included in COGS
    fig, ax = plt.subplots(figsize=(12, 7))

    x = display_dept['Year']
    width = 0.75

    # Create stacked bar chart
    bottoms = np.zeros(len(display_dept))

    # COGS (now includes development costs)
    cogs_bars = ax.bar(
        x, display_dept['COGS ($M)'], width, label='COGS (incl. Development)', color='#e74c3c')
    bottoms += display_dept['COGS ($M)']

    # Sales & Marketing
    sales_bars = ax.bar(x, display_dept['Sales & Marketing ($M)'], width,
                        bottom=bottoms, label='Sales & Marketing', color='#3498db')
    bottoms += display_dept['Sales & Marketing ($M)']

    # Operations
    ops_bars = ax.bar(x, display_dept['Operations ($M)'], width,
                      bottom=bottoms, label='Operations', color='#f39c12')
    bottoms += display_dept['Operations ($M)']

    # G&A
    ga_bars = ax.bar(x, display_dept['G&A ($M)'], width,
                     bottom=bottoms, label='G&A', color='#9b59b6')

    # Set labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Expenses ($M)', fontsize=12)
    ax.set_title('Expense Breakdown by Department',
                 fontsize=16, fontweight='bold')

    # Add legend
    ax.legend(loc='upper left')

    # Add headcount annotations
    for i, year_data in enumerate(display_dept.iterrows()):
        _, row = year_data
        total_height = row['COGS ($M)'] + row['Sales & Marketing ($M)'] + \
            row['Operations ($M)'] + row['G&A ($M)']
        ax.text(row['Year'], total_height + 0.5,
                f"Total HC: {row['Total HC']}", ha='center', fontweight='bold')

    plt.tight_layout()

    st.pyplot(fig)

    # Headcount chart
    st.subheader("Headcount Growth by Department")

    fig, ax = plt.subplots(figsize=(12, 7))

    x = display_dept['Year']
    width = 0.75

    # Create stacked bar chart for headcount
    bottoms = np.zeros(len(display_dept))

    # Development
    dev_hc_bars = ax.bar(
        x, display_dept['Dev HC'], width, label='Development', color='#2ecc71')
    bottoms += display_dept['Dev HC']

    # Sales & Marketing
    sales_hc_bars = ax.bar(x, display_dept['Sales HC'], width,
                           bottom=bottoms, label='Sales & Marketing', color='#3498db')
    bottoms += display_dept['Sales HC']

    # Operations
    ops_hc_bars = ax.bar(
        x, display_dept['Ops HC'], width, bottom=bottoms, label='Operations', color='#f39c12')
    bottoms += display_dept['Ops HC']

    # G&A
    ga_hc_bars = ax.bar(
        x, display_dept['G&A HC'], width, bottom=bottoms, label='G&A', color='#9b59b6')

    # Set labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Headcount', fontsize=12)
    ax.set_title('Headcount Growth by Department',
                 fontsize=16, fontweight='bold')

    # Add legend
    ax.legend(loc='upper left')

    # Add total headcount labels
    for i, year_data in enumerate(display_dept.iterrows()):
        _, row = year_data
        total_height = row['Dev HC'] + row['Sales HC'] + \
            row['Ops HC'] + row['G&A HC']
        ax.text(row['Year'], total_height + 2,
                f"{total_height}", ha='center', fontweight='bold')

    plt.tight_layout()

    st.pyplot(fig)

with tab4:
    st.subheader("Export Data")

    # Create download links for each dataset
    st.markdown("### Download CSV Files")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Core Financial Data")
        st.markdown(get_download_link(annual_df, "annual_summary",
                    "Download Annual Summary"), unsafe_allow_html=True)
        st.markdown(get_download_link(chart_df, "chart_data",
                    "Download Chart Data"), unsafe_allow_html=True)
        st.markdown(get_download_link(department_df, "department_breakdown",
                    "Download Department Breakdown"), unsafe_allow_html=True)

    with col2:
        st.markdown("#### Detailed Data")
        st.markdown(get_download_link(monthly_df, "monthly_projections",
                    "Download Monthly Projections"), unsafe_allow_html=True)

        # Create a dataframe with all parameters for export
        params_dict = {
            'parameter': [
                'initial_funding', 'start_year', 'initial_acv', 'acv_growth_rate', 'initial_customers',
                'cust_growth_y1', 'cust_growth_y2_y3', 'cust_growth_y4_plus',
                'churn_y1', 'churn_y2', 'churn_y3_plus',
                'cogs_y1', 'cogs_y2', 'cogs_y3', 'cogs_y4_plus',
                'hc_dev_initial', 'hc_sales_initial', 'hc_ops_initial', 'hc_ga_initial',
                'hc_growth_dev_y1', 'hc_growth_dev_y2', 'hc_growth_dev_y3', 'hc_growth_dev_y4_plus',
                'hc_growth_sales_y1', 'hc_growth_sales_y2', 'hc_growth_sales_y3', 'hc_growth_sales_y4_plus',
                'hc_growth_ops_y1', 'hc_growth_ops_y2', 'hc_growth_ops_y3', 'hc_growth_ops_y4_plus',
                'hc_growth_ga_y1', 'hc_growth_ga_y2', 'hc_growth_ga_y3', 'hc_growth_ga_y4_plus',
                'salary_dev', 'salary_sales', 'salary_ops', 'salary_ga',
                'benefits_multiplier', 'annual_salary_increase',
                'marketing_percent_y1', 'marketing_percent_y2', 'marketing_percent_y3', 'marketing_percent_y4_plus',
                'dev_tools_per_dev', 'cloud_infra_per_customer', 'cloud_fixed_monthly', 'office_per_employee',
                'ga_monthly_base', 'ga_percent_revenue',
                'efficiency_y3', 'efficiency_y4_plus'
            ],
            'value': [
                initial_funding, start_year, initial_acv, acv_growth_rate, initial_customers,
                cust_growth_y1, cust_growth_y2_y3, cust_growth_y4_plus,
                churn_y1, churn_y2, churn_y3_plus,
                cogs_y1, cogs_y2, cogs_y3, cogs_y4_plus,
                hc_dev_initial, hc_sales_initial, hc_ops_initial, hc_ga_initial,
                hc_growth_dev_y1, hc_growth_dev_y2, hc_growth_dev_y3, hc_growth_dev_y4_plus,
                hc_growth_sales_y1, hc_growth_sales_y2, hc_growth_sales_y3, hc_growth_sales_y4_plus,
                hc_growth_ops_y1, hc_growth_ops_y2, hc_growth_ops_y3, hc_growth_ops_y4_plus,
                hc_growth_ga_y1, hc_growth_ga_y2, hc_growth_ga_y3, hc_growth_ga_y4_plus,
                salary_dev, salary_sales, salary_ops, salary_ga,
                benefits_multiplier, annual_salary_increase,
                marketing_percent_y1, marketing_percent_y2, marketing_percent_y3, marketing_percent_y4_plus,
                dev_tools_per_dev, cloud_infra_per_customer, cloud_fixed_monthly, office_per_employee,
                ga_monthly_base, ga_percent_revenue,
                efficiency_y3, efficiency_y4_plus
            ],
            'description': [
                'Initial seed funding', 'Start year', 'Initial average contract value',
                'Annual growth rate for contract value', 'Initial customer count',
                'Monthly customer growth rate in year 1', 'Monthly customer growth rate in years 2-3',
                'Monthly customer growth rate in years 4+',
                'Annual customer churn rate in year 1', 'Annual customer churn rate in year 2',
                'Annual customer churn rate in years 3+',
                'COGS as % of revenue in year 1', 'COGS as % of revenue in year 2',
                'COGS as % of revenue in year 3', 'COGS as % of revenue in years 4+',
                'Initial development headcount', 'Initial sales & marketing headcount',
                'Initial operations headcount', 'Initial G&A headcount',
                'Development headcount growth rate in year 1', 'Development headcount growth rate in year 2',
                'Development headcount growth rate in year 3', 'Development headcount growth rate in years 4+',
                'Sales headcount growth rate in year 1', 'Sales headcount growth rate in year 2',
                'Sales headcount growth rate in year 3', 'Sales headcount growth rate in years 4+',
                'Operations headcount growth rate in year 1', 'Operations headcount growth rate in year 2',
                'Operations headcount growth rate in year 3', 'Operations headcount growth rate in years 4+',
                'G&A headcount growth rate in year 1', 'G&A headcount growth rate in year 2',
                'G&A headcount growth rate in year 3', 'G&A headcount growth rate in years 4+',
                'Average developer salary', 'Average sales salary', 'Average operations salary', 'Average G&A salary',
                'Benefits multiplier', 'Annual salary increase',
                'Marketing spend as % of revenue in year 1', 'Marketing spend as % of revenue in year 2',
                'Marketing spend as % of revenue in year 3', 'Marketing spend as % of revenue in years 4+',
                'Annual dev tools cost per developer', 'Monthly cloud cost per customer',
                'Fixed monthly cloud costs', 'Monthly office cost per employee',
                'Base monthly G&A expenses', 'G&A as % of revenue (when higher than base)',
                'Cost efficiency gains in year 3', 'Cost efficiency gains in years 4+'
            ]
        }

        params_df = pd.DataFrame(params_dict)
        st.markdown(get_download_link(params_df, "model_parameters",
                    "Download Model Parameters"), unsafe_allow_html=True)

    # Create full data export
    st.markdown("### Export All Data")

    # Create a zip file-like buffer
    if st.button("Export All Data (CSV)"):
        # Create a temporary directory and save the CSVs there
        st.markdown(
            "Exporting all data... Please click on each link below to download:")
        st.markdown("1. " + get_download_link(annual_df, "annual_summary",
                    "Annual Summary CSV"), unsafe_allow_html=True)
        st.markdown("2. " + get_download_link(monthly_df, "monthly_projections",
                    "Monthly Projections CSV"), unsafe_allow_html=True)
        st.markdown("3. " + get_download_link(chart_df, "chart_data",
                    "Chart Data CSV"), unsafe_allow_html=True)
        st.markdown("4. " + get_download_link(department_df, "department_breakdown",
                    "Department Breakdown CSV"), unsafe_allow_html=True)
        st.markdown("5. " + get_download_link(params_df, "model_parameters",
                    "Model Parameters CSV"), unsafe_allow_html=True)

    # Add config testing section
    st.subheader("Configuration Management")

    st.write("The model saves your parameter changes automatically to a CSV file called 'model_config.csv'. This allows your settings to persist between sessions.")

    # Display the current config
    if st.checkbox("View current saved configuration"):
        if os.path.exists(CONFIG_FILE):
            config_df = pd.read_csv(CONFIG_FILE)
            st.dataframe(config_df)
        else:
            st.warning(
                "No saved configuration file found yet. Make changes to parameters to create one.")

    # Test section
    st.write("### Testing the Config Saving")
    st.write("1. Change any parameter in the sidebar")
    st.write("2. The value will be automatically saved to the config file")
    st.write("3. Refresh the page to verify that your settings are preserved")

    # Manually update config
    if st.button("Refresh Config View"):
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**AI Governance Startup Financial Model** | Made with Streamlit")
st.markdown("This interactive financial model allows you to adjust all parameters and instantly see the impact on your projections.")
