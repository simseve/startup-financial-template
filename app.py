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

            # Customer Segments - replace old single ACV with segment-specific values
            'sme_initial_arr': 25000,
            'enterprise_initial_arr': 120000,
            'startup_initial_arr': 15000,

            # Contract length by segment (in years)
            'sme_contract_length': 1.0,
            'enterprise_contract_length': 2.0,
            'startup_contract_length': 1.0,

            # ARR growth rates by segment
            'sme_arr_growth_rate': 0.15,
            'enterprise_arr_growth_rate': 0.20,
            'startup_arr_growth_rate': 0.25,

            # Initial customers by segment
            'sme_initial_customers': 3,
            'enterprise_initial_customers': 1,
            'startup_initial_customers': 1,

            # Customer growth rates by segment and year
            'sme_cust_growth_y1': 0.10,
            'sme_cust_growth_y2_y3': 0.075,
            'sme_cust_growth_y4_plus': 0.04,

            'enterprise_cust_growth_y1': 0.08,
            'enterprise_cust_growth_y2_y3': 0.06,
            'enterprise_cust_growth_y4_plus': 0.03,

            'startup_cust_growth_y1': 0.15,
            'startup_cust_growth_y2_y3': 0.12,
            'startup_cust_growth_y4_plus': 0.08,

            # Churn rates by segment
            'sme_churn_y1': 0.20,
            'sme_churn_y2': 0.15,
            'sme_churn_y3_plus': 0.10,

            'enterprise_churn_y1': 0.10,
            'enterprise_churn_y2': 0.08,
            'enterprise_churn_y3_plus': 0.06,

            'startup_churn_y1': 0.30,
            'startup_churn_y2': 0.25,
            'startup_churn_y3_plus': 0.20,

            # Professional Services
            'ps_percent_of_arr': 0.30,
            'ps_margin': 0.40,
            'ps_growth_rate': 0.10,

            # Keeping the rest of the original parameters
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
            'marketing_budget_y1': 500000,
            'marketing_budget_y2': 750000,
            'marketing_budget_y3': 1000000,
            'marketing_budget_growth': 0.20,
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
with st.sidebar.expander("Revenue & Customer Segments", expanded=True):
    # SME segment
    st.subheader("SME Segment")
    sme_initial_arr = st.number_input(
        "SME Initial ARR ($)",
        value=int(config.get('sme_initial_arr', 25000)),
        step=1000,
        format="%d",
        key="sme_initial_arr"
    )
    if st.session_state.sme_initial_arr != int(config.get('sme_initial_arr', 25000)):
        update_config_value('sme_initial_arr',
                            st.session_state.sme_initial_arr)

    sme_initial_customers = st.number_input(
        "SME Initial Customers",
        value=int(config.get('sme_initial_customers', 3)),
        step=1,
        format="%d",
        key="sme_initial_customers"
    )
    if st.session_state.sme_initial_customers != int(config.get('sme_initial_customers', 3)):
        update_config_value('sme_initial_customers',
                            st.session_state.sme_initial_customers)

    sme_arr_growth_rate = st.slider(
        "SME Annual ARR Growth Rate (percentage points)",
        min_value=0.05,
        max_value=0.50,
        value=float(config.get('sme_arr_growth_rate', 0.15)),
        step=0.01,
        format="%.2f",
        key="sme_arr_growth_rate"
    )
    if st.session_state.sme_arr_growth_rate != float(config.get('sme_arr_growth_rate', 0.15)):
        update_config_value('sme_arr_growth_rate',
                            st.session_state.sme_arr_growth_rate)

    sme_contract_length = st.slider(
        "SME Contract Length (years)",
        min_value=1.0,
        max_value=3.0,
        value=float(config.get('sme_contract_length', 1.0)),
        step=0.5,
        format="%.1f",
        key="sme_contract_length"
    )
    if st.session_state.sme_contract_length != float(config.get('sme_contract_length', 1.0)):
        update_config_value('sme_contract_length',
                            st.session_state.sme_contract_length)

    # Enterprise segment
    st.subheader("Enterprise Segment")
    enterprise_initial_arr = st.number_input(
        "Enterprise Initial ARR ($)",
        value=int(config.get('enterprise_initial_arr', 120000)),
        step=5000,
        format="%d",
        key="enterprise_initial_arr"
    )
    if st.session_state.enterprise_initial_arr != int(config.get('enterprise_initial_arr', 120000)):
        update_config_value('enterprise_initial_arr',
                            st.session_state.enterprise_initial_arr)

    enterprise_initial_customers = st.number_input(
        "Enterprise Initial Customers",
        value=int(config.get('enterprise_initial_customers', 1)),
        step=1,
        format="%d",
        key="enterprise_initial_customers"
    )
    if st.session_state.enterprise_initial_customers != int(config.get('enterprise_initial_customers', 1)):
        update_config_value('enterprise_initial_customers',
                            st.session_state.enterprise_initial_customers)

    enterprise_arr_growth_rate = st.slider(
        "Enterprise Annual ARR Growth Rate (percentage points)",
        min_value=0.05,
        max_value=0.50,
        value=float(config.get('enterprise_arr_growth_rate', 0.20)),
        step=0.01,
        format="%.2f",
        key="enterprise_arr_growth_rate"
    )
    if st.session_state.enterprise_arr_growth_rate != float(config.get('enterprise_arr_growth_rate', 0.20)):
        update_config_value('enterprise_arr_growth_rate',
                            st.session_state.enterprise_arr_growth_rate)

    enterprise_contract_length = st.slider(
        "Enterprise Contract Length (years)",
        min_value=1.0,
        max_value=3.0,
        value=float(config.get('enterprise_contract_length', 2.0)),
        step=0.5,
        format="%.1f",
        key="enterprise_contract_length"
    )
    if st.session_state.enterprise_contract_length != float(config.get('enterprise_contract_length', 2.0)):
        update_config_value('enterprise_contract_length',
                            st.session_state.enterprise_contract_length)

    # Startup segment
    st.subheader("Startup Segment")
    startup_initial_arr = st.number_input(
        "Startup Initial ARR ($)",
        value=int(config.get('startup_initial_arr', 15000)),
        step=1000,
        format="%d",
        key="startup_initial_arr"
    )
    if st.session_state.startup_initial_arr != int(config.get('startup_initial_arr', 15000)):
        update_config_value('startup_initial_arr',
                            st.session_state.startup_initial_arr)

    startup_initial_customers = st.number_input(
        "Startup Initial Customers",
        value=int(config.get('startup_initial_customers', 1)),
        step=1,
        format="%d",
        key="startup_initial_customers"
    )
    if st.session_state.startup_initial_customers != int(config.get('startup_initial_customers', 1)):
        update_config_value('startup_initial_customers',
                            st.session_state.startup_initial_customers)

    startup_arr_growth_rate = st.slider(
        "Startup Annual ARR Growth Rate (percentage points)",
        min_value=0.05,
        max_value=0.60,
        value=float(config.get('startup_arr_growth_rate', 0.25)),
        step=0.01,
        format="%.2f",
        key="startup_arr_growth_rate"
    )
    if st.session_state.startup_arr_growth_rate != float(config.get('startup_arr_growth_rate', 0.25)):
        update_config_value('startup_arr_growth_rate',
                            st.session_state.startup_arr_growth_rate)

    startup_contract_length = st.slider(
        "Startup Contract Length (years)",
        min_value=1.0,
        max_value=3.0,
        value=float(config.get('startup_contract_length', 1.0)),
        step=0.5,
        format="%.1f",
        key="startup_contract_length"
    )
    if st.session_state.startup_contract_length != float(config.get('startup_contract_length', 1.0)):
        update_config_value('startup_contract_length',
                            st.session_state.startup_contract_length)

    # Professional Services section
    st.subheader("Professional Services")
    ps_percent_of_arr = st.slider(
        "Professional Services (% of ARR)",
        min_value=0.0,
        max_value=0.5,
        value=float(config.get('ps_percent_of_arr', 0.30)),
        step=0.05,
        format="%.2f",
        key="ps_percent_of_arr"
    )
    if st.session_state.ps_percent_of_arr != float(config.get('ps_percent_of_arr', 0.30)):
        update_config_value('ps_percent_of_arr',
                            st.session_state.ps_percent_of_arr)

    ps_margin = st.slider(
        "Professional Services Margin (%)",
        min_value=0.2,
        max_value=0.6,
        value=float(config.get('ps_margin', 0.40)),
        step=0.05,
        format="%.2f",
        key="ps_margin"
    )
    if st.session_state.ps_margin != float(config.get('ps_margin', 0.40)):
        update_config_value('ps_margin', st.session_state.ps_margin)

    ps_growth_rate = st.slider(
        "Professional Services Annual Growth Rate (%)",
        min_value=0.0,
        max_value=0.25,
        value=float(config.get('ps_growth_rate', 0.10)),
        step=0.05,
        format="%.2f",
        key="ps_growth_rate"
    )
    if st.session_state.ps_growth_rate != float(config.get('ps_growth_rate', 0.10)):
        update_config_value('ps_growth_rate', st.session_state.ps_growth_rate)

# Add exponential growth factors
with st.sidebar.expander("Customer Growth Curve", expanded=True):
    st.markdown("### Exponential Growth Curve Parameters")
    st.markdown("Control how customer acquisition accelerates during each year")

    growth_curve_steepness = st.slider(
        "Growth Curve Steepness",
        min_value=0.5,
        max_value=3.0,
        value=float(config.get('growth_curve_steepness', 2.0)),
        step=0.1,
        format="%.1f",
        key="growth_curve_steepness",
        help="Higher values create a steeper S-curve where growth accelerates more strongly later in the year"
    )
    if st.session_state.get('growth_curve_steepness') != float(config.get('growth_curve_steepness', 2.0)):
        update_config_value('growth_curve_steepness', growth_curve_steepness)

    # Adjustments for each customer segment
    st.markdown("#### Segment-specific adjustments")
    sme_curve_factor = st.slider(
        "SME Growth Curve Factor",
        min_value=0.5,
        max_value=2.0,
        value=float(config.get('sme_curve_factor', 1.0)),
        step=0.1,
        format="%.1f",
        key="sme_curve_factor"
    )
    if st.session_state.get('sme_curve_factor') != float(config.get('sme_curve_factor', 1.0)):
        update_config_value('sme_curve_factor', sme_curve_factor)

    enterprise_curve_factor = st.slider(
        "Enterprise Growth Curve Factor",
        min_value=0.5,
        max_value=2.0,
        value=float(config.get('enterprise_curve_factor', 1.0)),
        step=0.1,
        format="%.1f",
        key="enterprise_curve_factor"
    )
    if st.session_state.get('enterprise_curve_factor') != float(config.get('enterprise_curve_factor', 1.0)):
        update_config_value('enterprise_curve_factor', enterprise_curve_factor)

    startup_curve_factor = st.slider(
        "Startup Growth Curve Factor",
        min_value=0.5,
        max_value=2.0,
        value=float(config.get('startup_curve_factor', 1.0)),
        step=0.1,
        format="%.1f",
        key="startup_curve_factor"
    )
    if st.session_state.get('startup_curve_factor') != float(config.get('startup_curve_factor', 1.0)):
        update_config_value('startup_curve_factor', startup_curve_factor)

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
    # Combining Operations + G&A into SG&A
    hc_ops_initial = st.number_input("Operations (part of SG&A)", value=int(
        config['hc_ops_initial']), step=1, format="%d")
    hc_ga_initial = st.number_input("G&A (part of SG&A)", value=int(
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

    # SG&A (previously just G&A)
    st.markdown("SG&A Team:")
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
    st.markdown("#### Marketing Spend")
    st.markdown("In early-stage startups, marketing budget is typically set as a fixed allocation rather than as a percentage of revenue.")

    # Fixed marketing budget inputs
    marketing_budget_y1 = st.number_input(
        "Annual Marketing Budget - Year 1 ($)",
        min_value=50000,
        max_value=2000000,
        value=int(config.get('marketing_budget_y1', 500000)),
        step=50000,
        format="%d",
        key="marketing_budget_y1"
    )
    if st.session_state.get('marketing_budget_y1', 500000) != int(config.get('marketing_budget_y1', 500000)):
        update_config_value('marketing_budget_y1', marketing_budget_y1)

    marketing_budget_y2 = st.number_input(
        "Annual Marketing Budget - Year 2 ($)",
        min_value=100000,
        max_value=3000000,
        value=int(config.get('marketing_budget_y2', 750000)),
        step=50000,
        format="%d",
        key="marketing_budget_y2"
    )
    if st.session_state.get('marketing_budget_y2', 750000) != int(config.get('marketing_budget_y2', 750000)):
        update_config_value('marketing_budget_y2', marketing_budget_y2)

    marketing_budget_y3 = st.number_input(
        "Annual Marketing Budget - Year 3 ($)",
        min_value=200000,
        max_value=5000000,
        value=int(config.get('marketing_budget_y3', 1000000)),
        step=100000,
        format="%d",
        key="marketing_budget_y3"
    )
    if st.session_state.get('marketing_budget_y3', 1000000) != int(config.get('marketing_budget_y3', 1000000)):
        update_config_value('marketing_budget_y3', marketing_budget_y3)

    marketing_budget_growth = st.slider(
        "Marketing Budget Annual Growth Rate (Years 4+)",
        min_value=0.05,
        max_value=0.50,
        value=float(config.get('marketing_budget_growth', 0.20)),
        step=0.05,
        format="%.2f",
        key="marketing_budget_growth"
    )
    if st.session_state.get('marketing_budget_growth', 0.20) != float(config.get('marketing_budget_growth', 0.20)):
        update_config_value('marketing_budget_growth', marketing_budget_growth)

    st.markdown("#### Infrastructure & Office Costs")
    dev_tools_per_dev = st.number_input("Annual Dev Tools Cost per Developer ($)", value=int(
        config.get('dev_tools_per_dev', 5000)), step=500, format="%d")
    if st.session_state.get('dev_tools_per_dev') != int(config.get('dev_tools_per_dev', 5000)):
        update_config_value('dev_tools_per_dev', dev_tools_per_dev)

    cloud_infra_per_customer = st.number_input("Monthly Cloud Cost per Customer ($)", value=int(
        config.get('cloud_infra_per_customer', 40)), step=5, format="%d")
    if st.session_state.get('cloud_infra_per_customer') != int(config.get('cloud_infra_per_customer', 40)):
        update_config_value('cloud_infra_per_customer',
                            cloud_infra_per_customer)

    cloud_fixed_monthly = st.number_input("Fixed Monthly Cloud Costs ($)", value=int(
        config.get('cloud_fixed_monthly', 2500)), step=500, format="%d")
    if st.session_state.get('cloud_fixed_monthly') != int(config.get('cloud_fixed_monthly', 2500)):
        update_config_value('cloud_fixed_monthly', cloud_fixed_monthly)

    office_per_employee = st.number_input("Monthly Office Cost per Employee ($)", value=int(
        config.get('office_per_employee', 400)), step=50, format="%d")
    if st.session_state.get('office_per_employee') != int(config.get('office_per_employee', 400)):
        update_config_value('office_per_employee', office_per_employee)

    st.markdown("#### SG&A Expenses")
    ga_monthly_base = st.number_input("Base Monthly SG&A Expenses ($)", value=int(
        config.get('ga_monthly_base', 15000)), step=1000, format="%d")
    if st.session_state.get('ga_monthly_base') != int(config.get('ga_monthly_base', 15000)):
        update_config_value('ga_monthly_base', ga_monthly_base)

    ga_percent_revenue = st.slider("SG&A as % of Revenue (when higher than base)", min_value=0.03,
                                   max_value=0.15, value=float(config.get('ga_percent_revenue', 0.08)), step=0.01, format="%.2f")
    if st.session_state.get('ga_percent_revenue') != float(config.get('ga_percent_revenue', 0.08)):
        update_config_value('ga_percent_revenue', ga_percent_revenue)

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

    # Customer tracking by segment
    sme_customers = []
    enterprise_customers = []
    startup_customers = []
    total_customers = []

    # ARR tracking by segment
    sme_arr_values = []
    enterprise_arr_values = []
    startup_arr_values = []

    # Revenue tracking
    monthly_arr = []
    monthly_ps_revenue = []
    monthly_total_revenue = []

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

    # Professional services tracking
    ps_revenue = []
    ps_costs = []
    ps_margin_dollars = []

    # Summary financials
    total_expenses = []
    gross_profit = []
    ebitda = []
    cash_balance = []

    # Set initial values for each customer segment
    current_sme_customers = config.get('sme_initial_customers', 3)
    current_enterprise_customers = config.get(
        'enterprise_initial_customers', 1)
    current_startup_customers = config.get('startup_initial_customers', 1)

    # Set initial ARR values by segment
    current_sme_arr = config.get('sme_initial_arr', 25000)
    current_enterprise_arr = config.get('enterprise_initial_arr', 120000)
    current_startup_arr = config.get('startup_initial_arr', 15000)

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

        # Update ARR annually (at the beginning of each year)
        if month > 0 and month_of_year == 1:
            # Update ARR with annual growth rate per segment
            current_sme_arr *= (1 + config.get('sme_arr_growth_rate', 0.15))
            current_enterprise_arr *= (1 +
                                       config.get('enterprise_arr_growth_rate', 0.20))
            current_startup_arr *= (1 +
                                    config.get('startup_arr_growth_rate', 0.25))

            # Update salaries with annual increases
            current_salary_dev *= (1 + annual_salary_increase)
            current_salary_sales *= (1 + annual_salary_increase)
            current_salary_ops *= (1 + annual_salary_increase)
            current_salary_ga *= (1 + annual_salary_increase)

        # Determine growth rates based on year for each customer segment
        # SME segment growth/churn
        if year == 1:
            sme_customer_growth_rate = config.get('sme_cust_growth_y1', 0.10)
            sme_churn_rate = config.get('sme_churn_y1', 0.20)
        elif year <= 3:
            sme_customer_growth_rate = config.get(
                'sme_cust_growth_y2_y3', 0.075)
            sme_churn_rate = config.get('sme_churn_y2', 0.15) if year == 2 else config.get(
                'sme_churn_y3_plus', 0.10)
        else:
            sme_customer_growth_rate = config.get(
                'sme_cust_growth_y4_plus', 0.04)
            sme_churn_rate = config.get('sme_churn_y3_plus', 0.10)

        # Enterprise segment growth/churn
        if year == 1:
            enterprise_customer_growth_rate = config.get(
                'enterprise_cust_growth_y1', 0.08)
            enterprise_churn_rate = config.get('enterprise_churn_y1', 0.10)
        elif year <= 3:
            enterprise_customer_growth_rate = config.get(
                'enterprise_cust_growth_y2_y3', 0.06)
            enterprise_churn_rate = config.get(
                'enterprise_churn_y2', 0.08) if year == 2 else config.get('enterprise_churn_y3_plus', 0.06)
        else:
            enterprise_customer_growth_rate = config.get(
                'enterprise_cust_growth_y4_plus', 0.03)
            enterprise_churn_rate = config.get(
                'enterprise_churn_y3_plus', 0.06)

        # Startup segment growth/churn
        if year == 1:
            startup_customer_growth_rate = config.get(
                'startup_cust_growth_y1', 0.15)
            startup_churn_rate = config.get('startup_churn_y1', 0.30)
        elif year <= 3:
            startup_customer_growth_rate = config.get(
                'startup_cust_growth_y2_y3', 0.12)
            startup_churn_rate = config.get(
                'startup_churn_y2', 0.25) if year == 2 else config.get('startup_churn_y3_plus', 0.20)
        else:
            startup_customer_growth_rate = config.get(
                'startup_cust_growth_y4_plus', 0.08)
            startup_churn_rate = config.get('startup_churn_y3_plus', 0.20)

        # Get COGS and marketing percentages based on year
        if year == 1:
            cogs_percent = cogs_y1
            marketing_budget = marketing_budget_y1
        elif year == 2:
            cogs_percent = cogs_y2
            marketing_budget = marketing_budget_y2
        elif year == 3:
            cogs_percent = cogs_y3
            marketing_budget = marketing_budget_y3
        else:
            cogs_percent = cogs_y4_plus
            marketing_budget = marketing_budget_y3 * \
                (1 + marketing_budget_growth) ** (year - 3)

        # Headcount growth rates by department based on year
        if year == 1:
            dev_growth_rate = hc_growth_dev_y1
            sales_growth_rate = hc_growth_sales_y1
            ops_growth_rate = hc_growth_ops_y1
            ga_growth_rate = hc_growth_ga_y1
        elif year == 2:
            dev_growth_rate = hc_growth_dev_y2
            sales_growth_rate = hc_growth_sales_y2
            ops_growth_rate = hc_growth_ops_y2
            ga_growth_rate = hc_growth_ga_y2
        elif year == 3:
            dev_growth_rate = hc_growth_dev_y3
            sales_growth_rate = hc_growth_sales_y3
            ops_growth_rate = hc_growth_ops_y3
            ga_growth_rate = hc_growth_ga_y3
        else:
            dev_growth_rate = hc_growth_dev_y4_plus
            sales_growth_rate = hc_growth_sales_y4_plus
            ops_growth_rate = hc_growth_ops_y4_plus
            ga_growth_rate = hc_growth_ga_y4_plus

        # Calculate customer movement for each segment with exponential growth curve
        # Apply a multiplier based on where we are in the year progression to model
        # exponential growth (slow start, accelerating in following years)

        # Get curve parameters from config
        growth_curve_steepness = config.get('growth_curve_steepness', 2.0)
        sme_curve_factor = config.get('sme_curve_factor', 1.0)
        enterprise_curve_factor = config.get('enterprise_curve_factor', 1.0)
        startup_curve_factor = config.get('startup_curve_factor', 1.0)

        # Calculate year progression (0 to 1 scale within each year)
        year_progression = (month % 12) / 12

        # SME segment with exponential growth pattern
        # Start slow in early months of each year, accelerate later
        # Apply user-configurable curve steepness and segment factor
        sme_growth_multiplier = 0.5 + \
            (year_progression ** growth_curve_steepness) * 1.5 * sme_curve_factor
        new_sme_customers = max(
            1, round(current_sme_customers * sme_customer_growth_rate * sme_growth_multiplier / 12))

        # Enterprise segment - apply user-configurable curve steepness and segment factor
        enterprise_growth_multiplier = 0.3 + \
            (year_progression ** (growth_curve_steepness * 1.2)) * \
            2.1 * enterprise_curve_factor
        new_enterprise_customers = max(
            0, round(current_enterprise_customers * enterprise_customer_growth_rate * enterprise_growth_multiplier / 12))

        # Startup segment - apply user-configurable curve steepness and segment factor
        startup_growth_multiplier = 0.7 + \
            (year_progression ** (growth_curve_steepness * 0.8)) * \
            1.2 * startup_curve_factor
        new_startup_customers = max(
            0, round(current_startup_customers * startup_customer_growth_rate * startup_growth_multiplier / 12))

        # Handle customer churn based on contract length
        # Get contract lengths for each segment (in years)
        sme_contract_length = config.get('sme_contract_length', 1.0)
        enterprise_contract_length = config.get(
            'enterprise_contract_length', 2.0)
        startup_contract_length = config.get('startup_contract_length', 1.0)

        # Convert contract length to months for calculation
        sme_contract_months = int(sme_contract_length * 12)
        enterprise_contract_months = int(enterprise_contract_length * 12)
        startup_contract_months = int(startup_contract_length * 12)

        # Determine if we're in a churn-eligible month based on contract length
        # This simulates customers only being able to churn when their contract is up for renewal
        current_month = month + 1  # 1-based for clarity

        # For SME segment
        is_sme_churn_month = (current_month % sme_contract_months == 0)
        # For Enterprise segment
        is_enterprise_churn_month = (
            current_month % enterprise_contract_months == 0)
        # For Startup segment
        is_startup_churn_month = (current_month % startup_contract_months == 0)

        # Apply churn only in eligible months, with no churn in year 1
        if year == 1:
            # No churn in first year
            monthly_sme_churn_rate = 0
            monthly_enterprise_churn_rate = 0
            monthly_startup_churn_rate = 0
            churned_sme_customers = 0
            churned_enterprise_customers = 0
            churned_startup_customers = 0
        else:
            # Apply normalized churn rates for eligible months
            # We multiply by contract length to concentrate the churn in specific months
            if is_sme_churn_month:
                monthly_sme_churn_rate = sme_churn_rate * sme_contract_length
                churned_sme_customers = round(
                    current_sme_customers * monthly_sme_churn_rate)
            else:
                monthly_sme_churn_rate = 0
                churned_sme_customers = 0

            if is_enterprise_churn_month:
                monthly_enterprise_churn_rate = enterprise_churn_rate * enterprise_contract_length
                churned_enterprise_customers = round(
                    current_enterprise_customers * monthly_enterprise_churn_rate)
            else:
                monthly_enterprise_churn_rate = 0
                churned_enterprise_customers = 0

            if is_startup_churn_month:
                monthly_startup_churn_rate = startup_churn_rate * startup_contract_length
                churned_startup_customers = round(
                    current_startup_customers * monthly_startup_churn_rate)
            else:
                monthly_startup_churn_rate = 0
                churned_startup_customers = 0

        # Update customer counts
        current_sme_customers = current_sme_customers + \
            new_sme_customers - churned_sme_customers
        current_enterprise_customers = current_enterprise_customers + \
            new_enterprise_customers - churned_enterprise_customers
        current_startup_customers = current_startup_customers + \
            new_startup_customers - churned_startup_customers

        # Calculate total customers
        total_current_customers = current_sme_customers + \
            current_enterprise_customers + current_startup_customers

        # Calculate ARR by segment with customer acquisition influence
        # Now directly tie ARR growth to customer acquisition patterns
        monthly_sme_arr_per_customer = current_sme_arr / 12
        monthly_enterprise_arr_per_customer = current_enterprise_arr / 12
        monthly_startup_arr_per_customer = current_startup_arr / 12

        # Apply additional multiplier for new customers to account for higher pricing on new deals
        # Default 5% premium on new customers
        new_customer_premium = config.get('new_customer_premium', 1.05)

        # Calculate monthly ARR, with different rates for existing vs. new customers
        monthly_sme_arr = (
            # Existing customers at standard rate
            ((current_sme_customers - new_sme_customers) * monthly_sme_arr_per_customer) +
            # New customers with premium
            (new_sme_customers * monthly_sme_arr_per_customer * new_customer_premium)
        )

        monthly_enterprise_arr = (
            # Existing customers at standard rate
            ((current_enterprise_customers - new_enterprise_customers) * monthly_enterprise_arr_per_customer) +
            # New customers with premium
            (new_enterprise_customers *
             monthly_enterprise_arr_per_customer * new_customer_premium)
        )

        monthly_startup_arr = (
            # Existing customers at standard rate
            ((current_startup_customers - new_startup_customers) * monthly_startup_arr_per_customer) +
            # New customers with premium
            (new_startup_customers *
             monthly_startup_arr_per_customer * new_customer_premium)
        )

        # Calculate total monthly recurring revenue
        monthly_arr_total = monthly_sme_arr + \
            monthly_enterprise_arr + monthly_startup_arr

        # Calculate professional services revenue
        ps_percent = config.get('ps_percent_of_arr', 0.30)
        ps_growth = config.get('ps_growth_rate', 0.10)
        ps_margin_pct = config.get('ps_margin', 0.40)

        # Professional services revenue is a percentage of ARR with annual growth
        if month == 0:
            monthly_ps_rev = monthly_arr_total * ps_percent
        else:
            # Apply annual growth to ps_percent
            if month_of_year == 1 and month > 0:
                ps_percent *= (1 + ps_growth)
            monthly_ps_rev = monthly_arr_total * ps_percent

        # Calculate professional services costs and margin
        monthly_ps_cost = monthly_ps_rev * (1 - ps_margin_pct)
        monthly_ps_margin = monthly_ps_rev * ps_margin_pct

        # Total monthly revenue (ARR + PS)
        monthly_total_rev = monthly_arr_total + monthly_ps_rev

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
        # 1. COGS for ARR (software delivery costs)
        base_monthly_cogs = monthly_arr_total * cogs_percent

        # Development costs (part of COGS)
        monthly_salary_dev = (
            current_headcount_dev * current_salary_dev * (1 + benefits_multiplier)) / 12
        monthly_dev_tools = (current_headcount_dev * dev_tools_per_dev) / 12

        # Apply efficiency to all dev costs within COGS
        monthly_salary_dev *= efficiency_multiplier
        monthly_dev_tools *= efficiency_multiplier

        # Total COGS (ARR costs + development costs + professional services costs)
        monthly_cogs = base_monthly_cogs + monthly_salary_dev + \
            monthly_dev_tools + monthly_ps_cost

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
        monthly_marketing = marketing_budget / 12

        # Cloud infrastructure - based on total customers
        monthly_cloud = cloud_fixed_monthly + \
            (total_current_customers * cloud_infra_per_customer)

        # Office costs
        monthly_office = total_headcount * office_per_employee

        # SG&A expenses (non-salary)
        monthly_ga = max(
            ga_monthly_base, monthly_total_rev * ga_percent_revenue)

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

        # Calculate gross profit (Total revenue - COGS)
        monthly_gross_profit = monthly_total_rev - monthly_cogs

        # EBITDA
        monthly_ebitda = monthly_total_rev - monthly_total_expenses

        # Cash balance
        current_cash += monthly_ebitda

        # Append values to arrays
        # Customer tracking
        sme_customers.append(current_sme_customers)
        enterprise_customers.append(current_enterprise_customers)
        startup_customers.append(current_startup_customers)
        total_customers.append(total_current_customers)

        # ARR tracking
        sme_arr_values.append(current_sme_arr)
        enterprise_arr_values.append(current_enterprise_arr)
        startup_arr_values.append(current_startup_arr)

        # Revenue tracking
        monthly_arr.append(monthly_arr_total)
        monthly_ps_revenue.append(monthly_ps_rev)
        monthly_total_revenue.append(monthly_total_rev)

        # Headcount tracking
        headcount_dev.append(current_headcount_dev)
        headcount_sales.append(current_headcount_sales)
        headcount_ops.append(current_headcount_ops)
        headcount_ga.append(current_headcount_ga)
        headcount_total.append(total_headcount)

        # Expense tracking
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

        # Professional services tracking
        ps_revenue.append(monthly_ps_rev)
        ps_costs.append(monthly_ps_cost)
        ps_margin_dollars.append(monthly_ps_margin)

        # Summary financials
        total_expenses.append(monthly_total_expenses)
        gross_profit.append(monthly_gross_profit)
        ebitda.append(monthly_ebitda)
        cash_balance.append(current_cash)

    # Create monthly dataframe
    monthly_df = pd.DataFrame({
        'date': dates,
        'year': year_num,
        'month': [d.month for d in dates],
        'sme_customers': sme_customers,
        'enterprise_customers': enterprise_customers,
        'startup_customers': startup_customers,
        'total_customers': total_customers,
        'sme_arr': sme_arr_values,
        'enterprise_arr': enterprise_arr_values,
        'startup_arr': startup_arr_values,
        'monthly_arr': monthly_arr,
        'monthly_ps_revenue': monthly_ps_revenue,
        'monthly_total_revenue': monthly_total_revenue,
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
        'ps_revenue': ps_revenue,
        'ps_costs': ps_costs,
        'ps_margin': ps_margin_dollars,
        'total_expenses': total_expenses,
        'gross_profit': gross_profit,
        'ebitda': ebitda,
        'cash_balance': cash_balance
    })

    # Create annual summary with the new segmented data
    annual_summary = []

    for year in range(1, 7):
        year_data = monthly_df[monthly_df['year'] == year]

        # Year-end values
        year_end = year_data.iloc[-1]

        # Annual aggregates
        annual_arr = year_data['monthly_arr'].sum()
        annual_ps_revenue = year_data['monthly_ps_revenue'].sum()
        annual_total_revenue = year_data['monthly_total_revenue'].sum()
        annual_expenses = year_data['total_expenses'].sum()
        annual_ebitda = annual_total_revenue - annual_expenses
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

        # Professional services
        annual_ps_revenue = year_data['ps_revenue'].sum()
        annual_ps_costs = year_data['ps_costs'].sum()
        annual_ps_margin = year_data['ps_margin'].sum()

        # Calculate department totals
        annual_dev_total = annual_salary_dev + annual_dev_tools
        annual_sales_total = annual_salary_sales + annual_marketing
        annual_ops_total = annual_salary_ops + annual_cloud
        annual_ga_total = annual_salary_ga + annual_ga + annual_office

        annual_summary.append({
            'year': start_year + year - 1,
            'annual_arr': annual_arr,
            'annual_ps_revenue': annual_ps_revenue,
            'annual_total_revenue': annual_total_revenue,
            'annual_expenses': annual_expenses,
            'annual_ebitda': annual_ebitda,
            'annual_gross_profit': annual_gross_profit,
            'annual_cogs': annual_cogs,
            'annual_dev_total': annual_dev_total,
            'annual_sales_total': annual_sales_total,
            'annual_ops_total': annual_ops_total,
            'annual_ga_total': annual_ga_total,
            'annual_ps_costs': annual_ps_costs,
            'annual_ps_margin': annual_ps_margin,
            'year_end_sme_customers': year_end['sme_customers'],
            'year_end_enterprise_customers': year_end['enterprise_customers'],
            'year_end_startup_customers': year_end['startup_customers'],
            'year_end_total_customers': year_end['total_customers'],
            'year_end_headcount_total': year_end['headcount_total'],
            'year_end_headcount_dev': year_end['headcount_dev'],
            'year_end_headcount_sales': year_end['headcount_sales'],
            'year_end_headcount_ops': year_end['headcount_ops'],
            'year_end_headcount_ga': year_end['headcount_ga'],
            'year_end_cash_balance': year_end['cash_balance'],
            'gross_margin_percent': (annual_gross_profit / annual_total_revenue) * 100 if annual_total_revenue > 0 else 0,
            'ebitda_margin_percent': (annual_ebitda / annual_total_revenue) * 100 if annual_total_revenue > 0 else 0
        })

    annual_df = pd.DataFrame(annual_summary)

    # Format for chart display (millions)
    chart_data = []

    for _, row in annual_df.iterrows():
        revenue_millions = round(row['annual_total_revenue'] / 1000000)
        ebitda_millions = round(row['annual_ebitda'] / 1000000)
        arr_millions = round(row['annual_arr'] / 1000000, 1)
        ps_millions = round(row['annual_ps_revenue'] / 1000000, 1)

        revenue_label = f"${revenue_millions}M"
        if ebitda_millions >= 0:
            ebitda_label = f"${ebitda_millions}M"
        else:
            ebitda_label = f"-${abs(ebitda_millions)}M"

        chart_data.append({
            'year': row['year'],
            'revenue_millions': revenue_millions,
            'ebitda_millions': ebitda_millions,
            'arr_millions': arr_millions,
            'ps_millions': ps_millions,
            'revenue_label': revenue_label,
            'ebitda_label': ebitda_label,
            'sme_customers': int(row['year_end_sme_customers']),
            'enterprise_customers': int(row['year_end_enterprise_customers']),
            'startup_customers': int(row['year_end_startup_customers']),
            'total_customers': int(row['year_end_total_customers'])
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
            'ps_costs': round(row['annual_ps_costs'] / 1000000, 1),
            'ps_margin': round(row['annual_ps_margin'] / 1000000, 1),
            'headcount_dev': int(row['year_end_headcount_dev']),
            'headcount_sales': int(row['year_end_headcount_sales']),
            'headcount_ops': int(row['year_end_headcount_ops']),
            'headcount_ga': int(row['year_end_headcount_ga']),
            'headcount_total': int(row['year_end_headcount_total']),
            'sme_customers': int(row['year_end_sme_customers']),
            'enterprise_customers': int(row['year_end_enterprise_customers']),
            'startup_customers': int(row['year_end_startup_customers']),
        })

    department_df = pd.DataFrame(department_data)

    return monthly_df, annual_df, chart_df, department_df


# Run the model with current parameters
monthly_df, annual_df, chart_df, department_df = generate_financial_model()

# Display tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Key Metrics & Chart", "Annual Summary", "Department Breakdown", "Growth Curve Analysis", "Export Data"])

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
            x, chart_df['total_customers'], 'o-', color='#FF4500', linewidth=2, markersize=8, label='Customers')

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
        for i, customer in enumerate(chart_df['total_customers']):
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
        ax2.set_ylim(0, chart_df['total_customers'].max() * 1.2)

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
        final_customers = int(final_year['year_end_total_customers'])
        final_headcount = int(final_year['year_end_headcount_total'])
        final_revenue = final_year['annual_total_revenue'] / 1000000
        final_ebitda = final_year['annual_ebitda'] / 1000000
        final_margin = final_year['ebitda_margin_percent']

        # Customer metrics for year 5
        year5 = annual_df.iloc[4] if len(annual_df) > 4 else None

        if year5 is not None:
            # LTV calculation
            avg_annual_revenue = year5['annual_total_revenue'] / \
                year5['year_end_total_customers']
            customer_lifetime = 1 / \
                (config.get('sme_churn_y3_plus', 0.10)) if config.get(
                    'sme_churn_y3_plus', 0.10) > 0 else 10
            gross_margin = year5['gross_margin_percent'] / 100
            ltv = avg_annual_revenue * customer_lifetime * gross_margin

            # CAC approximation (simplified)
            y5_customers = year5['year_end_total_customers']
            y4_customers = annual_df.iloc[3]['year_end_total_customers'] if len(
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
    for col in ['annual_total_revenue', 'annual_expenses', 'annual_ebitda', 'annual_gross_profit',
                'annual_cogs', 'annual_dev_total', 'annual_sales_total', 'annual_ops_total',
                'annual_ga_total', 'year_end_cash_balance']:
        display_annual[col] = display_annual[col] / 1000000

    # Rename columns for better display
    display_annual = display_annual.rename(columns={
        'year': 'Year',
        'annual_total_revenue': 'Revenue ($M)',
        'annual_expenses': 'Expenses ($M)',
        'annual_ebitda': 'EBITDA ($M)',
        'annual_gross_profit': 'Gross Profit ($M)',
        'annual_cogs': 'COGS ($M)',
        'annual_dev_total': 'Development ($M)',
        'annual_sales_total': 'Sales & Marketing ($M)',
        'annual_ops_total': 'Operations ($M)',
        'annual_ga_total': 'G&A ($M)',
        'year_end_total_customers': 'Customers',
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
        'ps_costs': 'PS Costs ($M)',
        'ps_margin': 'PS Margin ($M)',
        'headcount_dev': 'Dev HC',
        'headcount_sales': 'Sales HC',
        'headcount_ops': 'Ops HC',
        'headcount_ga': 'G&A HC',
        'headcount_total': 'Total HC',
        'sme_customers': 'SME Customers',
        'enterprise_customers': 'Enterprise Customers',
        'startup_customers': 'Startup Customers'
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

    # Display segment breakdown visualization
    st.subheader("Customer Segment Breakdown")

    # Create a stacked bar chart for customer segments
    fig, ax = plt.subplots(figsize=(12, 7))

    x = display_dept['Year']
    width = 0.75

    # Customer segments as stacked bars
    sme_bars = ax.bar(
        x, display_dept['SME Customers'], width, label='SME', color='#3498db')
    enterprise_bars = ax.bar(x, display_dept['Enterprise Customers'], width,
                             bottom=display_dept['SME Customers'], label='Enterprise', color='#2c3e50')

    # Calculate the bottom position for startup bars
    enterprise_bottom = display_dept['SME Customers'] + \
        display_dept['Enterprise Customers']
    startup_bars = ax.bar(x, display_dept['Startup Customers'], width,
                          bottom=enterprise_bottom, label='Startup', color='#e74c3c')

    # Set labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title('Customer Segments Breakdown', fontsize=16, fontweight='bold')

    # Add legend
    ax.legend(loc='upper left')

    # Add total customer labels
    for i, year_data in enumerate(display_dept.iterrows()):
        _, row = year_data
        total_customers = row['SME Customers'] + \
            row['Enterprise Customers'] + row['Startup Customers']
        ax.text(row['Year'], total_customers + 2,
                f"Total: {total_customers}", ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # Create ARR breakdown visualization
    st.subheader("Annual Recurring Revenue by Segment")

    # Calculate ARR by segment for visualization
    arr_by_segment = []
    ps_revenue_data = []

    for _, row in annual_df.iterrows():
        year = row['year']
        sme_arr = row['year_end_sme_customers'] * config.get('sme_initial_arr', 25000) * (
            1 + config.get('sme_arr_growth_rate', 0.15)) ** (year - start_year)
        enterprise_arr = row['year_end_enterprise_customers'] * config.get('enterprise_initial_arr', 120000) * (
            1 + config.get('enterprise_arr_growth_rate', 0.20)) ** (year - start_year)
        startup_arr = row['year_end_startup_customers'] * config.get('startup_initial_arr', 15000) * (
            1 + config.get('startup_arr_growth_rate', 0.25)) ** (year - start_year)

        # Convert to millions for display
        sme_arr_m = sme_arr / 1000000
        enterprise_arr_m = enterprise_arr / 1000000
        startup_arr_m = startup_arr / 1000000
        ps_revenue_m = row['annual_ps_revenue'] / 1000000

        arr_by_segment.append({
            'year': year,
            'sme_arr_m': sme_arr_m,
            'enterprise_arr_m': enterprise_arr_m,
            'startup_arr_m': startup_arr_m,
            'total_arr_m': sme_arr_m + enterprise_arr_m + startup_arr_m,
            'ps_revenue_m': ps_revenue_m,
            'total_revenue_m': sme_arr_m + enterprise_arr_m + startup_arr_m + ps_revenue_m
        })

    # Create DataFrame for ARR visualization
    arr_df = pd.DataFrame(arr_by_segment)

    # Stacked bar chart for ARR by segment
    fig, ax = plt.subplots(figsize=(12, 7))

    x = arr_df['year']
    width = 0.75

    # ARR segments as stacked bars
    sme_arr_bars = ax.bar(x, arr_df['sme_arr_m'],
                          width, label='SME ARR', color='#3498db')
    enterprise_arr_bars = ax.bar(x, arr_df['enterprise_arr_m'], width,
                                 bottom=arr_df['sme_arr_m'], label='Enterprise ARR', color='#2c3e50')

    # Calculate bottom for startup ARR bars
    enterprise_arr_bottom = arr_df['sme_arr_m'] + arr_df['enterprise_arr_m']
    startup_arr_bars = ax.bar(x, arr_df['startup_arr_m'], width,
                              bottom=enterprise_arr_bottom, label='Startup ARR', color='#e74c3c')

    # Calculate bottom for PS revenue bars
    arr_bottom = arr_df['sme_arr_m'] + \
        arr_df['enterprise_arr_m'] + arr_df['startup_arr_m']
    ps_revenue_bars = ax.bar(x, arr_df['ps_revenue_m'], width,
                             bottom=arr_bottom, label='Professional Services', color='#27ae60')

    # Set labels
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Revenue ($M)', fontsize=12)
    ax.set_title('Revenue Breakdown by Segment',
                 fontsize=16, fontweight='bold')

    # Add legend
    ax.legend(loc='upper left')

    # Add total revenue labels
    for i, row in arr_df.iterrows():
        total_revenue = row['total_revenue_m']
        ax.text(row['year'], total_revenue + 1,
                f"${total_revenue:.1f}M", ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

with tab4:
    st.subheader("Growth Curve Analysis - Customer Acquisition Patterns")

    # Create tabs for different views within the Growth Curve Analysis
    gc_tab1, gc_tab2, gc_tab3 = st.tabs(
        ["Monthly Growth Visualization", "Annual Growth Summary", "S-Curve Parameters Impact"])

    with gc_tab1:
        st.markdown("### Monthly Customer Growth by Segment")
        st.markdown(
            "This visualization shows how customer acquisition follows an S-curve pattern throughout each year for each segment.")

        # Create a DataFrame for the monthly growth visualization
        monthly_growth_data = pd.DataFrame({
            'Month': range(1, len(monthly_df) + 1),
            'Date': monthly_df['date'],
            'Year': monthly_df['year'],
            'Month_of_Year': monthly_df['month'],
            'SME Customers': monthly_df['sme_customers'],
            'Enterprise Customers': monthly_df['enterprise_customers'],
            'Startup Customers': monthly_df['startup_customers'],
            'Total Customers': monthly_df['total_customers']
        })

        # Plot the customer growth curves
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot each segment
        ax.plot(monthly_growth_data['Month'], monthly_growth_data['SME Customers'],
                label='SME', color='#3498db', linewidth=2.5)
        ax.plot(monthly_growth_data['Month'], monthly_growth_data['Enterprise Customers'],
                label='Enterprise', color='#2c3e50', linewidth=2.5)
        ax.plot(monthly_growth_data['Month'], monthly_growth_data['Startup Customers'],
                label='Startup', color='#e74c3c', linewidth=2.5)
        ax.plot(monthly_growth_data['Month'], monthly_growth_data['Total Customers'],
                label='Total Customers', color='#27ae60', linewidth=3.5, linestyle='-')

        # Add year markers
        for year in range(1, 7):
            year_start_idx = monthly_growth_data[monthly_growth_data['Year']
                                                 == year].index[0]
            if year > 1:
                ax.axvline(x=year_start_idx, color='gray',
                           linestyle='--', alpha=0.7)
                ax.text(year_start_idx, ax.get_ylim()[1]*0.95, f"Year {year}",
                        ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

        # Customize the plot
        ax.set_title('Customer Growth Curves Over 6 Years', fontsize=16)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Number of Customers', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Show the plot
        st.pyplot(fig)

        # Add metrics to show how the growth curve steepness affects acquisition patterns
        st.markdown("### Growth Curve Steepness Impact")

        # Create columns for metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Growth Curve Steepness",
                      f"{config.get('growth_curve_steepness', 2.0):.1f}")
            st.caption("Higher values create steeper S-curves")

        with col2:
            # Calculate ratio of growth in last quarter vs first quarter of year 1
            y1_first_quarter = monthly_growth_data[(monthly_growth_data['Year'] == 1) &
                                                   (monthly_growth_data['Month_of_Year'] <= 3)]
            y1_last_quarter = monthly_growth_data[(monthly_growth_data['Year'] == 1) &
                                                  (monthly_growth_data['Month_of_Year'] >= 10)]

            first_q_growth = y1_first_quarter['Total Customers'].iloc[-1] - \
                y1_first_quarter['Total Customers'].iloc[0]
            last_q_growth = y1_last_quarter['Total Customers'].iloc[-1] - \
                y1_last_quarter['Total Customers'].iloc[0]

            if first_q_growth > 0:
                growth_ratio = last_q_growth / first_q_growth
                st.metric("Year 1 Q4:Q1 Growth Ratio", f"{growth_ratio:.2f}x")
                st.caption("Shows acceleration from beginning to end of year")
            else:
                st.metric("Year 1 Q4:Q1 Growth Ratio", "N/A")
                st.caption("Insufficient data to calculate ratio")

        with col3:
            # Calculate average new customers per month
            avg_new_customers = monthly_growth_data['Total Customers'].diff(
            ).mean()
            st.metric("Avg. New Customers/Month", f"{avg_new_customers:.1f}")
            st.caption("Average customer acquisition rate")

    with gc_tab2:
        st.markdown("### Annual Customer Growth by Segment")

        # Create a DataFrame for annual growth visualization
        annual_growth_data = pd.DataFrame({
            'Year': chart_df['year'],
            'SME Customers': chart_df['sme_customers'],
            'Enterprise Customers': chart_df['enterprise_customers'],
            'Startup Customers': chart_df['startup_customers'],
            'Total Customers': chart_df['total_customers']
        })

        # Display the data as a table
        st.dataframe(annual_growth_data, use_container_width=True)

        # Calculate growth rates year over year
        annual_growth_rates = pd.DataFrame(
            {'Year': annual_growth_data['Year'][1:]})

        for segment in ['SME Customers', 'Enterprise Customers', 'Startup Customers', 'Total Customers']:
            annual_growth_rates[f'{segment} Growth %'] = [
                ((annual_growth_data[segment].iloc[i] /
                 annual_growth_data[segment].iloc[i-1]) - 1) * 100
                for i in range(1, len(annual_growth_data))
            ]

        # Display growth rates
        st.markdown("### Year-over-Year Growth Rates (%)")
        st.dataframe(annual_growth_rates.round(1), use_container_width=True)

        # Create stacked area chart for customer segments
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create stacked area chart
        ax.stackplot(annual_growth_data['Year'],
                     annual_growth_data['SME Customers'],
                     annual_growth_data['Enterprise Customers'],
                     annual_growth_data['Startup Customers'],
                     labels=['SME', 'Enterprise', 'Startup'],
                     alpha=0.7,
                     colors=['#3498db', '#2c3e50', '#e74c3c'])

        # Add a line for total customers
        ax.plot(annual_growth_data['Year'], annual_growth_data['Total Customers'],
                color='#27ae60', linewidth=3, label='Total Customers')

        # Add data labels
        for i, row in annual_growth_data.iterrows():
            ax.text(row['Year'], row['Total Customers'] + 5,
                    f"{row['Total Customers']}", ha='center', fontweight='bold')

            # Add segment breakdown labels every other year
            if i % 2 == 0:
                sme_y_pos = row['SME Customers'] / 2
                ent_y_pos = row['SME Customers'] + \
                    row['Enterprise Customers'] / 2
                startup_y_pos = row['SME Customers'] + \
                    row['Enterprise Customers'] + row['Startup Customers'] / 2

                ax.text(row['Year'], sme_y_pos, f"SME: {row['SME Customers']}",
                        ha='center', color='white', fontweight='bold')
                ax.text(row['Year'], ent_y_pos, f"Ent: {row['Enterprise Customers']}",
                        ha='center', color='white', fontweight='bold')
                ax.text(row['Year'], startup_y_pos, f"Startup: {row['Startup Customers']}",
                        ha='center', color='white', fontweight='bold')

        # Set labels and title
        ax.set_title('Cumulative Customer Growth by Segment', fontsize=16)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Number of Customers', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

        # Create a bar chart showing the segment proportions
        st.markdown("### Customer Segment Distribution")

        # Calculate percentage distribution for each year
        segment_pcts = pd.DataFrame({'Year': annual_growth_data['Year']})

        for segment in ['SME Customers', 'Enterprise Customers', 'Startup Customers']:
            segment_pcts[f'{segment} %'] = (
                annual_growth_data[segment] / annual_growth_data['Total Customers'] * 100).round(1)

        # Plot the percentage distribution
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create the stacked percentage bars
        bottom = np.zeros(len(segment_pcts))

        for segment, color in zip(['SME Customers %', 'Enterprise Customers %', 'Startup Customers %'],
                                  ['#3498db', '#2c3e50', '#e74c3c']):
            segment_name = segment.split(' ')[0]
            ax.bar(segment_pcts['Year'], segment_pcts[segment], bottom=bottom,
                   label=segment_name, color=color, width=0.6)

            # Add percentage labels
            for i, value in enumerate(segment_pcts[segment]):
                if value > 5:  # Only show label if segment is at least 5%
                    ax.text(segment_pcts['Year'].iloc[i], bottom[i] + value/2,
                            f"{value:.1f}%", ha='center', va='center', color='white', fontweight='bold')

            bottom += segment_pcts[segment]

        # Customize plot
        ax.set_title('Customer Segment Distribution by Year (%)', fontsize=16)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Percentage of Customers', fontsize=14)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right')

        st.pyplot(fig)

    with gc_tab3:
        st.markdown("### S-Curve Parameters and Their Impact")

        st.markdown("""
        The S-curve growth pattern is controlled by several parameters that you can adjust in the sidebar:
        
        1. **Growth Curve Steepness** (currently set to `{:.1f}`): Controls how quickly growth accelerates within each year
        2. **Segment-specific factors**:
           - SME Growth Factor: `{:.1f}`
           - Enterprise Growth Factor: `{:.1f}`
           - Startup Growth Factor: `{:.1f}`
        
        These parameters control the shape of the exponential growth curve for each customer segment:
        """.format(
            config.get('growth_curve_steepness', 2.0),
            config.get('sme_curve_factor', 1.0),
            config.get('enterprise_curve_factor', 1.0),
            config.get('startup_curve_factor', 1.0)
        ))

        # Generate S-curve visualization for different parameters
        st.markdown("### Visualizing the S-Curve Shape")

        # Create sample data for s-curve visualization
        months = np.linspace(0, 12, 100)
        progress = months / 12  # Convert to 0-1 scale

        # Calculate curves for different steepness values
        steepness_values = [0.5, 1.0, 2.0, 3.0]
        curve_data = []

        # Current steepness
        current_steepness = config.get('growth_curve_steepness', 2.0)

        # Calculate curves including the current one
        if current_steepness not in steepness_values:
            steepness_values.append(current_steepness)
            steepness_values.sort()

        # Calculate visualization for the growth multipliers
        fig, ax = plt.subplots(figsize=(12, 6))

        for steepness in steepness_values:
            # Calculate the curve for this steepness
            growth_multiplier = 0.5 + (progress ** steepness) * 1.5

            # Different line style for current value
            if steepness == current_steepness:
                ax.plot(months, growth_multiplier, linewidth=4,
                        label=f'Current ({steepness:.1f})', color='red')
            else:
                ax.plot(months, growth_multiplier, linewidth=2,
                        label=f'Steepness {steepness:.1f}', alpha=0.7)

        # Add segment-specific curves
        sme_multiplier = 0.5 + (progress ** current_steepness) * \
            1.5 * config.get('sme_curve_factor', 1.0)
        enterprise_multiplier = 0.3 + \
            (progress ** (current_steepness * 1.2)) * 2.1 * \
            config.get('enterprise_curve_factor', 1.0)
        startup_multiplier = 0.7 + \
            (progress ** (current_steepness * 0.8)) * \
            1.2 * config.get('startup_curve_factor', 1.0)

        ax.plot(months, sme_multiplier, '--',
                label='SME Segment', color='#3498db')
        ax.plot(months, enterprise_multiplier, '--',
                label='Enterprise Segment', color='#2c3e50')
        ax.plot(months, startup_multiplier, '--',
                label='Startup Segment', color='#e74c3c')

        # Customize the plot
        ax.set_title(
            'S-Curve Shapes with Different Steepness Values', fontsize=16)
        ax.set_xlabel('Month of Year', fontsize=14)
        ax.set_ylabel('Growth Multiplier', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax.set_xticks(np.arange(0, 13, 1))

        st.pyplot(fig)

        st.markdown("""
        ### How to Interpret the S-Curve
        
        The growth multiplier shows how customer acquisition varies throughout the year:
        
        - **Values < 1.0**: Growth slower than the base rate
        - **Values = 1.0**: Growth at exactly the base rate
        - **Values > 1.0**: Growth faster than the base rate
        
        Most SaaS businesses experience slower growth at the beginning of each year, with acceleration toward the end as:
        
        1. Marketing efforts compound
        2. Sales team becomes more effective
        3. Seasonal budget cycles favor Q4 spending
        
        Adjust the steepness and segment factors to match your expected growth patterns.
        """)

        # Display segment-specific curve information
        st.markdown("### Segment-Specific Growth Patterns")

        # Create columns for metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("SME - Year Start Multiplier", f"{0.5:.2f}x")
            st.metric("SME - Year End Multiplier",
                      f"{(0.5 + 1.5 * config.get('sme_curve_factor', 1.0)):.2f}x")

        with col2:
            st.metric("Enterprise - Year Start Multiplier", f"{0.3:.2f}x")
            st.metric("Enterprise - Year End Multiplier",
                      f"{(0.3 + 2.1 * config.get('enterprise_curve_factor', 1.0)):.2f}x")

        with col3:
            st.metric("Startup - Year Start Multiplier", f"{0.7:.2f}x")
            st.metric("Startup - Year End Multiplier",
                      f"{(0.7 + 1.2 * config.get('startup_curve_factor', 1.0)):.2f}x")

with tab5:
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
                'initial_funding', 'start_year', 'sme_initial_arr', 'enterprise_initial_arr', 'startup_initial_arr',
                'sme_arr_growth_rate', 'enterprise_arr_growth_rate', 'startup_arr_growth_rate',
                'sme_initial_customers', 'enterprise_initial_customers', 'startup_initial_customers',
                'sme_cust_growth_y1', 'sme_cust_growth_y2_y3', 'sme_cust_growth_y4_plus',
                'enterprise_cust_growth_y1', 'enterprise_cust_growth_y2_y3', 'enterprise_cust_growth_y4_plus',
                'startup_cust_growth_y1', 'startup_cust_growth_y2_y3', 'startup_cust_growth_y4_plus',
                'sme_churn_y1', 'sme_churn_y2', 'sme_churn_y3_plus',
                'enterprise_churn_y1', 'enterprise_churn_y2', 'enterprise_churn_y3_plus',
                'startup_churn_y1', 'startup_churn_y2', 'startup_churn_y3_plus',
                'ps_percent_of_arr', 'ps_margin', 'ps_growth_rate',
                'cogs_y1', 'cogs_y2', 'cogs_y3', 'cogs_y4_plus',
                'hc_dev_initial', 'hc_sales_initial', 'hc_ops_initial', 'hc_ga_initial',
                'hc_growth_dev_y1', 'hc_growth_dev_y2', 'hc_growth_dev_y3', 'hc_growth_dev_y4_plus',
                'hc_growth_sales_y1', 'hc_growth_sales_y2', 'hc_growth_sales_y3', 'hc_growth_sales_y4_plus',
                'hc_growth_ops_y1', 'hc_growth_ops_y2', 'hc_growth_ops_y3', 'hc_growth_ops_y4_plus',
                'hc_growth_ga_y1', 'hc_growth_ga_y2', 'hc_growth_ga_y3', 'hc_growth_ga_y4_plus',
                'salary_dev', 'salary_sales', 'salary_ops', 'salary_ga',
                'benefits_multiplier', 'annual_salary_increase',
                'marketing_budget_y1', 'marketing_budget_y2', 'marketing_budget_y3', 'marketing_budget_growth',
                'dev_tools_per_dev', 'cloud_infra_per_customer', 'cloud_fixed_monthly', 'office_per_employee',
                'ga_monthly_base', 'ga_percent_revenue',
                'efficiency_y3', 'efficiency_y4_plus'
            ],
            'value': [
                initial_funding, start_year, sme_initial_arr, enterprise_initial_arr, startup_initial_arr,
                sme_arr_growth_rate, enterprise_arr_growth_rate, startup_arr_growth_rate,
                sme_initial_customers, enterprise_initial_customers, startup_initial_customers,
                config.get('sme_cust_growth_y1', 0.10), config.get(
                    'sme_cust_growth_y2_y3', 0.075), config.get('sme_cust_growth_y4_plus', 0.04),
                config.get('enterprise_cust_growth_y1', 0.08), config.get(
                    'enterprise_cust_growth_y2_y3', 0.06), config.get('enterprise_cust_growth_y4_plus', 0.03),
                config.get('startup_cust_growth_y1', 0.15), config.get(
                    'startup_cust_growth_y2_y3', 0.12), config.get('startup_cust_growth_y4_plus', 0.08),
                config.get('sme_churn_y1', 0.20), config.get(
                    'sme_churn_y2', 0.15), config.get('sme_churn_y3_plus', 0.10),
                config.get('enterprise_churn_y1', 0.10), config.get(
                    'enterprise_churn_y2', 0.08), config.get('enterprise_churn_y3_plus', 0.06),
                config.get('startup_churn_y1', 0.30), config.get(
                    'startup_churn_y2', 0.25), config.get('startup_churn_y3_plus', 0.20),
                ps_percent_of_arr, ps_margin, ps_growth_rate,
                cogs_y1, cogs_y2, cogs_y3, cogs_y4_plus,
                hc_dev_initial, hc_sales_initial, hc_ops_initial, hc_ga_initial,
                hc_growth_dev_y1, hc_growth_dev_y2, hc_growth_dev_y3, hc_growth_dev_y4_plus,
                hc_growth_sales_y1, hc_growth_sales_y2, hc_growth_sales_y3, hc_growth_sales_y4_plus,
                hc_growth_ops_y1, hc_growth_ops_y2, hc_growth_ops_y3, hc_growth_ops_y4_plus,
                hc_growth_ga_y1, hc_growth_ga_y2, hc_growth_ga_y3, hc_growth_ga_y4_plus,
                salary_dev, salary_sales, salary_ops, salary_ga,
                benefits_multiplier, annual_salary_increase,
                marketing_budget_y1, marketing_budget_y2, marketing_budget_y3, marketing_budget_growth,
                dev_tools_per_dev, cloud_infra_per_customer, cloud_fixed_monthly, office_per_employee,
                ga_monthly_base, ga_percent_revenue,
                efficiency_y3, efficiency_y4_plus
            ],
            'description': [
                'Initial seed funding', 'Start year', 'SME initial ARR', 'Enterprise initial ARR', 'Startup initial ARR',
                'SME ARR growth rate', 'Enterprise ARR growth rate', 'Startup ARR growth rate',
                'SME initial customers', 'Enterprise initial customers', 'Startup initial customers',
                'SME customer growth rate year 1', 'SME customer growth rate years 2-3', 'SME customer growth rate years 4+',
                'Enterprise customer growth rate year 1', 'Enterprise customer growth rate years 2-3', 'Enterprise customer growth rate years 4+',
                'Startup customer growth rate year 1', 'Startup customer growth rate years 2-3', 'Startup customer growth rate years 4+',
                'SME churn rate year 1', 'SME churn rate year 2', 'SME churn rate years 3+',
                'Enterprise churn rate year 1', 'Enterprise churn rate year 2', 'Enterprise churn rate years 3+',
                'Startup churn rate year 1', 'Startup churn rate year 2', 'Startup churn rate years 3+',
                'Professional services percent of ARR', 'Professional services margin', 'Professional services growth rate',
                'COGS as % of revenue year 1', 'COGS as % of revenue year 2', 'COGS as % of revenue year 3', 'COGS as % of revenue years 4+',
                'Initial development headcount', 'Initial sales headcount', 'Initial operations headcount', 'Initial G&A headcount',
                'Development headcount growth rate year 1', 'Development headcount growth rate year 2', 'Development headcount growth rate year 3', 'Development headcount growth rate years 4+',
                'Sales headcount growth rate year 1', 'Sales headcount growth rate year 2', 'Sales headcount growth rate year 3', 'Sales headcount growth rate years 4+',
                'Operations headcount growth rate year 1', 'Operations headcount growth rate year 2', 'Operations headcount growth rate year 3', 'Operations headcount growth rate years 4+',
                'G&A headcount growth rate year 1', 'G&A headcount growth rate year 2', 'G&A headcount growth rate year 3', 'G&A headcount growth rate years 4+',
                'Average developer salary', 'Average sales salary', 'Average operations salary', 'Average G&A salary',
                'Benefits multiplier', 'Annual salary increase',
                'Marketing budget year 1', 'Marketing budget year 2', 'Marketing budget year 3', 'Marketing budget growth rate',
                'Annual dev tools cost per developer', 'Monthly cloud cost per customer', 'Fixed monthly cloud costs', 'Monthly office cost per employee',
                'Base monthly SG&A expenses', 'SG&A as % of revenue (when higher than base)',
                'Cost efficiency gains year 3', 'Cost efficiency gains years 4+'
            ]
        }

        params_df = pd.DataFrame(params_dict)
        st.markdown(get_download_link(params_df, "model_parameters",
                    "Download Model Parameters"), unsafe_allow_html=True)

    # Create full data export
    st.markdown("### Export All Data")

    # Create a zip file-like buffer
    if st.button("Export All Data (CSV)"):
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
