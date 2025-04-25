import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from models.growth_model import SaaSGrowthModel
from models.cost_model import AISaaSCostModel
from models.financial_model import SaaSFinancialModel
from app import (
    run_baseline_model, 
    run_growth_profile_model, 
    run_european_strategy,
    load_configs,
    save_reports
)

st.set_page_config(
    page_title="2025 Financial Model",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

def update_s_curve_params(config, segment, year, midpoint, steepness, max_monthly):
    """Update s-curve parameters for a specific segment and year"""
    if segment in config['s_curve'] and year in config['s_curve'][segment]:
        config['s_curve'][segment][year]['midpoint'] = midpoint
        config['s_curve'][segment][year]['steepness'] = steepness
        config['s_curve'][segment][year]['max_monthly'] = max_monthly
    return config

# Initialize session state for configurations
if 'revenue_config' not in st.session_state:
    st.session_state.revenue_config, st.session_state.cost_config = load_configs()

if 'growth_model' not in st.session_state:
    st.session_state.growth_model = None
    st.session_state.cost_model = None
    st.session_state.financial_model = None
    
# Function to save configurations to JSON files
def save_configs(revenue_config, cost_config):
    with open(os.path.join('configs', 'revenue_config.json'), 'w') as f:
        json.dump(revenue_config, f, indent=2)
        
    with open(os.path.join('configs', 'cost_config.json'), 'w') as f:
        json.dump(cost_config, f, indent=2)
    
    st.success("Configurations saved successfully!")

# Main app layout
st.title("2025 Financial Model Dashboard")

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")
    
    # Strategy selection
    strategy = st.selectbox(
        "Select Strategy",
        ["baseline", "profile", "european"],
        help="Choose the strategy to run the financial model"
    )
    
    # Growth profile options (for profile strategy)
    if strategy == "profile":
        profile = st.selectbox(
            "Growth Profile",
            ["conservative", "baseline", "aggressive", "hypergrowth"],
            help="Select the growth profile for customer acquisition"
        )
    
    # European strategy options
    if strategy == "european":
        breakeven_target = st.number_input(
            "Breakeven Target Month",
            min_value=1,
            max_value=72,
            value=24,
            help="Target month to achieve breakeven (1-72)"
        )
        
        revenue_target = st.number_input(
            "Annual Revenue Target ($)",
            min_value=1000000,
            max_value=500000000,
            value=10000000,
            step=1000000,
            format="%d",
            help="Annual revenue target in USD"
        )
        
        revenue_target_year = st.number_input(
            "Revenue Target Year",
            min_value=1,
            max_value=6,
            value=3,
            help="Year to achieve annual revenue target (1-6)"
        )
        
        max_iterations = st.number_input(
            "Max Optimization Iterations",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of iterations for optimization"
        )
    
    # Initial investment input (common to all strategies)
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000000,
        max_value=50000000,
        value=5000000,
        step=1000000,
        format="%d",
        help="Initial capital investment amount"
    )
    
    # Run model button
    if st.button("Run Model"):
        with st.spinner("Running model..."):
            if strategy == "baseline":
                st.session_state.growth_model, st.session_state.cost_model, st.session_state.financial_model = run_baseline_model(initial_investment)
            elif strategy == "profile":
                st.session_state.growth_model, st.session_state.cost_model, st.session_state.financial_model = run_growth_profile_model(profile, initial_investment)
            elif strategy == "european":
                st.session_state.growth_model, st.session_state.cost_model, st.session_state.financial_model = run_european_strategy(
                    breakeven_target,
                    revenue_target,
                    revenue_target_year,
                    initial_investment,
                    max_iterations
                )
        st.success(f"Model run complete with {strategy} strategy!")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Results Overview", 
    "Growth Configuration", 
    "Cost Configuration", 
    "Reports", 
    "Import/Export"
])

# Tab 1: Results Overview
with tab1:
    if st.session_state.financial_model is not None:
        st.header("Financial Summary")
        
        # Display key metrics table
        key_metrics = st.session_state.financial_model.get_key_metrics_table()
        st.dataframe(key_metrics)
        
        # Display financial summary plot
        st.subheader("Financial Summary Plot")
        fig = st.session_state.financial_model.plot_financial_summary()
        st.pyplot(fig)
        
        # Display break-even analysis plot
        st.subheader("Break-Even Analysis")
        fig = st.session_state.financial_model.plot_break_even_analysis()
        st.pyplot(fig)
        
        # Display unit economics
        st.subheader("Unit Economics")
        unit_economics = st.session_state.cost_model.calculate_unit_economics(st.session_state.growth_model)
        unit_economics_table = st.session_state.cost_model.display_unit_economics_table(unit_economics)
        st.dataframe(unit_economics_table)
        
        fig = st.session_state.cost_model.plot_unit_economics(unit_economics)
        st.pyplot(fig)
    else:
        st.info("Run the model to view results")

# Tab 2: Growth Configuration
with tab2:
    st.header("Revenue Model Configuration")
    
    revenue_config = st.session_state.revenue_config
    
    # Display general configuration
    with st.expander("General Configuration", expanded=True):
        projection_months = st.number_input(
            "Projection Months",
            min_value=12,
            max_value=120,
            value=revenue_config['projection_months'],
            step=12
        )
        revenue_config['projection_months'] = projection_months
        
        segments = st.multiselect(
            "Customer Segments",
            options=["Enterprise", "Mid-Market", "SMB", "Startup"],
            default=revenue_config['segments']
        )
        if segments:  # Only update if not empty
            revenue_config['segments'] = segments
    
    # Display segment-specific configurations
    with st.expander("Segment Configuration", expanded=True):
        # Create columns for segment-specific parameters
        segment_tabs = st.tabs(revenue_config['segments'])
        
        for i, segment in enumerate(revenue_config['segments']):
            with segment_tabs[i]:
                st.subheader(f"{segment} Configuration")
                
                # Initial ARR and customers
                col1, col2 = st.columns(2)
                with col1:
                    initial_arr = st.number_input(
                        f"Initial ARR ($) - {segment}",
                        min_value=1000,
                        max_value=1000000,
                        value=revenue_config['initial_arr'].get(segment, 10000),
                        step=1000
                    )
                    revenue_config['initial_arr'][segment] = initial_arr
                
                with col2:
                    initial_customers = st.number_input(
                        f"Initial Customers - {segment}",
                        min_value=0,
                        max_value=1000,
                        value=revenue_config['initial_customers'].get(segment, 0)
                    )
                    revenue_config['initial_customers'][segment] = initial_customers
                
                # Contract length and churn rates
                col1, col2 = st.columns(2)
                with col1:
                    contract_length = st.number_input(
                        f"Contract Length (years) - {segment}",
                        min_value=0.25,
                        max_value=5.0,
                        value=revenue_config['contract_length'].get(segment, 1.0),
                        step=0.25
                    )
                    revenue_config['contract_length'][segment] = contract_length
                
                with col2:
                    churn_rate = st.number_input(
                        f"Annual Churn Rate - {segment}",
                        min_value=0.01,
                        max_value=0.50,
                        value=revenue_config['churn_rates'].get(segment, 0.15),
                        step=0.01,
                        format="%.2f"
                    )
                    revenue_config['churn_rates'][segment] = churn_rate
                
                # Annual price increase
                annual_price_increase = st.number_input(
                    f"Annual Price Increase (%) - {segment}",
                    min_value=0.0,
                    max_value=20.0,
                    value=revenue_config['annual_price_increases'].get(segment, 0.0) * 100,
                    step=0.5,
                    format="%.1f"
                )
                revenue_config['annual_price_increases'][segment] = annual_price_increase / 100
                
                # S-curve parameters
                st.subheader("S-Curve Parameters by Year")
                
                # Initialize s_curve for segment if it doesn't exist
                if segment not in revenue_config['s_curve']:
                    revenue_config['s_curve'][segment] = {}
                
                # Create a year selector
                years = list(range(1, 7))
                s_curve_years = st.multiselect(
                    f"Select Years to Configure - {segment}",
                    options=years,
                    default=years
                )
                
                for year in s_curve_years:
                    # Initialize year data if needed
                    if year not in revenue_config['s_curve'][segment]:
                        revenue_config['s_curve'][segment][year] = {
                            "midpoint": 6,
                            "steepness": 0.5,
                            "max_monthly": 2
                        }
                    
                    # Display year configuration in columns
                    st.write(f"Year {year}")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        midpoint = st.number_input(
                            f"Midpoint (Month 1-12) - Y{year}",
                            min_value=1,
                            max_value=12,
                            value=revenue_config['s_curve'][segment][year].get('midpoint', 6),
                            key=f"mp_{segment}_{year}"
                        )
                    
                    with col2:
                        steepness = st.number_input(
                            f"Steepness - Y{year}",
                            min_value=0.05,
                            max_value=2.0,
                            value=revenue_config['s_curve'][segment][year].get('steepness', 0.5),
                            step=0.05,
                            format="%.2f",
                            key=f"st_{segment}_{year}"
                        )
                    
                    with col3:
                        max_monthly = st.number_input(
                            f"Max Monthly Customers - Y{year}",
                            min_value=0,
                            max_value=100,
                            value=revenue_config['s_curve'][segment][year].get('max_monthly', 2),
                            key=f"mm_{segment}_{year}"
                        )
                    
                    # Update config with the new values
                    revenue_config = update_s_curve_params(
                        revenue_config, segment, year, midpoint, steepness, max_monthly
                    )
    
    # Seasonality
    with st.expander("Seasonality Configuration", expanded=False):
        st.subheader("Monthly Seasonality Factors")
        st.write("Values above 1.0 increase growth, below 1.0 decrease growth")
        
        # Create 4 rows of 3 columns for all 12 months
        for row in range(4):
            cols = st.columns(3)
            for col in range(3):
                month_idx = row * 3 + col + 1
                if month_idx <= 12:
                    month_name = ["January", "February", "March", "April", "May", "June", 
                                "July", "August", "September", "October", "November", "December"][month_idx-1]
                    
                    seasonality = st.number_input(
                        f"{month_name} (Month {month_idx})",
                        min_value=0.5,
                        max_value=2.0,
                        value=revenue_config['seasonality'].get(str(month_idx), revenue_config['seasonality'].get(month_idx, 1.0)),
                        step=0.1,
                        format="%.1f"
                    )
                    
                    # Convert string keys to integers in seasonality (this should match app.py logic)
                    revenue_config['seasonality'][month_idx] = seasonality
    
    # Save button for revenue config
    if st.button("Save Revenue Configuration"):
        # Update session state
        st.session_state.revenue_config = revenue_config
        
        # Save to file
        save_configs(revenue_config, st.session_state.cost_config)

# Tab 3: Cost Configuration
with tab3:
    st.header("Cost Model Configuration")
    
    cost_config = st.session_state.cost_config
    
    # COGS Configuration
    with st.expander("COGS Configuration", expanded=True):
        st.subheader("Cost of Goods Sold (% of ARR)")
        
        # Display each COGS category
        cogs_categories = [k for k in cost_config['cogs'].keys() if k != "_comment"]
        
        for category in cogs_categories:
            cogs_value = st.number_input(
                f"{category.replace('_', ' ').title()} (% of ARR)",
                min_value=0.0,
                max_value=0.5,
                value=cost_config['cogs'].get(category, 0.0),
                step=0.01,
                format="%.2f"
            )
            cost_config['cogs'][category] = cogs_value
    
    # Headcount Configuration
    with st.expander("Headcount Configuration", expanded=True):
        st.subheader("Department Headcount")
        
        departments = [k for k in cost_config['headcount'].keys() if k != "_comment" and isinstance(cost_config['headcount'][k], dict)]
        
        dept_tabs = st.tabs(departments)
        
        for i, dept in enumerate(departments):
            with dept_tabs[i]:
                st.write(f"### {dept.replace('_', ' ').title()} Department")
                
                # Starting headcount and salary
                col1, col2 = st.columns(2)
                with col1:
                    starting_count = st.number_input(
                        f"Starting Headcount - {dept}",
                        min_value=0,
                        max_value=100,
                        value=cost_config['headcount'][dept].get('starting_count', 0)
                    )
                    cost_config['headcount'][dept]['starting_count'] = starting_count
                
                with col2:
                    avg_salary = st.number_input(
                        f"Average Annual Salary ($) - {dept}",
                        min_value=30000,
                        max_value=300000,
                        value=cost_config['headcount'][dept].get('avg_salary', 100000),
                        step=5000
                    )
                    cost_config['headcount'][dept]['avg_salary'] = avg_salary
                
                # Growth factors by year
                st.write("#### Growth Factors by Year")
                
                # Initialize growth_factors if it doesn't exist
                if 'growth_factors' not in cost_config['headcount'][dept]:
                    cost_config['headcount'][dept]['growth_factors'] = {}
                
                # Create columns for each year
                year_cols = st.columns(6)
                for year in range(1, 7):
                    with year_cols[year-1]:
                        growth_factor = st.number_input(
                            f"Year {year}",
                            min_value=0.5,
                            max_value=5.0,
                            value=cost_config['headcount'][dept]['growth_factors'].get(year, 
                                  cost_config['headcount'][dept]['growth_factors'].get(str(year), 1.0)),
                            step=0.1,
                            format="%.1f",
                            key=f"hc_{dept}_{year}"
                        )
                        cost_config['headcount'][dept]['growth_factors'][year] = growth_factor
    
    # Marketing Expenses
    with st.expander("Marketing Expenses", expanded=False):
        st.subheader("Marketing Expenses (% of ARR)")
        
        # Display each marketing category
        marketing_categories = [k for k in cost_config['marketing_expenses'].keys() if k != "_comment"]
        
        for category in marketing_categories:
            marketing_value = st.number_input(
                f"{category.replace('_', ' ').title()} (% of ARR)",
                min_value=0.0,
                max_value=0.5,
                value=cost_config['marketing_expenses'].get(category, 0.0),
                step=0.01,
                format="%.2f"
            )
            cost_config['marketing_expenses'][category] = marketing_value
        
        # Marketing efficiency by year
        st.write("#### Marketing Efficiency by Year")
        st.write("Values below 1.0 improve efficiency (reduce spending)")
        
        # Create columns for each year
        year_cols = st.columns(6)
        for year in range(1, 7):
            with year_cols[year-1]:
                efficiency = st.number_input(
                    f"Year {year}",
                    min_value=0.5,
                    max_value=1.5,
                    value=cost_config['marketing_efficiency'].get(year, 
                          cost_config['marketing_efficiency'].get(str(year), 1.0)),
                    step=0.05,
                    format="%.2f",
                    key=f"me_{year}"
                )
                cost_config['marketing_efficiency'][year] = efficiency
    
    # Sales Expenses
    with st.expander("Sales Expenses", expanded=False):
        st.subheader("Sales Expenses")
        
        col1, col2 = st.columns(2)
        with col1:
            commission_rate = st.number_input(
                "Commission Rate (% of new ARR)",
                min_value=0.0,
                max_value=0.5,
                value=cost_config['sales_expenses'].get('commission_rate', 0.0),
                step=0.01,
                format="%.2f"
            )
            cost_config['sales_expenses']['commission_rate'] = commission_rate
        
        with col2:
            tools_rate = st.number_input(
                "Tools & Enablement (% of ARR)",
                min_value=0.0,
                max_value=0.1,
                value=cost_config['sales_expenses'].get('tools_and_enablement', 0.0),
                step=0.005,
                format="%.3f"
            )
            cost_config['sales_expenses']['tools_and_enablement'] = tools_rate
    
    # R&D Expenses
    with st.expander("R&D Expenses", expanded=False):
        st.subheader("R&D Expenses (% of ARR)")
        
        # Display each R&D category
        rd_categories = [k for k in cost_config['r_and_d_expenses'].keys() if k != "_comment"]
        
        for category in rd_categories:
            rd_value = st.number_input(
                f"{category.replace('_', ' ').title()} (% of ARR)",
                min_value=0.0,
                max_value=0.5,
                value=cost_config['r_and_d_expenses'].get(category, 0.0),
                step=0.01,
                format="%.2f"
            )
            cost_config['r_and_d_expenses'][category] = rd_value
    
    # G&A Expenses
    with st.expander("G&A Expenses", expanded=False):
        st.subheader("General & Administrative Expenses")
        
        # Display each G&A category
        ga_categories = [k for k in cost_config['g_and_a_expenses'].keys() if k != "_comment"]
        
        for category in ga_categories:
            ga_value = st.number_input(
                f"{category.replace('_', ' ').title()} ($/month)",
                min_value=0,
                max_value=100000,
                value=cost_config['g_and_a_expenses'].get(category, 0),
                step=100
            )
            cost_config['g_and_a_expenses'][category] = ga_value
    
    # Save button for cost config
    if st.button("Save Cost Configuration"):
        # Update session state
        st.session_state.cost_config = cost_config
        
        # Save to file
        save_configs(st.session_state.revenue_config, cost_config)

# Tab 4: Reports
with tab4:
    st.header("Reports")
    
    if st.session_state.growth_model is not None:
        report_type = st.selectbox(
            "Select Report Type",
            ["Growth", "Cost", "Financial", "Unit Economics"]
        )
        
        if report_type == "Growth":
            # Growth model reports
            st.subheader("Growth Model Reports")
            
            # Display growth curves
            fig = st.session_state.growth_model.plot_growth_curves()
            st.pyplot(fig)
            
            # Display annual metrics
            fig = st.session_state.growth_model.plot_annual_metrics()
            st.pyplot(fig)
            
            # Display customer segment shares
            fig = st.session_state.growth_model.plot_customer_segment_shares()
            st.pyplot(fig)
            
            # Display summary metrics
            st.write("### Growth Summary Metrics")
            summary = st.session_state.growth_model.display_summary_metrics()
            st.dataframe(summary)
        
        elif report_type == "Cost":
            # Cost model reports
            st.subheader("Cost Model Reports")
            
            # Display expense breakdown
            fig = st.session_state.cost_model.plot_expense_breakdown()
            st.pyplot(fig)
            
            # Display headcount growth
            fig = st.session_state.cost_model.plot_headcount_growth()
            st.pyplot(fig)
            
            # Display summary metrics
            st.write("### Cost Summary Metrics")
            summary = st.session_state.cost_model.display_summary_metrics()
            st.dataframe(summary)
        
        elif report_type == "Financial":
            # Financial model reports
            st.subheader("Financial Model Reports")
            
            # Display financial summary
            fig = st.session_state.financial_model.plot_financial_summary()
            st.pyplot(fig)
            
            # Display break-even analysis
            fig = st.session_state.financial_model.plot_break_even_analysis()
            st.pyplot(fig)
            
            # Display runway and capital
            fig = st.session_state.financial_model.plot_runway_and_capital()
            st.pyplot(fig)
            
            # Display key metrics
            st.write("### Key Financial Metrics")
            key_metrics = st.session_state.financial_model.get_key_metrics_table()
            st.dataframe(key_metrics)
        
        elif report_type == "Unit Economics":
            # Unit economics reports
            st.subheader("Unit Economics Reports")
            
            # Calculate unit economics
            unit_economics = st.session_state.cost_model.calculate_unit_economics(st.session_state.growth_model)
            
            # Display unit economics plot
            fig = st.session_state.cost_model.plot_unit_economics(unit_economics)
            st.pyplot(fig)
            
            # Display unit economics table
            st.write("### Unit Economics Metrics")
            unit_economics_table = st.session_state.cost_model.display_unit_economics_table(unit_economics)
            st.dataframe(unit_economics_table)
    else:
        st.info("Run the model to view reports")

# Tab 5: Import/Export
with tab5:
    st.header("Configuration Import/Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Configurations")
        
        # Export revenue config
        if st.button("Export Revenue Config"):
            # Convert the config to JSON string
            revenue_json = json.dumps(st.session_state.revenue_config, indent=2)
            
            # Create a download link
            st.download_button(
                label="Download Revenue Config JSON",
                data=revenue_json,
                file_name="revenue_config.json",
                mime="application/json"
            )
        
        # Export cost config
        if st.button("Export Cost Config"):
            # Convert the config to JSON string
            cost_json = json.dumps(st.session_state.cost_config, indent=2)
            
            # Create a download link
            st.download_button(
                label="Download Cost Config JSON",
                data=cost_json,
                file_name="cost_config.json",
                mime="application/json"
            )
    
    with col2:
        st.subheader("Import Configurations")
        
        # Import revenue config
        uploaded_revenue = st.file_uploader("Upload Revenue Config", type="json")
        if uploaded_revenue is not None:
            try:
                # Load JSON from the uploaded file
                revenue_config = json.load(uploaded_revenue)
                
                # Validate essential fields
                required_fields = ['start_date', 'projection_months', 'segments', 'initial_arr', 'initial_customers']
                if all(field in revenue_config for field in required_fields):
                    st.session_state.revenue_config = revenue_config
                    st.success("Revenue configuration loaded successfully!")
                    
                    # Option to save to file
                    if st.button("Save Imported Revenue Config to File"):
                        save_configs(revenue_config, st.session_state.cost_config)
                else:
                    st.error("Invalid revenue configuration file. Missing required fields.")
            except Exception as e:
                st.error(f"Error loading revenue configuration: {str(e)}")
        
        # Import cost config
        uploaded_cost = st.file_uploader("Upload Cost Config", type="json")
        if uploaded_cost is not None:
            try:
                # Load JSON from the uploaded file
                cost_config = json.load(uploaded_cost)
                
                # Validate essential fields
                required_fields = ['start_date', 'projection_months', 'cogs', 'headcount']
                if all(field in cost_config for field in required_fields):
                    st.session_state.cost_config = cost_config
                    st.success("Cost configuration loaded successfully!")
                    
                    # Option to save to file
                    if st.button("Save Imported Cost Config to File"):
                        save_configs(st.session_state.revenue_config, cost_config)
                else:
                    st.error("Invalid cost configuration file. Missing required fields.")
            except Exception as e:
                st.error(f"Error loading cost configuration: {str(e)}")

# App footer
st.markdown("---")
st.caption("2025 Financial Model Dashboard - Run with `python streamlit_app.py`")