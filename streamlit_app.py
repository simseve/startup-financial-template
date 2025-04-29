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
    page_title="2025 AI SaaS Financial Model",
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
    
    # Fix for the s_curve data structure - convert string keys to integers
    # This matches the conversion done in app.py
    if 's_curve' in st.session_state.revenue_config:
        for segment in st.session_state.revenue_config['segments']:
            if segment in st.session_state.revenue_config['s_curve']:
                # Convert string year keys to integers
                st.session_state.revenue_config['s_curve'][segment] = {
                    int(year): params for year, params in st.session_state.revenue_config['s_curve'][segment].items()
                    if year != "_comment"
                }
    
    # Convert string month keys to integers in seasonality
    if 'seasonality' in st.session_state.revenue_config:
        st.session_state.revenue_config['seasonality'] = {
            int(month): factor for month, factor in st.session_state.revenue_config['seasonality'].items()
            if month != "_comment"
        }

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
            value=48,
            help="Target month to achieve breakeven (1-72)"
        )

        revenue_target = st.number_input(
            "Annual Revenue Target ($)",
            min_value=1000000,
            max_value=500000000,
            value=30000000,
            step=1000000,
            format="%d",
            help="Annual revenue target in USD"
        )

        revenue_target_year = st.number_input(
            "Revenue Target Year",
            min_value=1,
            max_value=6,
            value=5,
            help="Year to achieve annual revenue target (1-6)"
        )

        max_iterations = st.number_input(
            "Max Optimization Iterations",
            min_value=1,
            max_value=50,
            value=15,
            help="Maximum number of iterations for optimization"
        )

    # Initial investment input (common to all strategies)
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000000,
        max_value=50000000,
        value=20000000,
        step=1000000,
        format="%d",
        help="Initial capital investment amount"
    )

    # Run model button
    if st.button("Run Model"):
        with st.spinner("Running model..."):
            if strategy == "baseline":
                st.session_state.growth_model, st.session_state.cost_model, st.session_state.financial_model = run_baseline_model(
                    initial_investment)
            elif strategy == "profile":
                st.session_state.growth_model, st.session_state.cost_model, st.session_state.financial_model = run_growth_profile_model(
                    profile, initial_investment)
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Results Overview",
    "Growth Configuration",
    "Cost Configuration",
    "Reports",
    "Import/Export",
    "S-Curve Visualization"
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
        unit_economics = st.session_state.cost_model.calculate_unit_economics(
            st.session_state.growth_model)
        unit_economics_table = st.session_state.cost_model.display_unit_economics_table(
            unit_economics)
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
                        value=revenue_config['initial_arr'].get(
                            segment, 10000),
                        step=1000
                    )
                    revenue_config['initial_arr'][segment] = initial_arr

                with col2:
                    initial_customers = st.number_input(
                        f"Initial Customers - {segment}",
                        min_value=0,
                        max_value=1000,
                        value=revenue_config['initial_customers'].get(
                            segment, 0)
                    )
                    revenue_config['initial_customers'][segment] = initial_customers

                # Contract length and churn rates
                col1, col2 = st.columns(2)
                with col1:
                    contract_length = st.number_input(
                        f"Contract Length (years) - {segment}",
                        min_value=0.25,
                        max_value=5.0,
                        value=revenue_config['contract_length'].get(
                            segment, 1.0),
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
                    value=revenue_config['annual_price_increases'].get(
                        segment, 0.0) * 100,
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
                            value=revenue_config['s_curve'][segment][year].get(
                                'midpoint', 6),
                            key=f"mp_{segment}_{year}"
                        )

                    with col2:
                        steepness = st.number_input(
                            f"Steepness - Y{year}",
                            min_value=0.05,
                            max_value=2.0,
                            value=revenue_config['s_curve'][segment][year].get(
                                'steepness', 0.5),
                            step=0.05,
                            format="%.2f",
                            key=f"st_{segment}_{year}"
                        )

                    with col3:
                        max_monthly = st.number_input(
                            f"Max Monthly Customers - Y{year}",
                            min_value=0,
                            max_value=100,
                            value=revenue_config['s_curve'][segment][year].get(
                                'max_monthly', 2),
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
                        value=revenue_config['seasonality'].get(
                            str(month_idx), revenue_config['seasonality'].get(month_idx, 1.0)),
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
        cogs_categories = [
            k for k in cost_config['cogs'].keys() if k != "_comment"]

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

        departments = [k for k in cost_config['headcount'].keys(
        ) if k != "_comment" and isinstance(cost_config['headcount'][k], dict)]

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
                        value=cost_config['headcount'][dept].get(
                            'starting_count', 0)
                    )
                    cost_config['headcount'][dept]['starting_count'] = starting_count

                with col2:
                    avg_salary = st.number_input(
                        f"Average Annual Salary ($) - {dept}",
                        min_value=30000,
                        max_value=300000,
                        value=cost_config['headcount'][dept].get(
                            'avg_salary', 100000),
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
        marketing_categories = [
            k for k in cost_config['marketing_expenses'].keys() if k != "_comment"]

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
                value=cost_config['sales_expenses'].get(
                    'commission_rate', 0.0),
                step=0.01,
                format="%.2f"
            )
            cost_config['sales_expenses']['commission_rate'] = commission_rate

        with col2:
            tools_rate = st.number_input(
                "Tools & Enablement (% of ARR)",
                min_value=0.0,
                max_value=0.1,
                value=cost_config['sales_expenses'].get(
                    'tools_and_enablement', 0.0),
                step=0.005,
                format="%.3f"
            )
            cost_config['sales_expenses']['tools_and_enablement'] = tools_rate

    # R&D Expenses
    with st.expander("R&D Expenses", expanded=False):
        st.subheader("R&D Expenses (% of ARR)")

        # Display each R&D category
        rd_categories = [
            k for k in cost_config['r_and_d_expenses'].keys() if k != "_comment"]

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
        ga_categories = [
            k for k in cost_config['g_and_a_expenses'].keys() if k != "_comment"]

        for category in ga_categories:
            ga_value = st.number_input(
                f"{category.replace('_', ' ').title()} ($/month)",
                min_value=0,
                max_value=100000,
                value=cost_config['g_and_a_expenses'].get(category, 0),
                step=100
            )
            cost_config['g_and_a_expenses'][category] = ga_value

    # One-Time Expenses
    with st.expander("One-Time Expenses", expanded=False):
        st.subheader("One-Time Expenses")
        st.write("Non-recurring expenses that occur only once")

        # Initialize one_time_expenses if it doesn't exist or create items array if needed
        if 'one_time_expenses' not in cost_config:
            cost_config['one_time_expenses'] = {}
        if 'items' not in cost_config['one_time_expenses']:
            cost_config['one_time_expenses']['items'] = []

        # Display existing one-time expenses
        if len(cost_config['one_time_expenses']['items']) > 0:
            st.write("### Current One-Time Expenses")

            # Create a DataFrame for better display
            expense_data = []
            for idx, expense in enumerate(cost_config['one_time_expenses']['items']):
                expense_data.append({
                    'Month': expense[0],
                    'Name': expense[1],
                    'Amount ($)': f"${expense[2]:,.2f}",
                    'Description': expense[3]
                })

            expense_df = pd.DataFrame(expense_data)
            st.dataframe(expense_df, hide_index=True)

            # Option to delete expenses
            if st.button("Delete All One-Time Expenses"):
                cost_config['one_time_expenses']['items'] = []
                st.success("All one-time expenses deleted!")

        # Add new one-time expense
        st.write("### Add New One-Time Expense")

        col1, col2 = st.columns(2)
        with col1:
            new_expense_month = st.number_input(
                "Month Number (1-72)",
                min_value=1,
                max_value=72,
                value=1
            )

            new_expense_name = st.text_input(
                "Expense Name",
                value=""
            )

        with col2:
            new_expense_amount = st.number_input(
                "Amount ($)",
                min_value=0,
                max_value=10000000,
                value=0
            )

            new_expense_description = st.text_input(
                "Description",
                value=""
            )

        # Add button
        if st.button("Add One-Time Expense"):
            if new_expense_name.strip() and new_expense_amount > 0:
                # Add the new expense to the list
                cost_config['one_time_expenses']['items'].append([
                    new_expense_month,
                    new_expense_name,
                    new_expense_amount,
                    new_expense_description
                ])
                st.success(
                    f"Added one-time expense: {new_expense_name} (${new_expense_amount:,.2f}) in month {new_expense_month}")
            else:
                st.error("Please provide a name and an amount greater than zero")

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
            unit_economics = st.session_state.cost_model.calculate_unit_economics(
                st.session_state.growth_model)

            # Display unit economics plot
            fig = st.session_state.cost_model.plot_unit_economics(
                unit_economics)
            st.pyplot(fig)

            # Display unit economics table
            st.write("### Unit Economics Metrics")
            unit_economics_table = st.session_state.cost_model.display_unit_economics_table(
                unit_economics)
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
            revenue_json = json.dumps(
                st.session_state.revenue_config, indent=2)

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
        uploaded_revenue = st.file_uploader(
            "Upload Revenue Config", type="json")
        if uploaded_revenue is not None:
            try:
                # Load JSON from the uploaded file
                revenue_config = json.load(uploaded_revenue)

                # Validate essential fields
                required_fields = ['start_date', 'projection_months',
                                   'segments', 'initial_arr', 'initial_customers']
                if all(field in revenue_config for field in required_fields):
                    st.session_state.revenue_config = revenue_config
                    st.success("Revenue configuration loaded successfully!")

                    # Option to save to file
                    if st.button("Save Imported Revenue Config to File"):
                        save_configs(revenue_config,
                                     st.session_state.cost_config)
                else:
                    st.error(
                        "Invalid revenue configuration file. Missing required fields.")
            except Exception as e:
                st.error(f"Error loading revenue configuration: {str(e)}")

        # Import cost config
        uploaded_cost = st.file_uploader("Upload Cost Config", type="json")
        if uploaded_cost is not None:
            try:
                # Load JSON from the uploaded file
                cost_config = json.load(uploaded_cost)

                # Validate essential fields
                required_fields = ['start_date',
                                   'projection_months', 'cogs', 'headcount']
                if all(field in cost_config for field in required_fields):
                    st.session_state.cost_config = cost_config
                    st.success("Cost configuration loaded successfully!")

                    # Option to save to file
                    if st.button("Save Imported Cost Config to File"):
                        save_configs(
                            st.session_state.revenue_config, cost_config)
                else:
                    st.error(
                        "Invalid cost configuration file. Missing required fields.")
            except Exception as e:
                st.error(f"Error loading cost configuration: {str(e)}")

# Tab 6: S-Curve Visualization
with tab6:
    st.header("S-Curve Parameter Visualization")
    
    # Create a function to visualize S-curves
    def plot_s_curve(segment, year, midpoint, steepness, max_monthly):
        """Plot the S-curve for a specific segment and year"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create the month position array (0-11)
        month_positions = np.arange(12)
        
        # Calculate the S-curve values
        midpoint_idx = midpoint - 1  # Convert to 0-indexed
        s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
        
        # Apply seasonality if available
        if 'seasonality' in st.session_state.revenue_config:
            seasonality_factors = [st.session_state.revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
            seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
            
            # Plot both the raw S-curve and the seasonality-adjusted curve
            ax.plot(month_positions, s_curve_values, label='Raw S-Curve', linestyle='--')
            ax.plot(month_positions, seasonal_values, label='With Seasonality', linewidth=2)
            ax.bar(month_positions, seasonal_values, alpha=0.3, label='Monthly New Customers')
        else:
            # Just plot the raw S-curve
            ax.plot(month_positions, s_curve_values, label='S-Curve', linewidth=2)
            ax.bar(month_positions, s_curve_values, alpha=0.3, label='Monthly New Customers')
        
        # Formatting
        ax.set_title(f'{segment} Year {year} S-Curve (Midpoint: {midpoint}, Steepness: {steepness}, Max: {max_monthly})')
        ax.set_xlabel('Month of Year')
        ax.set_ylabel('New Customers')
        ax.set_xticks(month_positions)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    # Create a function to plot all years for a segment
    def plot_segment_years(segment, s_curve_params):
        """Plot all years for a specific segment"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        month_positions = np.arange(12)
        years = sorted([y for y in s_curve_params.keys()])
        
        for year in years:
            # Get the parameters for this year
            params = s_curve_params[year]
            midpoint = params.get('midpoint', 6)
            steepness = params.get('steepness', 0.5)
            max_monthly = params.get('max_monthly', 2)
            
            # Calculate the S-curve values
            midpoint_idx = midpoint - 1  # Convert to 0-indexed
            s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
            
            # Apply seasonality
            seasonality_factors = [st.session_state.revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
            seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
            
            # Plot the curve for this year
            ax.plot(month_positions, seasonal_values, label=f'Year {year}', marker='o')
        
        # Formatting
        ax.set_title(f'{segment} S-Curves Across All Years')
        ax.set_xlabel('Month of Year')
        ax.set_ylabel('New Customers')
        ax.set_xticks(month_positions)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    # Create a function to plot all segments for a specific year
    def plot_year_segments(year, segments, s_curve_data):
        """Plot all segments for a specific year"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        month_positions = np.arange(12)
        
        for segment in segments:
            # Get the parameters for this segment
            if segment in s_curve_data and year in s_curve_data[segment]:
                params = s_curve_data[segment][year]
                midpoint = params.get('midpoint', 6)
                steepness = params.get('steepness', 0.5)
                max_monthly = params.get('max_monthly', 2)
                
                # Calculate the S-curve values
                midpoint_idx = midpoint - 1  # Convert to 0-indexed
                s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
                
                # Apply seasonality
                seasonality_factors = [st.session_state.revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
                seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
                
                # Plot the curve for this segment
                ax.plot(month_positions, seasonal_values, label=segment, marker='o')
        
        # Formatting
        ax.set_title(f'Year {year} S-Curves Across All Segments')
        ax.set_xlabel('Month of Year')
        ax.set_ylabel('New Customers')
        ax.set_xticks(month_positions)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    # Create a combined view with all segments and years
    def plot_all_s_curves(s_curve_data, segments):
        """Plot all S-curves in a grid layout"""
        years = 6
        fig, axs = plt.subplots(len(segments), years, figsize=(18, 10), sharex=True)
        
        month_positions = np.arange(12)
        
        for i, segment in enumerate(segments):
            for j in range(years):
                year = j + 1
                ax = axs[i, j]
                
                # Get the parameters for this segment and year
                if segment in s_curve_data and year in s_curve_data[segment]:
                    params = s_curve_data[segment][year]
                    midpoint = params.get('midpoint', 6)
                    steepness = params.get('steepness', 0.5)
                    max_monthly = params.get('max_monthly', 2)
                    
                    # Calculate the S-curve values
                    midpoint_idx = midpoint - 1  # Convert to 0-indexed
                    s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
                    
                    # Apply seasonality
                    seasonality_factors = [st.session_state.revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
                    seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
                    
                    # Plot the curve
                    ax.plot(month_positions, seasonal_values)
                    ax.fill_between(month_positions, seasonal_values, alpha=0.3)
                    
                    # Add annotations
                    ax.text(0.5, 0.9, f"Mid: {midpoint}", transform=ax.transAxes, ha='center', fontsize=8)
                    ax.text(0.5, 0.8, f"Steep: {steepness}", transform=ax.transAxes, ha='center', fontsize=8)
                    ax.text(0.5, 0.7, f"Max: {max_monthly}", transform=ax.transAxes, ha='center', fontsize=8)
                
                # Set titles only for top row and left column
                if i == 0:
                    ax.set_title(f'Year {year}', fontsize=10)
                if j == 0:
                    ax.set_ylabel(segment, fontsize=10)
                
                # Set xticks only for bottom row
                if i == len(segments) - 1:
                    ax.set_xticks([0, 6, 11])
                    ax.set_xticklabels(['Jan', 'Jul', 'Dec'], fontsize=8)
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    # Display visualization options
    viz_option = st.radio(
        "Select Visualization Type",
        ["Individual S-Curves", "Segment Comparison", "Year Comparison", "All S-Curves Grid"]
    )
    
    revenue_config = st.session_state.revenue_config
    
    # Debug: Print S-curve parameters to see what's actually loaded
    st.write("### Loaded S-Curve Parameters")
    if 's_curve' in revenue_config:
        for segment in revenue_config['segments']:
            if segment in revenue_config['s_curve']:
                st.write(f"**{segment}**")
                for year in range(1, 7):
                    if year in revenue_config['s_curve'][segment]:
                        params = revenue_config['s_curve'][segment][year]
                        st.write(f"Year {year}: Midpoint={params.get('midpoint', 'N/A')}, "
                                f"Steepness={params.get('steepness', 'N/A')}, "
                                f"Max Monthly={params.get('max_monthly', 'N/A')}")
    
    if viz_option == "Individual S-Curves":
        # Select segment and year
        col1, col2 = st.columns(2)
        
        with col1:
            segment = st.selectbox("Select Segment", revenue_config['segments'])
        
        with col2:
            year = st.selectbox("Select Year", list(range(1, 7)))
        
        # Get the S-curve parameters for the selected segment and year
        if segment in revenue_config['s_curve'] and year in revenue_config['s_curve'][segment]:
            params = revenue_config['s_curve'][segment][year]
            midpoint = params.get('midpoint', 6)
            steepness = params.get('steepness', 0.5)
            max_monthly = params.get('max_monthly', 2)
            
            # Display the parameters
            st.write(f"### S-Curve Parameters for {segment} in Year {year}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Midpoint", midpoint)
            col2.metric("Steepness", steepness)
            col3.metric("Max Monthly", max_monthly)
            
            # Add parameter explanations
            st.write("""
            **Parameter Explanation:**
            - **Midpoint:** Month (1-12) where the growth curve reaches its inflection point
            - **Steepness:** Controls how rapidly growth accelerates/decelerates (higher = steeper curve)
            - **Max Monthly:** Maximum number of new customers that can be acquired per month
            """)
            
            # Interactive parameters
            st.write("### Interactive Parameter Adjustment")
            st.write("Adjust the parameters to see how they affect the S-curve shape:")
            
            int_col1, int_col2, int_col3 = st.columns(3)
            with int_col1:
                interactive_midpoint = st.slider("Midpoint", 1, 12, midpoint)
            with int_col2:
                interactive_steepness = st.slider("Steepness", 0.05, 2.0, float(steepness), 0.05)
            with int_col3:
                interactive_max = st.slider("Max Monthly", 1, max(50, max_monthly*2), max_monthly)
            
            # Show both the original and interactive curves
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create the month position array (0-11)
            month_positions = np.arange(12)
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Original S-curve values
            midpoint_idx = midpoint - 1  # Convert to 0-indexed
            s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
            
            # Interactive S-curve values
            interactive_midpoint_idx = interactive_midpoint - 1  # Convert to 0-indexed
            interactive_s_curve_values = [interactive_max / (1 + np.exp(-interactive_steepness * (m - interactive_midpoint_idx))) 
                                         for m in month_positions]
            
            # Apply seasonality to both
            seasonality_factors = [revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
            seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
            interactive_seasonal_values = [s * f for s, f in zip(interactive_s_curve_values, seasonality_factors)]
            
            # Calculate annual totals
            original_annual = sum(seasonal_values)
            interactive_annual = sum(interactive_seasonal_values)
            
            # Plot the original curve
            ax.plot(month_positions, s_curve_values, label=f'Original S-Curve (Annual: {original_annual:.1f})', 
                   linestyle='--', alpha=0.7, color='blue')
            ax.plot(month_positions, seasonal_values, label=f'Original with Seasonality', 
                   linewidth=2, alpha=0.7, color='darkblue')
            
            # Plot the interactive curve
            ax.plot(month_positions, interactive_s_curve_values, label=f'Interactive S-Curve (Annual: {interactive_annual:.1f})', 
                   linestyle='--', alpha=0.7, color='red')
            ax.plot(month_positions, interactive_seasonal_values, label=f'Interactive with Seasonality', 
                   linewidth=2, alpha=0.7, color='darkred')
            
            # Plot the bar chart for the original curve (seasonal)
            ax.bar(month_positions, seasonal_values, alpha=0.1, color='blue', width=0.4, 
                  align='edge', label=None)
            
            # Plot the bar chart for the interactive curve (seasonal)
            ax.bar(month_positions, interactive_seasonal_values, alpha=0.1, color='red', width=-0.4, 
                  align='edge', label=None)
            
            # Formatting
            ax.set_title(f'{segment} Year {year} S-Curve Comparison', fontsize=16)
            ax.set_xlabel('Month of Year', fontsize=14)
            ax.set_ylabel('New Customers', fontsize=14)
            ax.set_xticks(month_positions)
            ax.set_xticklabels(month_labels)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Visualization explanation
            st.write("""
            ### Understanding the S-Curve
            
            The S-curve models how customer acquisition grows over time within a year:
            
            - **Early phase:** Slow initial growth at the beginning of the curve
            - **Middle phase:** Rapid growth around the midpoint (inflection point)
            - **Later phase:** Leveling off as the market segment becomes saturated
            
            **Seasonality factors** (shown in the curve with solid lines) adjust the base S-curve to account for seasonal variations in customer acquisition.
            
            **Total annual customers** is the sum of all monthly values with seasonality applied.
            """)
            
            # Also show the original S-curve
            st.write("### Original S-Curve Visualization")
            fig = plot_s_curve(segment, year, midpoint, steepness, max_monthly)
            st.pyplot(fig)
        else:
            st.warning(f"No S-curve parameters found for {segment} in Year {year}")
    
    elif viz_option == "Segment Comparison":
        # Select year
        year = st.selectbox("Select Year", list(range(1, 7)))
        
        st.write(f"### Segment Comparison for Year {year}")
        st.write("Compare how different customer segments grow within the same year")
        
        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        month_positions = np.arange(12)
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        segments = revenue_config['segments']
        
        # Color map for segments
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
        
        # Annual totals and parameters
        segment_totals = []
        segment_params = []
        
        # Plot each segment
        for i, segment in enumerate(segments):
            if segment in revenue_config['s_curve'] and year in revenue_config['s_curve'][segment]:
                # Get parameters
                params = revenue_config['s_curve'][segment][year]
                midpoint = params.get('midpoint', 6)
                steepness = params.get('steepness', 0.5)
                max_monthly = params.get('max_monthly', 2)
                
                # Save parameters for display
                segment_params.append((segment, midpoint, steepness, max_monthly))
                
                # Calculate values
                midpoint_idx = midpoint - 1
                s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
                
                # Apply seasonality
                seasonality_factors = [revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
                seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
                
                # Calculate annual total
                annual_total = sum(seasonal_values)
                segment_totals.append(annual_total)
                
                # Plot the curve
                ax1.plot(month_positions, seasonal_values, 
                       label=f'{segment} (Total: {annual_total:.1f})', 
                       color=colors[i], linewidth=2.5, marker='o')
                
                # Add light fill under the curve
                ax1.fill_between(month_positions, seasonal_values, alpha=0.15, color=colors[i])
        
        # Plot cumulative total line
        if segment_totals:
            cumulative_values = []
            for m in month_positions:
                monthly_sum = 0
                for i, segment in enumerate(segments):
                    if segment in revenue_config['s_curve'] and year in revenue_config['s_curve'][segment]:
                        params = revenue_config['s_curve'][segment][year]
                        midpoint = params.get('midpoint', 6)
                        steepness = params.get('steepness', 0.5)
                        max_monthly = params.get('max_monthly', 2)
                        
                        # Calculate this segment's contribution to this month
                        midpoint_idx = midpoint - 1
                        s_value = max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx)))
                        monthly_sum += s_value * revenue_config['seasonality'].get(m+1, 1.0)
                
                cumulative_values.append(monthly_sum)
            
            # Plot the combined line
            ax1.plot(month_positions, cumulative_values, 
                   label=f'All Segments (Total: {sum(cumulative_values):.1f})', 
                   color='black', linewidth=3, linestyle='-', marker='s')
        
        # Formatting for top plot
        ax1.set_title(f'Monthly New Customer Acquisition by Segment - Year {year}', fontsize=16)
        ax1.set_xlabel('Month of Year', fontsize=14)
        ax1.set_ylabel('New Customers per Month', fontsize=14)
        ax1.set_xticks(month_positions)
        ax1.set_xticklabels(month_labels)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Create the stacked bar chart for bottom plot
        segment_annual_data = {}
        for i, segment in enumerate(segments):
            if segment in revenue_config['s_curve'] and year in revenue_config['s_curve'][segment]:
                by_month_values = []
                params = revenue_config['s_curve'][segment][year]
                midpoint = params.get('midpoint', 6)
                steepness = params.get('steepness', 0.5)
                max_monthly = params.get('max_monthly', 2)
                
                for m in range(12):
                    midpoint_idx = midpoint - 1
                    s_value = max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx)))
                    seasonal_value = s_value * revenue_config['seasonality'].get(m+1, 1.0)
                    by_month_values.append(seasonal_value)
                
                segment_annual_data[segment] = by_month_values
        
        # Create bottom stacked bar chart
        bottom = np.zeros(12)
        for i, segment in enumerate(segments):
            if segment in segment_annual_data:
                ax2.bar(month_positions, segment_annual_data[segment], bottom=bottom, 
                       label=segment, color=colors[i], alpha=0.7)
                bottom += np.array(segment_annual_data[segment])
        
        # Formatting for bottom plot
        ax2.set_title(f'Monthly Customer Acquisition Composition - Year {year}', fontsize=14)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('New Customers', fontsize=12)
        ax2.set_xticks(month_positions)
        ax2.set_xticklabels(month_labels)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display the parameters table
        st.write(f"### S-Curve Parameters for Year {year}")
        
        param_data = {
            'Segment': [p[0] for p in segment_params],
            'Midpoint': [p[1] for p in segment_params],
            'Steepness': [p[2] for p in segment_params],
            'Max Monthly': [p[3] for p in segment_params],
            'Annual Total': segment_totals
        }
        
        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, hide_index=True)
        
        # Calculate the relative contribution of each segment
        if sum(segment_totals) > 0:
            st.write("### Segment Contribution Analysis")
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(10, 7))
            wedges, texts, autotexts = ax.pie(
                segment_totals, 
                labels=[p[0] for p in segment_params],
                autopct='%1.1f%%',
                explode=[0.05] * len(segment_totals),
                colors=colors[:len(segment_totals)],
                shadow=True,
                startangle=90
            )
            
            # Style the text and percentages
            for text in texts:
                text.set_fontsize(12)
            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_fontweight('bold')
                
            ax.set_title(f'Segment Contribution to New Customers - Year {year}', fontsize=16)
            st.pyplot(fig)
        
        # Original visualization
        st.write("### Original Segment Comparison")
        fig = plot_year_segments(year, revenue_config['segments'], revenue_config['s_curve'])
        st.pyplot(fig)
    
    elif viz_option == "Year Comparison":
        # Select segment
        segment = st.selectbox("Select Segment", revenue_config['segments'])
        
        # Plot all years for the selected segment
        if segment in revenue_config['s_curve']:
            st.write("### Multi-Year Growth Evolution")
            st.write(f"Visualizing how the growth curve for **{segment}** evolves across years")
            
            # Create an enhanced visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
            
            month_positions = np.arange(12)
            month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            years = sorted([y for y in revenue_config['s_curve'][segment].keys()])
            
            # Color map for years
            colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
            
            # Annual totals and parameters
            annual_totals = []
            year_params = []
            
            # Plot each year
            for i, year in enumerate(years):
                # Get the parameters
                params = revenue_config['s_curve'][segment][year]
                midpoint = params.get('midpoint', 6)
                steepness = params.get('steepness', 0.5)
                max_monthly = params.get('max_monthly', 2)
                
                # Save parameters for display
                year_params.append((year, midpoint, steepness, max_monthly))
                
                # Calculate values
                midpoint_idx = midpoint - 1
                s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
                
                # Apply seasonality
                seasonality_factors = [revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
                seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
                
                # Calculate annual total
                annual_total = sum(seasonal_values)
                annual_totals.append(annual_total)
                
                # Plot the curve
                ax1.plot(month_positions, seasonal_values, 
                       label=f'Year {year} (Total: {annual_total:.1f})', 
                       color=colors[i], linewidth=2.5, marker='o')
                
                # Add light fill under the curve
                ax1.fill_between(month_positions, seasonal_values, alpha=0.1, color=colors[i])
            
            # Formatting for top plot
            ax1.set_title(f'Monthly New Customer Acquisition for {segment} - Year-by-Year Comparison', fontsize=16)
            ax1.set_xlabel('Month of Year', fontsize=14)
            ax1.set_ylabel('New Customers per Month', fontsize=14)
            ax1.set_xticks(month_positions)
            ax1.set_xticklabels(month_labels)
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Plot the annual totals in the bottom subplot
            bars = ax2.bar(years, annual_totals, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, annual_totals):
                ax2.text(bar.get_x() + bar.get_width()/2, value + 0.5, 
                        f'{value:.1f}', ha='center', va='bottom', fontsize=12)
            
            # Formatting for bottom plot
            ax2.set_title(f'Total Annual New Customers - {segment}', fontsize=14)
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('Total New Customers', fontsize=12)
            ax2.set_xticks(years)
            ax2.set_xticklabels([f'Year {y}' for y in years])
            ax2.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display the parameters table
            st.write("### S-Curve Parameters by Year")
            
            param_data = {
                'Year': [f'Year {p[0]}' for p in year_params],
                'Midpoint': [p[1] for p in year_params],
                'Steepness': [p[2] for p in year_params],
                'Max Monthly': [p[3] for p in year_params],
                'Annual Total': annual_totals
            }
            
            param_df = pd.DataFrame(param_data)
            st.dataframe(param_df, hide_index=True)
            
            # Show original visualization as well
            st.write("### Original Year Comparison")
            fig = plot_segment_years(segment, revenue_config['s_curve'][segment])
            st.pyplot(fig)
        else:
            st.warning(f"No S-curve parameters found for {segment}")
    
    elif viz_option == "All S-Curves Grid":
        st.write("### Complete S-Curve Overview")
        st.write("Comprehensive visualization of all segments and years in a single view")
        
        # Enhanced S-curve grid
        segments = revenue_config['segments']
        years = list(range(1, 7))
        
        # Calculate the total numbers for the heatmap
        heatmap_data = np.zeros((len(segments), len(years)))
        
        for i, segment in enumerate(segments):
            if segment in revenue_config['s_curve']:
                for j, year in enumerate(years):
                    if year in revenue_config['s_curve'][segment]:
                        params = revenue_config['s_curve'][segment][year]
                        midpoint = params.get('midpoint', 6)
                        steepness = params.get('steepness', 0.5)
                        max_monthly = params.get('max_monthly', 2)
                        
                        # Calculate monthly values
                        month_positions = np.arange(12)
                        midpoint_idx = midpoint - 1
                        s_curve_values = [max_monthly / (1 + np.exp(-steepness * (m - midpoint_idx))) for m in month_positions]
                        
                        # Apply seasonality
                        seasonality_factors = [revenue_config['seasonality'].get(m+1, 1.0) for m in month_positions]
                        seasonal_values = [s * f for s, f in zip(s_curve_values, seasonality_factors)]
                        
                        # Calculate annual total
                        annual_total = sum(seasonal_values)
                        heatmap_data[i, j] = annual_total
        
        # Create a heatmap visualization of annual totals
        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(heatmap_data, cmap='YlOrRd')
        
        # Add labels
        ax.set_xticks(np.arange(len(years)))
        ax.set_yticks(np.arange(len(segments)))
        ax.set_xticklabels([f'Year {y}' for y in years])
        ax.set_yticklabels(segments)
        
        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        
        # Add value annotations
        for i in range(len(segments)):
            for j in range(len(years)):
                text = ax.text(j, i, f"{heatmap_data[i, j]:.1f}",
                               ha="center", va="center", color="black" if heatmap_data[i, j] < np.max(heatmap_data)*0.7 else "white")
        
        # Add title and colorbar
        ax.set_title("Annual New Customer Acquisition by Segment and Year")
        plt.colorbar(im, ax=ax, label="New Customers per Year")
        
        fig.tight_layout()
        st.pyplot(fig)
        
        # Cumulative totals over 6 years
        cumulative_by_segment = np.sum(heatmap_data, axis=1)
        cumulative_by_year = np.sum(heatmap_data, axis=0)
        
        # Create summary table
        summary_data = {
            'Segment': segments,
            'Total New Customers (6 Years)': cumulative_by_segment,
            'Percentage of Total': [val/np.sum(cumulative_by_segment)*100 for val in cumulative_by_segment]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        st.write("### Summary by Segment (6-Year Total)")
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        # Create bar chart of segment totals
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(segments, cumulative_by_segment, color=plt.cm.tab10(np.linspace(0, 1, len(segments))))
        
        # Add data labels on top of bars
        for bar, value in zip(bars, cumulative_by_segment):
            ax.text(bar.get_x() + bar.get_width()/2, value + 1, 
                   f'{value:.1f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_title('Total New Customers by Segment (6-Year Period)', fontsize=16)
        ax.set_ylabel('Total New Customers', fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        
        st.pyplot(fig)
        
        # Year-by-year growth chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years, cumulative_by_year, marker='o', linewidth=2, markersize=10, color='darkblue')
        
        # Add data labels above points
        for x, y in zip(years, cumulative_by_year):
            ax.text(x, y + 2, f'{y:.1f}', ha='center', va='bottom', fontsize=12)
        
        ax.set_title('Total New Customers by Year (All Segments)', fontsize=16)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('New Customers', fontsize=14)
        ax.set_xticks(years)
        ax.set_xticklabels([f'Year {y}' for y in years])
        ax.grid(True, alpha=0.3)
        
        # Calculate year-over-year growth rates
        growth_rates = []
        for i in range(1, len(cumulative_by_year)):
            if cumulative_by_year[i-1] > 0:
                growth_rate = (cumulative_by_year[i] / cumulative_by_year[i-1] - 1) * 100
                growth_rates.append(growth_rate)
            else:
                growth_rates.append(0)
        
        # Add growth rate annotations
        for i, rate in enumerate(growth_rates):
            year = years[i+1]
            y_pos = cumulative_by_year[i+1]
            ax.text(year, y_pos - 5, f"{rate:.1f}%", ha='center', va='top', 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        st.pyplot(fig)
        
        # Original grid visualization
        st.write("### Original S-Curve Grid")
        fig = plot_all_s_curves(revenue_config['s_curve'], revenue_config['segments'])
        st.pyplot(fig)
    
    # Add seasonality visualization
    st.write("### Monthly Seasonality Factors")
    seasonality = revenue_config['seasonality']
    
    # Plot seasonality
    fig, ax = plt.subplots(figsize=(12, 6))
    months = list(range(1, 13))
    seasonality_values = [seasonality.get(m, 1.0) for m in months]
    
    ax.bar(months, seasonality_values)
    ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline (1.0)')
    
    # Formatting
    ax.set_title('Monthly Seasonality Factors')
    ax.set_xlabel('Month')
    ax.set_ylabel('Seasonality Factor')
    ax.set_xticks(months)
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    for i, v in enumerate(seasonality_values):
        ax.text(i+1, v + 0.05, f"{v:.1f}", ha='center')
    
    st.pyplot(fig)

# App footer
st.markdown("---")
st.caption("2025 AI SaaS Financial Model Dashboard - Run with `python streamlit_app.py`")
