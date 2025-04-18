import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app import run_integrated_financial_model
import seaborn as sns
from PIL import Image
import os

st.set_page_config(
    layout="wide", page_title="AI SaaS Financial Model", page_icon="💰")


def main():
    st.title("AI SaaS Financial Model Dashboard")
    st.subheader("Interactive Financial Projections for AI SaaS Companies")

    # Set up sidebar
    st.sidebar.header("Model Parameters")
    initial_investment = st.sidebar.slider(
        "Initial Investment ($M)",
        min_value=5.0,
        max_value=50.0,
        value=20.0,
        step=1.0
    ) * 1000000  # Convert to actual dollars

    # COGS Parameters
    st.sidebar.subheader("COGS Parameters (% of ARR)")
    cloud_hosting = st.sidebar.slider("Cloud Hosting", 0.0, 0.30, 0.18, 0.01,
                                      format="%0.0f%%", help="Percentage of ARR spent on cloud hosting")
    customer_support = st.sidebar.slider(
        "Customer Support", 0.0, 0.20, 0.08, 0.01, format="%0.0f%%")
    third_party_apis = st.sidebar.slider(
        "Third-Party APIs", 0.0, 0.15, 0.06, 0.01, format="%0.0f%%")
    professional_services = st.sidebar.slider(
        "Professional Services", 0.0, 0.10, 0.03, 0.01, format="%0.0f%%")

    # Compensation Parameters
    st.sidebar.subheader("Compensation Parameters")
    benefits_multiplier = st.sidebar.slider(
        "Benefits Multiplier", 1.0, 1.5, 1.28, 0.01, help="1.28 means benefits are 28% of base salary")
    bonus_rate = st.sidebar.slider("Bonus Rate", 0.0, 0.25, 0.12, 0.01,
                                   format="%0.0f%%", help="Annual bonus as percentage of salary")
    equity_compensation = st.sidebar.slider(
        "Equity Compensation", 0.0, 0.30, 0.18, 0.01, format="%0.0f%%", help="Equity comp as percentage of salary")

    # Growth Curve Parameters for Enterprise
    st.sidebar.subheader("Enterprise Growth Parameters")
    st.sidebar.markdown("S-curve steepness by year:")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        enterprise_y1 = st.number_input(
            "Year 1", min_value=0.5, max_value=1.0, value=0.6, step=0.1)
    with col2:
        enterprise_y2 = st.number_input(
            "Year 2", min_value=0.5, max_value=1.0, value=0.7, step=0.1)
    with col3:
        enterprise_y3 = st.number_input(
            "Year 3", min_value=0.5, max_value=1.0, value=0.8, step=0.1)

    # Run button
    if st.sidebar.button("Run Financial Model"):
        # Run the model with parameters
        with st.spinner("Running financial model..."):
            # Create the model
            financial_model, revenue_model, cost_model = run_integrated_financial_model(
                initial_investment)

            # Update COGS parameters
            cost_model.config['cogs']['cloud_hosting'] = cloud_hosting
            cost_model.config['cogs']['customer_support'] = customer_support
            cost_model.config['cogs']['third_party_apis'] = third_party_apis
            cost_model.config['cogs']['professional_services'] = professional_services

            # Update compensation parameters
            cost_model.config['salary']['benefits_multiplier'] = benefits_multiplier
            cost_model.config['salary']['bonus_rate'] = bonus_rate
            cost_model.config['salary']['equity_compensation'] = equity_compensation

            # Update growth curve for Enterprise segment
            steepness_config = {1: enterprise_y1,
                                2: enterprise_y2, 3: enterprise_y3}
            tuned_growth_model = revenue_model.tune_s_curve_steepness(
                'Enterprise', steepness_config)
            tuned_growth_model.run_model()

            # Re-run models with updated parameters
            financial_model.revenue_model = tuned_growth_model
            cost_model.run_model(tuned_growth_model.monthly_data)
            financial_model.run_model()

            # Get results from model
            monthly_data = financial_model.get_monthly_data()
            annual_data = financial_model.get_annual_data()

            # Display results in main area
            display_financial_summary(
                financial_model, revenue_model, cost_model)
    else:
        st.info(
            "Adjust the parameters in the sidebar and click 'Run Financial Model' to see the projections.")

        # If reports directory exists, show some existing outputs
        if os.path.exists('reports/combined/financial_summary.png'):
            st.subheader("Latest Model Results")

            col1, col2 = st.columns(2)
            with col1:
                st.image('reports/combined/financial_summary.png',
                         caption='Financial Summary', use_column_width=True)
            with col2:
                st.image('reports/combined/break_even_analysis.png',
                         caption='Break-Even Analysis', use_column_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.image('reports/combined/runway_and_capital.png',
                         caption='Runway and Capital', use_column_width=True)
            with col4:
                st.image('reports/combined/unit_economics.png',
                         caption='Unit Economics', use_column_width=True)


def display_financial_summary(financial_model, revenue_model, cost_model):
    """Display financial summary in the Streamlit app"""
    # Get data
    monthly_data = financial_model.get_monthly_data()
    annual_data = financial_model.get_annual_data()

    # Key metrics
    st.subheader("Key Financial Metrics")

    # Convert the metrics table to a nice format
    metrics_table = financial_model.get_key_metrics_table()
    st.dataframe(metrics_table, use_container_width=True)

    # Financial Summary
    st.subheader("Financial Summary")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary", "Break-Even", "Runway & Capital", "Unit Economics", "Growth"
    ])

    with tab1:
        fig = financial_model.plot_financial_summary(figsize=(12, 7))
        st.pyplot(fig)

    with tab2:
        fig = financial_model.plot_break_even_analysis(figsize=(12, 7))
        st.pyplot(fig)

    with tab3:
        fig = financial_model.plot_runway_and_capital(figsize=(12, 7))
        st.pyplot(fig)

    with tab4:
        fig = financial_model.plot_unit_economics(figsize=(12, 7))
        st.pyplot(fig)

    with tab5:
        fig = revenue_model.plot_growth_curves(figsize=(12, 7))
        st.pyplot(fig)

    # Calculate key metrics
    profitable_month_data = monthly_data[monthly_data['ebitda'] > 0]
    if len(profitable_month_data) > 0:
        profitable_month = profitable_month_data['month_number'].min()
        profitable_year = (profitable_month // 12) + 1
        profitable_month_in_year = (profitable_month % 12) or 12
        profitability_text = f"Month {profitable_month} (Year {profitable_year}, Month {profitable_month_in_year})"
    else:
        profitability_text = "Not reached within forecast period"

    # Calculate burn before profitability
    pre_profit_data = monthly_data[monthly_data['month_number'] < (
        profitable_month if len(profitable_month_data) > 0 else 99999)]
    negative_cash_flows = pre_profit_data[pre_profit_data['cash_flow']
                                          < 0]['cash_flow']
    total_burn = -negative_cash_flows.sum()

    # Calculate terminal metrics
    terminal_year = annual_data.iloc[-1]
    terminal_revenue = terminal_year['annual_revenue']
    terminal_ebitda = terminal_year['annual_ebitda']
    terminal_ebitda_margin = terminal_year['annual_ebitda_margin']

    # Display financial metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Month to Profitability", profitability_text)
        st.metric("Total Burn Before Profitability",
                  f"${total_burn/1000000:.2f}M")

    with col2:
        st.metric("Terminal Revenue (Year 6)",
                  f"${terminal_revenue/1000000:.2f}M")
        st.metric("Terminal EBITDA (Year 6)",
                  f"${terminal_ebitda/1000000:.2f}M")

    with col3:
        st.metric("Terminal EBITDA Margin",
                  f"{terminal_ebitda_margin*100:.1f}%")
        min_capital = monthly_data['capital'].min()
        st.metric("Min. Capital Position", f"${min_capital/1000000:.2f}M")

    # Valuation estimates
    st.subheader("Potential Valuation Estimates (Year 6)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("At 5x Revenue", f"${(terminal_revenue*5)/1000000:.2f}M")
    with col2:
        st.metric("At 8x Revenue", f"${(terminal_revenue*8)/1000000:.2f}M")
    with col3:
        st.metric("At 12x EBITDA", f"${(terminal_ebitda*12)/1000000:.2f}M")
    with col4:
        st.metric("At 18x EBITDA", f"${(terminal_ebitda*18)/1000000:.2f}M")

    # Raw data explorer
    st.subheader("Raw Data Explorer")
    tab6, tab7, tab8 = st.tabs(["Monthly Data", "Annual Data", "Cost Data"])

    with tab6:
        display_cols = ['date', 'monthly_revenue', 'total_cogs', 'total_operating_expenses',
                        'ebitda', 'ebitda_margin', 'cash_flow', 'capital']
        st.dataframe(monthly_data[display_cols])

    with tab7:
        st.dataframe(annual_data)

    with tab8:
        cost_annual = cost_model.get_annual_data()
        st.dataframe(cost_annual)


if __name__ == "__main__":
    main()
