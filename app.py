import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.growth_model import SaaSGrowthModel
from models.cost_model import AISaaSCostModel
from models.financial_model import SaaSFinancialModel

def load_configs():
    """
    Load configuration files from the configs directory
    
    Returns:
    --------
    tuple : (revenue_config, cost_config)
    """
    # Load revenue configuration
    with open(os.path.join('configs', 'revenue_config.json'), 'r') as f:
        revenue_config = json.load(f)
    
    # Load cost configuration
    with open(os.path.join('configs', 'cost_config.json'), 'r') as f:
        cost_config = json.load(f)
        
    return revenue_config, cost_config

def save_reports(growth_model, cost_model, financial_model, prefix=""):
    """
    Save model reports to the reports directory
    
    Parameters:
    -----------
    growth_model : SaaSGrowthModel
        The growth model instance
    cost_model : AISaaSCostModel
        The cost model instance
    financial_model : SaaSFinancialModel
        The integrated financial model instance
    prefix : str, optional
        Prefix to add to report filenames
    """
    # Create report directories if they don't exist
    os.makedirs(os.path.join('reports', 'growth'), exist_ok=True)
    os.makedirs(os.path.join('reports', 'cost'), exist_ok=True)
    os.makedirs(os.path.join('reports', 'combined'), exist_ok=True)
    os.makedirs(os.path.join('reports', 'optimization'), exist_ok=True)
    
    # Add prefix to filenames if provided
    filename_prefix = f"{prefix}_" if prefix else ""
    
    # Save growth model data and plots
    growth_model.get_monthly_data().to_csv(os.path.join('reports', 'growth', f'{filename_prefix}monthly_data.csv'), index=False)
    growth_model.get_annual_data().to_csv(os.path.join('reports', 'growth', f'{filename_prefix}annual_data.csv'), index=False)
    
    growth_summary = growth_model.display_summary_metrics()
    growth_summary.to_csv(os.path.join('reports', 'growth', f'{filename_prefix}growth_summary.csv'))
    
    growth_curves_fig = growth_model.plot_growth_curves()
    growth_curves_fig.savefig(os.path.join('reports', 'growth', f'{filename_prefix}growth_curves.png'), dpi=300, bbox_inches='tight')
    
    annual_metrics_fig = growth_model.plot_annual_metrics()
    annual_metrics_fig.savefig(os.path.join('reports', 'growth', f'{filename_prefix}annual_metrics.png'), dpi=300, bbox_inches='tight')
    
    segment_shares_fig = growth_model.plot_customer_segment_shares()
    segment_shares_fig.savefig(os.path.join('reports', 'growth', f'{filename_prefix}segment_shares.png'), dpi=300, bbox_inches='tight')
    
    # Save cost model data and plots
    cost_model.get_monthly_data().to_csv(os.path.join('reports', 'cost', f'{filename_prefix}monthly_cost_data.csv'), index=False)
    cost_model.get_annual_data().to_csv(os.path.join('reports', 'cost', f'{filename_prefix}annual_cost_data.csv'), index=False)
    
    cost_summary = cost_model.display_summary_metrics()
    cost_summary.to_csv(os.path.join('reports', 'cost', f'{filename_prefix}cost_summary.csv'))
    
    expense_breakdown_fig = cost_model.plot_expense_breakdown()
    expense_breakdown_fig.savefig(os.path.join('reports', 'cost', f'{filename_prefix}expense_breakdown.png'), dpi=300, bbox_inches='tight')
    
    headcount_growth_fig = cost_model.plot_headcount_growth()
    headcount_growth_fig.savefig(os.path.join('reports', 'cost', f'{filename_prefix}headcount_growth.png'), dpi=300, bbox_inches='tight')
    
    # Save financial model data and plots
    financial_model.get_monthly_data().to_csv(os.path.join('reports', 'combined', f'{filename_prefix}monthly_cashflow.csv'), index=False)
    financial_model.get_annual_data().to_csv(os.path.join('reports', 'combined', f'{filename_prefix}annual_cashflow.csv'), index=False)
    
    key_metrics = financial_model.get_key_metrics_table()
    key_metrics.to_csv(os.path.join('reports', 'combined', f'{filename_prefix}key_metrics.csv'))
    
    # Calculate unit economics
    unit_economics = cost_model.calculate_unit_economics(growth_model)
    unit_economics_table = cost_model.display_unit_economics_table(unit_economics)
    unit_economics_table.to_csv(os.path.join('reports', 'combined', f'{filename_prefix}unit_economics.csv'))
    
    unit_economics_fig = cost_model.plot_unit_economics(unit_economics)
    unit_economics_fig.savefig(os.path.join('reports', 'combined', f'{filename_prefix}unit_economics.png'), dpi=300, bbox_inches='tight')
    
    # Save combined financial plots
    financial_summary_fig = financial_model.plot_financial_summary()
    financial_summary_fig.savefig(os.path.join('reports', 'combined', f'{filename_prefix}financial_summary.png'), dpi=300, bbox_inches='tight')
    
    break_even_fig = financial_model.plot_break_even_analysis()
    break_even_fig.savefig(os.path.join('reports', 'combined', f'{filename_prefix}break_even_analysis.png'), dpi=300, bbox_inches='tight')
    
    runway_fig = financial_model.plot_runway_and_capital()
    runway_fig.savefig(os.path.join('reports', 'combined', f'{filename_prefix}runway_and_capital.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')

def run_baseline_model(initial_investment=2000000):
    """
    Run the baseline financial model
    
    Parameters:
    -----------
    initial_investment : float, optional
        Initial capital investment amount
        
    Returns:
    --------
    tuple : (growth_model, cost_model, financial_model)
    """
    # Load configurations
    revenue_config, cost_config = load_configs()
    
    # Initialize models
    growth_model = SaaSGrowthModel(revenue_config)
    cost_model = AISaaSCostModel(cost_config)
    
    # Convert string year keys to integers in s_curve config
    for segment in growth_model.config['segments']:
        growth_model.config['s_curve'][segment] = {
            int(year): params for year, params in growth_model.config['s_curve'][segment].items() 
            if year != "_comment"
        }
    
    # Convert string month keys to integers in seasonality
    growth_model.config['seasonality'] = {
        int(month): factor for month, factor in growth_model.config['seasonality'].items()
        if month != "_comment"
    }
    
    # Convert string year keys to integers in headcount growth factors
    for dept in cost_model.config['headcount']:
        if dept != "_comment" and isinstance(cost_model.config['headcount'][dept], dict):
            if 'growth_factors' in cost_model.config['headcount'][dept]:
                cost_model.config['headcount'][dept]['growth_factors'] = {
                    int(year): factor for year, factor in cost_model.config['headcount'][dept]['growth_factors'].items()
                    if year != "_comment"
                }
    
    # Run growth model
    growth_model.run_model()
    
    # Run cost model with growth model
    cost_model.run_model(growth_model)
    
    # Initialize and run financial model
    financial_model = SaaSFinancialModel(
        revenue_model=growth_model,
        cost_model=cost_model,
        initial_investment=initial_investment
    )
    financial_model.run_model()
    
    # Save reports
    save_reports(growth_model, cost_model, financial_model)
    
    # Print summary table
    print("\nFinancial Summary:\n" + "-" * 100)
    key_metrics = financial_model.get_key_metrics_table()
    
    # Create a summary table with the most important metrics
    summary_columns = ['ARR ($M)', 'Revenue ($M)', 'Total Expenses ($M)', 'EBITDA ($M)', 'EBITDA Margin (%)', 
                      'Customers', 'Headcount', 'CAC ($)', 'LTV ($)', 'LTV/CAC Ratio', 'Rule of 40 Score', 'Capital Position ($M)']
    
    # Print header
    header = f"{'Year':<8}"
    for col in summary_columns:
        header += f"{col:<15}"
    print(header)
    print("-" * 175)
    
    # Print data rows
    for idx, year in enumerate(key_metrics.index):
        row = f"{year:<8}"
        for col in summary_columns:
            if col in key_metrics.columns:
                row += f"{key_metrics.iloc[idx][col]:<15}"
        print(row)
    
    # Print growth rates
    print("\nGrowth Metrics:")
    if len(key_metrics.index) > 1:
        for idx in range(1, len(key_metrics.index)):
            prev_arr = float(key_metrics.iloc[idx-1]['ARR ($M)'].replace('$', '').replace('M', ''))
            curr_arr = float(key_metrics.iloc[idx]['ARR ($M)'].replace('$', '').replace('M', ''))
            arr_growth = ((curr_arr / prev_arr) - 1) * 100 if prev_arr > 0 else 0
            
            prev_customers = int(key_metrics.iloc[idx-1]['Customers'].replace(',', ''))
            curr_customers = int(key_metrics.iloc[idx]['Customers'].replace(',', ''))
            customer_growth = ((curr_customers / prev_customers) - 1) * 100 if prev_customers > 0 else 0
            
            year = key_metrics.index[idx]
            print(f"{year}: ARR Growth: {arr_growth:.1f}% | Customer Growth: {customer_growth:.1f}%")
    
    # Print unit economics summary
    print("\nUnit Economics Summary:")
    unit_economics = cost_model.calculate_unit_economics(growth_model)
    unit_econ_table = cost_model.display_unit_economics_table(unit_economics)
    
    # Only display a few key unit economics metrics for the latest year
    latest_year_idx = len(unit_econ_table.index) - 1
    latest_year = unit_econ_table.index[latest_year_idx]
    churn = unit_econ_table.loc[latest_year, 'Effective Churn Rate (%)']
    lifetime = unit_econ_table.loc[latest_year, 'Customer Lifetime (Years)']
    payback = unit_econ_table.loc[latest_year, 'CAC Payback (Months)']
    
    print(f"Latest Metrics: Churn Rate: {churn} | Avg. Customer Lifetime: {lifetime} | CAC Payback: {payback}")
    
    # Print break-even point
    monthly_data = financial_model.get_monthly_data()
    if 'profitable_month' in monthly_data.columns:
        profitable_months = monthly_data[monthly_data['profitable_month'] == True]
        if len(profitable_months) > 0:
            first_profitable = profitable_months.iloc[0]
            month_number = first_profitable['month_number']
            year = first_profitable['year']
            print(f"\nBreak-even Point: Month {month_number} (Year {year}, Month {first_profitable['month']})")
        else:
            print("\nBreak-even Point: Not achieved within projection period (6 years)")
    
    # Print capital efficiency
    total_investment = initial_investment
    latest_year_idx = len(key_metrics.index) - 1
    final_arr = float(key_metrics.iloc[latest_year_idx]['ARR ($M)'].replace('$', '').replace('M', '')) * 1000000
    capital_efficiency = final_arr / total_investment if total_investment > 0 else 0
    
    print(f"\nCapital Efficiency: ${capital_efficiency:.2f} ARR per $1 invested")
    
    print("-" * 100)
    
    return growth_model, cost_model, financial_model

def run_growth_profile_model(profile_name, initial_investment=5000000):
    """
    Run the model with a specific growth profile
    
    Parameters:
    -----------
    profile_name : str
        Growth profile name ('baseline', 'conservative', 'aggressive', 'hypergrowth')
    initial_investment : float, optional
        Initial capital investment amount
        
    Returns:
    --------
    tuple : (growth_model, cost_model, financial_model)
    """
    # Load configurations
    revenue_config, cost_config = load_configs()
    
    # Initialize base model
    base_model = SaaSGrowthModel(revenue_config)
    
    # Convert string year keys to integers in s_curve config
    for segment in base_model.config['segments']:
        base_model.config['s_curve'][segment] = {
            int(year): params for year, params in base_model.config['s_curve'][segment].items()
            if year != "_comment"
        }
    
    # Convert string month keys to integers in seasonality
    base_model.config['seasonality'] = {
        int(month): factor for month, factor in base_model.config['seasonality'].items()
        if month != "_comment"
    }
    
    # Initialize cost model
    cost_model = AISaaSCostModel(cost_config)
    
    # Convert string year keys to integers in headcount growth factors
    for dept in cost_model.config['headcount']:
        if dept != "_comment" and isinstance(cost_model.config['headcount'][dept], dict):
            if 'growth_factors' in cost_model.config['headcount'][dept]:
                cost_model.config['headcount'][dept]['growth_factors'] = {
                    int(year): factor for year, factor in cost_model.config['headcount'][dept]['growth_factors'].items()
                    if year != "_comment"
                }
    
    # Apply growth profile
    growth_model = base_model.apply_growth_profile(profile_name)
    
    # Run growth model
    growth_model.run_model()
    
    # Run cost model with growth model
    cost_model.run_model(growth_model)
    
    # Initialize and run financial model
    financial_model = SaaSFinancialModel(
        revenue_model=growth_model,
        cost_model=cost_model,
        initial_investment=initial_investment
    )
    financial_model.run_model()
    
    # Save reports
    save_reports(growth_model, cost_model, financial_model)
    
    # Print summary table
    print(f"\nFinancial Summary ({profile_name.capitalize()} Growth Profile):\n" + "-" * 100)
    key_metrics = financial_model.get_key_metrics_table()
    
    # Create a summary table with the most important metrics
    summary_columns = ['ARR ($M)', 'Revenue ($M)', 'Total Expenses ($M)', 'EBITDA ($M)', 'EBITDA Margin (%)', 
                      'Customers', 'Headcount', 'CAC ($)', 'LTV ($)', 'LTV/CAC Ratio', 'Rule of 40 Score', 'Capital Position ($M)']
    
    # Print header
    header = f"{'Year':<8}"
    for col in summary_columns:
        header += f"{col:<15}"
    print(header)
    print("-" * 175)
    
    # Print data rows
    for idx, year in enumerate(key_metrics.index):
        row = f"{year:<8}"
        for col in summary_columns:
            if col in key_metrics.columns:
                row += f"{key_metrics.iloc[idx][col]:<15}"
        print(row)
    
    # Print growth rates
    print("\nGrowth Metrics:")
    if len(key_metrics.index) > 1:
        for idx in range(1, len(key_metrics.index)):
            prev_arr = float(key_metrics.iloc[idx-1]['ARR ($M)'].replace('$', '').replace('M', ''))
            curr_arr = float(key_metrics.iloc[idx]['ARR ($M)'].replace('$', '').replace('M', ''))
            arr_growth = ((curr_arr / prev_arr) - 1) * 100 if prev_arr > 0 else 0
            
            prev_customers = int(key_metrics.iloc[idx-1]['Customers'].replace(',', ''))
            curr_customers = int(key_metrics.iloc[idx]['Customers'].replace(',', ''))
            customer_growth = ((curr_customers / prev_customers) - 1) * 100 if prev_customers > 0 else 0
            
            year = key_metrics.index[idx]
            print(f"{year}: ARR Growth: {arr_growth:.1f}% | Customer Growth: {customer_growth:.1f}%")
    
    # Print break-even point
    monthly_data = financial_model.get_monthly_data()
    if 'profitable_month' in monthly_data.columns:
        profitable_months = monthly_data[monthly_data['profitable_month'] == True]
        if len(profitable_months) > 0:
            first_profitable = profitable_months.iloc[0]
            month_number = first_profitable['month_number']
            year = first_profitable['year']
            print(f"\nBreak-even Point: Month {month_number} (Year {year}, Month {first_profitable['month']})")
        else:
            print("\nBreak-even Point: Not achieved within projection period (6 years)")
    
    # Print capital efficiency
    total_investment = initial_investment
    latest_year_idx = len(key_metrics.index) - 1
    final_arr = float(key_metrics.iloc[latest_year_idx]['ARR ($M)'].replace('$', '').replace('M', '')) * 1000000
    capital_efficiency = final_arr / total_investment if total_investment > 0 else 0
    
    print(f"\nCapital Efficiency: ${capital_efficiency:.2f} ARR per $1 invested")
    
    print("-" * 100)
    
    return growth_model, cost_model, financial_model

def run_european_strategy(breakeven_target_month, annual_revenue_target, revenue_target_year, initial_investment=5000000, max_iterations=10):
    """
    Run a specialized European market strategy optimizing for:
    1. Initial focus on Enterprise and Mid-Market
    2. Strong SMB growth in Year 2 and beyond
    3. Achieving breakeven by a target month
    4. Reaching specific annual revenue target
    5. Minimizing total number of customers needed
    
    Parameters:
    -----------
    breakeven_target_month : int
        Target month to achieve breakeven
    annual_revenue_target : float
        Target annual revenue in USD
    revenue_target_year : int
        Year to achieve the annual revenue target (1-6)
    initial_investment : float, optional
        Initial capital investment amount
    max_iterations : int, optional
        Maximum number of iterations for optimization
        
    Returns:
    --------
    tuple : (growth_model, cost_model, financial_model)
    """
    print(f"\nRunning European Market Strategy Optimization:\n" + "-" * 100)
    print(f"Breakeven Target: Month {breakeven_target_month}")
    print(f"Annual Revenue Target: ${annual_revenue_target/1000000:.1f}M in Year {revenue_target_year}")
    print(f"Initial Investment: ${initial_investment:,.0f}")
    print("-" * 100)
    
    # Load configurations
    revenue_config, cost_config = load_configs()
    
    # Initialize base model
    base_model = SaaSGrowthModel(revenue_config)
    
    # Convert string year keys to integers in s_curve config
    for segment in base_model.config['segments']:
        base_model.config['s_curve'][segment] = {
            int(year): params for year, params in base_model.config['s_curve'][segment].items()
            if year != "_comment"
        }
    
    # Convert string month keys to integers in seasonality
    base_model.config['seasonality'] = {
        int(month): factor for month, factor in base_model.config['seasonality'].items()
        if month != "_comment"
    }
    
    # Initialize cost model
    cost_model = AISaaSCostModel(cost_config)
    
    # Convert string year keys to integers in headcount growth factors
    for dept in cost_model.config['headcount']:
        if dept != "_comment" and isinstance(cost_model.config['headcount'][dept], dict):
            if 'growth_factors' in cost_model.config['headcount'][dept]:
                cost_model.config['headcount'][dept]['growth_factors'] = {
                    int(year): factor for year, factor in cost_model.config['headcount'][dept]['growth_factors'].items()
                    if year != "_comment"
                }
    
    # Initialize segment-year growth multipliers based on target year
    segment_year_multipliers = {}
    
    # Enterprise segment
    segment_year_multipliers['Enterprise'] = {}
    if revenue_target_year >= 5:
        # Much more conservative growth for longer-term targets
        segment_year_multipliers['Enterprise'][1] = 0.5  # Very limited focus in year 1
        segment_year_multipliers['Enterprise'][2] = 0.4  # Very limited focus in year 2
        segment_year_multipliers['Enterprise'][3] = 0.3  # Minimal focus in year 3
        segment_year_multipliers['Enterprise'][4] = 0.3  # Minimal focus in year 4
        segment_year_multipliers['Enterprise'][5] = 0.3  # Minimal focus in year 5
        segment_year_multipliers['Enterprise'][6] = 0.3  # Minimal focus in year 6
    else:
        # Standard growth for earlier targets
        segment_year_multipliers['Enterprise'][1] = 2.0  # Strong focus in year 1
        segment_year_multipliers['Enterprise'][2] = 1.5  # Continued focus in year 2
        segment_year_multipliers['Enterprise'][3] = 1.0  # Standard growth in year 3
        segment_year_multipliers['Enterprise'][4] = 0.7  # Reduced focus in year 4
        segment_year_multipliers['Enterprise'][5] = 0.5  # Minimal focus in year 5
        segment_year_multipliers['Enterprise'][6] = 0.3  # Minimal focus in year 6
    
    # Mid-Market segment
    segment_year_multipliers['Mid-Market'] = {}
    if revenue_target_year >= 5:
        segment_year_multipliers['Mid-Market'][1] = 0.3  # Limited focus in year 1
        segment_year_multipliers['Mid-Market'][2] = 0.5  # Limited focus in year 2
        segment_year_multipliers['Mid-Market'][3] = 0.7  # Moderate focus in year 3
        segment_year_multipliers['Mid-Market'][4] = 0.9  # Standard focus in year 4
        segment_year_multipliers['Mid-Market'][5] = 1.2  # Increased focus in year 5
        segment_year_multipliers['Mid-Market'][6] = 1.0  # Standard focus in year 6
    else:
        segment_year_multipliers['Mid-Market'][1] = 1.5  # Strong focus in year 1
        segment_year_multipliers['Mid-Market'][2] = 2.0  # Increased focus in year 2
        segment_year_multipliers['Mid-Market'][3] = 2.5  # Peak focus in year 3
        segment_year_multipliers['Mid-Market'][4] = 2.0  # Strong focus in year 4
        segment_year_multipliers['Mid-Market'][5] = 1.5  # Moderate focus in year 5
        segment_year_multipliers['Mid-Market'][6] = 1.2  # Continued focus in year 6
    
    # SMB segment
    segment_year_multipliers['SMB'] = {}
    if revenue_target_year >= 5:
        segment_year_multipliers['SMB'][1] = 0.2  # Very limited focus in year 1
        segment_year_multipliers['SMB'][2] = 0.3  # Very limited focus in year 2
        segment_year_multipliers['SMB'][3] = 0.5  # Limited focus in year 3
        segment_year_multipliers['SMB'][4] = 0.7  # Moderate focus in year 4
        segment_year_multipliers['SMB'][5] = 1.0  # Standard focus in year 5
        segment_year_multipliers['SMB'][6] = 1.2  # Increased focus in year 6
    else:
        segment_year_multipliers['SMB'][1] = 0.5  # Limited focus in year 1
        segment_year_multipliers['SMB'][2] = 1.5  # Increased focus in year 2
        segment_year_multipliers['SMB'][3] = 2.5  # Strong focus in year 3
        segment_year_multipliers['SMB'][4] = 3.0  # Peak focus in year 4
        segment_year_multipliers['SMB'][5] = 3.5  # Sustained peak focus in year 5
        segment_year_multipliers['SMB'][6] = 3.0  # Strong focus in year 6
    
    best_model = None
    best_financial_model = None
    best_cost_model = None
    best_total_customers = float('inf')
    best_multipliers = None
    best_breakeven_month = None
    best_revenue = 0
    best_score = float('inf')  # Initialize best score (lower is better)
    
    print("Starting optimization iterations...")
    
    # Iterative optimization
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration+1}/{max_iterations}")
        
        # Apply growth strategy with current multipliers
        growth_model = base_model.apply_dynamic_growth_strategy(segment_year_multipliers)
        
        # Run growth model
        growth_model.run_model()
        
        # Run cost model with growth model
        cost_model.run_model(growth_model)
        
        # Initialize and run financial model
        financial_model = SaaSFinancialModel(
            revenue_model=growth_model,
            cost_model=cost_model,
            initial_investment=initial_investment
        )
        financial_model.run_model()
        
        # Check if we have achieved our targets
        monthly_data = financial_model.get_monthly_data()
        annual_data = financial_model.get_annual_data()
        
        # 1. Check breakeven target
        breakeven_achieved = False
        actual_breakeven_month = None
        
        if 'profitable_month' in monthly_data.columns:
            profitable_months = monthly_data[monthly_data['profitable_month'] == True]
            if len(profitable_months) > 0:
                actual_breakeven_month = profitable_months.iloc[0]['month_number']
                breakeven_achieved = actual_breakeven_month <= breakeven_target_month
                print(f"Breakeven in month {actual_breakeven_month} (target: {breakeven_target_month})")
            else:
                print("No breakeven achieved within projection period")
        
        # 2. Check revenue target - we want to be close to target WITHOUT exceeding it
        revenue_achieved = False
        actual_revenue = 0
        
        if revenue_target_year <= len(annual_data):
            target_year_idx = revenue_target_year - 1
            actual_revenue = annual_data.iloc[target_year_idx]['annual_revenue']
            
            # Calculate how close we are to the target as a percentage
            revenue_ratio = actual_revenue / annual_revenue_target
            
            # We want revenue that is 90-100% of target, not exceeding it
            # Anything below 90% is too low, anything above 100% is too high
            revenue_achieved = 0.90 <= revenue_ratio <= 1.00
            
            print(f"Year {revenue_target_year} revenue: ${actual_revenue/1000000:.2f}M (target: ${annual_revenue_target/1000000:.2f}M, ratio: {revenue_ratio:.2f})")
        
        # 3. Get total customers at target year
        total_customers = None
        if revenue_target_year <= len(annual_data):
            target_year_idx = revenue_target_year - 1
            total_customers = monthly_data[monthly_data['year_number'] == revenue_target_year]['total_customers'].max()
            print(f"Total customers at target year: {total_customers}")
        
        # Check if this is the best solution so far
        is_best = False
        
        # Only consider solutions that meet both targets, or improve on one target without worsening the other too much
        # For our specific goal of hitting $30M in Year 5 (not exceeding it) and breaking even by Year 4 (month 48)
        
        # Calculate how close we are to the target revenue (0.95-1.00 is ideal)
        revenue_closeness = actual_revenue / annual_revenue_target if actual_revenue > 0 else 0
        
        # Calculate how close we are to target breakeven (lower is better, 1.0 is exact)
        breakeven_closeness = actual_breakeven_month / breakeven_target_month if actual_breakeven_month else 2.0
        
        # We don't want to exceed the revenue target (penalty for going over)
        revenue_penalty = max(0, revenue_closeness - 1.0) * 10
        
        # Calculate a score (lower is better)
        # Prioritize:
        # 1. Being close to but not over revenue target
        # 2. Breaking even by target month
        # 3. Minimizing customer count
        
        # Adjust weights to prioritize what matters most
        revenue_weight = 10.0
        breakeven_weight = 5.0
        customer_weight = 0.01
        
        # Calculate overall score (lower is better)
        # Perfect score would be close to 0
        score = abs(0.98 - revenue_closeness) * revenue_weight + \
               (breakeven_closeness if breakeven_closeness <= 1.0 else breakeven_closeness * 2) * breakeven_weight + \
               (total_customers / 100) * customer_weight + \
               revenue_penalty
               
        print(f"Solution score: {score:.2f} (lower is better)")
        
        # Check if this solution is better than the current best
        if best_model is None or score < best_score:
            is_best = True
            best_score = score
        else:
            is_best = False
        
        # Store the best model
        if is_best:
            best_model = growth_model
            best_financial_model = financial_model
            best_cost_model = cost_model
            best_total_customers = total_customers if total_customers else float('inf')
            best_multipliers = segment_year_multipliers.copy()
            best_breakeven_month = actual_breakeven_month
            best_revenue = actual_revenue
            print("New best solution found!")
        
        # Adjust multipliers for next iteration based on results
        if not breakeven_achieved and actual_breakeven_month:
            # Breakeven adjustment - increase early-year growth for faster breakeven
            segment_year_multipliers['Enterprise'][1] *= 1.1
            segment_year_multipliers['Mid-Market'][1] *= 1.1
            segment_year_multipliers['Mid-Market'][2] *= 1.05
        
        # Handle revenue target adjustments
        revenue_ratio = actual_revenue / annual_revenue_target if actual_revenue > 0 else 0
        
        if revenue_ratio < 0.90:
            # Revenue too low - increase growth but with smaller increments
            for year in range(1, revenue_target_year):
                # Smaller adjustment factor, more conservative
                adjustment_factor = min(1.1, (revenue_target_year - year) * 0.03 + 1.0)
                
                if year == 1:
                    segment_year_multipliers['Enterprise'][year] *= adjustment_factor
                elif year == 2:
                    segment_year_multipliers['Mid-Market'][year] *= adjustment_factor
                else:
                    segment_year_multipliers['SMB'][year] *= adjustment_factor
                    segment_year_multipliers['Mid-Market'][year] *= adjustment_factor
        
        elif revenue_ratio > 1.00:
            # Revenue too high - need to reduce growth more aggressively
            reduction_factor = max(0.5, 1.0 - (revenue_ratio - 1.0) * 2.0)
            
            # Apply more reduction to earlier years which have compounding effects
            for year in range(1, revenue_target_year):
                # More aggressive reduction for earlier years (which compound more)
                year_specific_factor = reduction_factor ** (2.0 / (year + 1))
                
                segment_year_multipliers['Enterprise'][year] *= year_specific_factor
                segment_year_multipliers['Mid-Market'][year] *= year_specific_factor
                segment_year_multipliers['SMB'][year] *= year_specific_factor
        
        # Breakeven timing adjustments        
        if breakeven_achieved and actual_breakeven_month < breakeven_target_month * 0.8:
            # We're achieving breakeven too early, can reduce early growth
            segment_year_multipliers['Enterprise'][1] *= 0.95
            segment_year_multipliers['Mid-Market'][1] *= 0.95
                
        # Cap multipliers at reasonable values to avoid extreme growth
        for segment in segment_year_multipliers:
            for year in segment_year_multipliers[segment]:
                segment_year_multipliers[segment][year] = min(4.0, max(0.1, segment_year_multipliers[segment][year]))
    
    if best_model:
        print("\nOptimization completed. Final results:")
        
        # Display the best multipliers
        print("\nOptimized Growth Multipliers:")
        for segment in best_multipliers:
            print(f"{segment}: ", end="")
            for year in range(1, 7):
                print(f"Y{year}:{best_multipliers[segment][year]:.2f} ", end="")
            print("")
        
        # Print summary of best model
        monthly_data = best_financial_model.get_monthly_data()
        annual_data = best_financial_model.get_annual_data()
        
        # Print breakeven status
        if 'profitable_month' in monthly_data.columns:
            profitable_months = monthly_data[monthly_data['profitable_month'] == True]
            if len(profitable_months) > 0:
                first_profitable = profitable_months.iloc[0]
                month_number = first_profitable['month_number']
                year = first_profitable['year']
                month = first_profitable['month']
                print(f"\nBreak-even Point: Month {month_number} (Year {year}, Month {month})")
            else:
                print("\nBreak-even Point: Not achieved within projection period (6 years)")
        
        # Print revenue status
        if revenue_target_year <= len(annual_data):
            target_year_idx = revenue_target_year - 1
            actual_revenue = annual_data.iloc[target_year_idx]['annual_revenue']
            print(f"Year {revenue_target_year} revenue: ${actual_revenue/1000000:.2f}M (target: ${annual_revenue_target/1000000:.2f}M)")
        
        # Print customer count
        total_customers = None
        if revenue_target_year <= len(annual_data):
            total_customers = monthly_data[monthly_data['year_number'] == revenue_target_year]['total_customers'].max()
            print(f"Total customers at target year: {int(total_customers)}")
        
        try:
            # Save reports with "optimized" prefix
            save_reports(best_model, best_cost_model, best_financial_model, prefix="optimized")
        except Exception as e:
            print(f"\nError saving reports: {e}")
            print("However, optimization was successful. Here are the key results:")
            
            # Display key metrics manually
            monthly_data = best_financial_model.get_monthly_data()
            annual_data = best_financial_model.get_annual_data()
            
            print("\nAnnual Revenue:")
            for year in range(1, 6):
                if year <= len(annual_data):
                    year_revenue = annual_data.iloc[year-1]['annual_revenue'] / 1000000
                    print(f"Year {year}: ${year_revenue:.2f}M")
                    
            print("\nTotal Customers by Year End:")
            for year in range(1, 6):
                if year <= len(annual_data):
                    customers = best_model.monthly_data[best_model.monthly_data['year_number'] == year]['total_customers'].max()
                    print(f"Year {year}: {int(customers)}")
                    
            # Still return the models for potential further analysis
            return best_model, best_cost_model, best_financial_model
        
        # Return the best models
        return best_model, best_cost_model, best_financial_model
    else:
        print("Optimization failed - no viable solution found.")
        return None, None, None

def main():
    """
    Main function to parse arguments and run the appropriate model
    """
    parser = argparse.ArgumentParser(description='Run 2025 Financial Model')
    
    # Basic parameters
    parser.add_argument('--initial-investment', type=float, default=5000000,
                        help='Initial capital investment amount (default: 5,000,000)')
    
    # Strategy options
    parser.add_argument('--strategy', type=str, default='baseline',
                        choices=['baseline', 'profile', 'european'],
                        help='Strategy to use for the model (baseline, profile, or european)')
                        
    # Growth profile options
    parser.add_argument('--profile', type=str, default='conservative',
                        choices=['conservative', 'baseline', 'aggressive', 'hypergrowth'],
                        help='Growth profile to use when strategy=profile')
    
    # European strategy options
    parser.add_argument('--breakeven-target', type=int, default=24,
                        help='Target month to achieve breakeven (for european strategy)')
    parser.add_argument('--revenue-target', type=float, default=10000000,
                        help='Annual revenue target in USD (for european strategy)')
    parser.add_argument('--revenue-target-year', type=int, default=3, choices=range(1, 7),
                        help='Year to achieve annual revenue target (1-6) (for european strategy)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum optimization iterations (for european strategy)')

    args = parser.parse_args()
    
    print(f"Running 2025 Financial Model with strategy: {args.strategy}")
    print(f"Initial investment: ${args.initial_investment:,.0f}")
    
    # Run the appropriate model based on strategy
    if args.strategy == 'baseline':
        run_baseline_model(args.initial_investment)
    elif args.strategy == 'profile':
        print(f"Using growth profile: {args.profile}")
        run_growth_profile_model(args.profile, args.initial_investment)
    elif args.strategy == 'european':
        print(f"Using European Strategy with:")
        print(f" - Breakeven target: Month {args.breakeven_target}")
        print(f" - Revenue target: ${args.revenue_target/1000000:.1f}M in Year {args.revenue_target_year}")
        run_european_strategy(
            args.breakeven_target,
            args.revenue_target,
            args.revenue_target_year,
            args.initial_investment,
            args.max_iterations
        )

    print("\nModel run complete. Reports saved to the 'reports' directory.")

if __name__ == "__main__":
    main()