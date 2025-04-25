import os
import argparse
import json
import pandas as pd
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

def save_reports(growth_model, cost_model, financial_model):
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
    """
    # Create report directories if they don't exist
    os.makedirs(os.path.join('reports', 'growth'), exist_ok=True)
    os.makedirs(os.path.join('reports', 'cost'), exist_ok=True)
    os.makedirs(os.path.join('reports', 'combined'), exist_ok=True)
    os.makedirs(os.path.join('reports', 'optimization'), exist_ok=True)
    
    # Save growth model data and plots
    growth_model.get_monthly_data().to_csv(os.path.join('reports', 'growth', 'monthly_data.csv'), index=False)
    growth_model.get_annual_data().to_csv(os.path.join('reports', 'growth', 'annual_data.csv'), index=False)
    
    growth_summary = growth_model.display_summary_metrics()
    growth_summary.to_csv(os.path.join('reports', 'growth', 'growth_summary.csv'))
    
    growth_curves_fig = growth_model.plot_growth_curves()
    growth_curves_fig.savefig(os.path.join('reports', 'growth', 'growth_curves.png'), dpi=300, bbox_inches='tight')
    
    annual_metrics_fig = growth_model.plot_annual_metrics()
    annual_metrics_fig.savefig(os.path.join('reports', 'growth', 'annual_metrics.png'), dpi=300, bbox_inches='tight')
    
    segment_shares_fig = growth_model.plot_customer_segment_shares()
    segment_shares_fig.savefig(os.path.join('reports', 'growth', 'segment_shares.png'), dpi=300, bbox_inches='tight')
    
    # Save cost model data and plots
    cost_model.get_monthly_data().to_csv(os.path.join('reports', 'cost', 'monthly_cost_data.csv'), index=False)
    cost_model.get_annual_data().to_csv(os.path.join('reports', 'cost', 'annual_cost_data.csv'), index=False)
    
    cost_summary = cost_model.display_summary_metrics()
    cost_summary.to_csv(os.path.join('reports', 'cost', 'cost_summary.csv'))
    
    expense_breakdown_fig = cost_model.plot_expense_breakdown()
    expense_breakdown_fig.savefig(os.path.join('reports', 'cost', 'expense_breakdown.png'), dpi=300, bbox_inches='tight')
    
    headcount_growth_fig = cost_model.plot_headcount_growth()
    headcount_growth_fig.savefig(os.path.join('reports', 'cost', 'headcount_growth.png'), dpi=300, bbox_inches='tight')
    
    # Save financial model data and plots
    financial_model.get_monthly_data().to_csv(os.path.join('reports', 'combined', 'monthly_cashflow.csv'), index=False)
    financial_model.get_annual_data().to_csv(os.path.join('reports', 'combined', 'annual_cashflow.csv'), index=False)
    
    key_metrics = financial_model.get_key_metrics_table()
    key_metrics.to_csv(os.path.join('reports', 'combined', 'key_metrics.csv'))
    
    # Calculate unit economics
    unit_economics = cost_model.calculate_unit_economics(growth_model)
    unit_economics_table = cost_model.display_unit_economics_table(unit_economics)
    unit_economics_table.to_csv(os.path.join('reports', 'combined', 'unit_economics.csv'))
    
    unit_economics_fig = cost_model.plot_unit_economics(unit_economics)
    unit_economics_fig.savefig(os.path.join('reports', 'combined', 'unit_economics.png'), dpi=300, bbox_inches='tight')
    
    # Save combined financial plots
    financial_summary_fig = financial_model.plot_financial_summary()
    financial_summary_fig.savefig(os.path.join('reports', 'combined', 'financial_summary.png'), dpi=300, bbox_inches='tight')
    
    break_even_fig = financial_model.plot_break_even_analysis()
    break_even_fig.savefig(os.path.join('reports', 'combined', 'break_even_analysis.png'), dpi=300, bbox_inches='tight')
    
    runway_fig = financial_model.plot_runway_and_capital()
    runway_fig.savefig(os.path.join('reports', 'combined', 'runway_and_capital.png'), dpi=300, bbox_inches='tight')
    
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
                        choices=['baseline', 'profile'],
                        help='Strategy to use for the model (baseline or profile)')
                        
    # Growth profile options
    parser.add_argument('--profile', type=str, default='conservative',
                        choices=['conservative', 'baseline', 'aggressive', 'hypergrowth'],
                        help='Growth profile to use when strategy=profile')

    args = parser.parse_args()
    
    print(f"Running 2025 Financial Model with strategy: {args.strategy}")
    print(f"Initial investment: ${args.initial_investment:,.0f}")
    
    # Run the appropriate model based on strategy
    if args.strategy == 'baseline':
        run_baseline_model(args.initial_investment)
    elif args.strategy == 'profile':
        print(f"Using growth profile: {args.profile}")
        run_growth_profile_model(args.profile, args.initial_investment)

    print("\nModel run complete. Reports saved to the 'reports' directory.")

if __name__ == "__main__":
    main()