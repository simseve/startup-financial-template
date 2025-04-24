#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime

# Import our model classes
from models.cost_model import AISaaSCostModel
from models.growth_model import SaaSGrowthModel
from models.financial_model import SaaSFinancialModel
from app import load_config

def optimize_for_annual_revenue(target_annual_revenue=360000000, target_year=5, initial_investment=20000000):
    """
    Optimize the financial model to achieve a specific annual revenue target by a given year.
    Uses segment-specific growth strategies for precise control.
    
    Parameters:
    -----------
    target_annual_revenue : float
        Target annual revenue to achieve (not ARR)
    target_year : int
        Target year to achieve the revenue (1-indexed, 5 = Year 5)
    initial_investment : float
        Initial investment amount
    
    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    # Convert year to end-of-year month for calculations
    target_month = target_year * 12
    
    print(f"Optimizing to achieve ${target_annual_revenue/1000000:.2f}M annual revenue by year {target_year}...")
    
    # Step 1: Load configurations
    revenue_config_path = "configs/revenue_config.json"
    cost_config_path = "configs/cost_config.json"
    
    revenue_config = load_config(revenue_config_path)
    if not revenue_config:
        raise ValueError(f"Failed to load revenue configuration from {revenue_config_path}")
    
    cost_config = load_config(cost_config_path)
    if not cost_config:
        raise ValueError(f"Failed to load cost configuration from {cost_config_path}")
    
    # Step 2: Create base models
    base_revenue_model = SaaSGrowthModel(revenue_config)
    
    # Step 3: Run the baseline model to get current numbers
    base_revenue_model.run_model()
    baseline_annual_data = base_revenue_model.get_annual_data()
    
    if target_year > len(baseline_annual_data):
        print(f"Error: Target year {target_year} exceeds the projection period of {len(baseline_annual_data)} years")
        return None, None, None, None
    
    # Get baseline annual revenue for the target year (actual annual revenue, not ARR)
    # Calculate baseline annual revenue as the sum of monthly revenues
    # Get monthly data for the target year
    baseline_year_months = base_revenue_model.get_monthly_data()[base_revenue_model.get_monthly_data()['year_number'] == target_year]
    # Annual revenue is the sum of monthly revenues (ARR/12) for all months in the year
    baseline_annual_revenue = baseline_year_months['total_arr'].sum() / 12
    print(f"Baseline model: ${baseline_annual_revenue/1000000:.2f}M annual revenue in year {target_year}")
    print(f"Target revenue: ${target_annual_revenue/1000000:.2f}M annual revenue")
    
    # Calculate growth needed
    growth_factor = target_annual_revenue / baseline_annual_revenue if baseline_annual_revenue > 0 else 20.0
    print(f"Overall growth factor needed: {growth_factor:.2f}x")
    
    # Step 4: Create a custom segment-specific growth strategy that tapers growth over time
    # This avoids excessive growth in later years by focusing more on initial years
    
    # Define a segment-specific growth strategy
    # Higher focus on Enterprise and Mid-Market, with more controlled SMB growth
    segment_multipliers = {
        'Enterprise': min(growth_factor * 1.2, 20),  # Slightly higher focus on Enterprise
        'Mid-Market': min(growth_factor * 1.1, 18),  # Moderate focus on Mid-Market
        'SMB': min(growth_factor * 0.9, 15)          # Lower multiplier for SMB to prevent excessive growth
    }
    
    # Further refine with segment-specific year-by-year strategy
    # This allows more controlled growth that avoids massive later-year increases
    segment_year_multipliers = {}
    for segment in revenue_config['segments']:
        base_multiplier = segment_multipliers[segment]
        # Taper the multiplier over time for more realistic growth
        segment_year_multipliers[segment] = {
            1: base_multiplier,                      # Year 1: Full strength
            2: base_multiplier * 0.9,                # Year 2: Slightly reduced
            3: base_multiplier * 0.8,                # Year 3: Further reduced
            4: base_multiplier * 0.7,                # Year 4: More controlled growth
            5: base_multiplier * 0.6,                # Year 5: Very controlled growth
            6: base_multiplier * 0.5                 # Year 6: Minimal additional growth
        }
    
    print("\nApplying segment-specific growth strategy:")
    for segment, years in segment_year_multipliers.items():
        print(f"  {segment}: ", end="")
        for year, multiplier in years.items():
            print(f"Y{year}: {multiplier:.1f}x ", end="")
        print()
    
    # Apply the segment-specific year-by-year growth strategy
    custom_model = base_revenue_model.apply_dynamic_growth_strategy(segment_year_multipliers)
    custom_model.run_model()
    custom_annual_data = custom_model.get_annual_data()
    
    # Check if we're close to target - using actual annual revenue
    # Get the true annual revenue by calculating sum of monthly revenue for the target year
    # Get monthly data for the target year
    target_year_months = custom_model.get_monthly_data()[custom_model.get_monthly_data()['year_number'] == target_year]
    # Annual revenue is the sum of monthly revenues (ARR/12) for all months in the year
    achieved_annual_revenue = target_year_months['total_arr'].sum() / 12
    relative_error = abs(achieved_annual_revenue - target_annual_revenue) / target_annual_revenue
    
    print(f"\nAchieved: ${achieved_annual_revenue/1000000:.2f}M annual revenue in year {target_year}")
    print(f"Target: ${target_annual_revenue/1000000:.2f}M annual revenue")
    print(f"Error: {relative_error*100:.1f}%")
    
    # Fine-tune if we're too far from target
    tolerance = 0.05  # 5% tolerance
    if relative_error > tolerance:
        # Adjust the strategy using the error ratio
        adjustment_factor = target_annual_revenue / achieved_annual_revenue
        print(f"\nFine-tuning growth strategy with adjustment factor: {adjustment_factor:.2f}x")
        
        # Apply fine-tuning
        adjusted_segment_year_multipliers = {}
        for segment in revenue_config['segments']:
            adjusted_segment_year_multipliers[segment] = {
                year: multiplier * adjustment_factor 
                for year, multiplier in segment_year_multipliers[segment].items()
            }
        
        # Apply the adjusted strategy
        custom_model = base_revenue_model.apply_dynamic_growth_strategy(adjusted_segment_year_multipliers)
        custom_model.run_model()
        custom_annual_data = custom_model.get_annual_data()
        
        # Check results - get the true annual revenue
        # Get monthly data for the target year
        target_year_months = custom_model.get_monthly_data()[custom_model.get_monthly_data()['year_number'] == target_year]
        # Annual revenue is the sum of monthly revenues (ARR/12) for all months in the year
        achieved_annual_revenue = target_year_months['total_arr'].sum() / 12
        relative_error = abs(achieved_annual_revenue - target_annual_revenue) / target_annual_revenue
        
        print(f"After fine-tuning: ${achieved_annual_revenue/1000000:.2f}M annual revenue (Error: {relative_error*100:.1f}%)")
    
    # Step 5: Run the cost model with the optimized revenue model
    revenue_model = custom_model
    cost_model = AISaaSCostModel(cost_config)
    cost_model.run_model(revenue_model)
    
    # Step 6: Create and run the financial model
    financial_model = SaaSFinancialModel(
        revenue_model=revenue_model, 
        cost_model=cost_model,
        initial_investment=initial_investment
    )
    financial_model.run_model()
    
    # Step 7: Analyze the results
    monthly_data = financial_model.get_monthly_data()
    annual_data = financial_model.get_annual_data()
    revenue_monthly_data = revenue_model.get_monthly_data()
    revenue_annual_data = revenue_model.get_annual_data()
    
    # Get customer data by segment at target year
    customers_by_segment = {}
    total_customers = revenue_annual_data.iloc[target_year-1]['total_ending_customers']
    
    for segment in revenue_config['segments']:
        segment_customers = revenue_annual_data.iloc[target_year-1][f'{segment}_ending_customers'] \
            if f'{segment}_ending_customers' in revenue_annual_data.columns else 0
        customers_by_segment[segment] = segment_customers
    
    # Store optimization results
    optimization_results = {
        "target": f"${target_annual_revenue/1000000:.2f}M annual revenue",
        "target_year": target_year,
        "achieved": achieved_annual_revenue,
        "total_customers_required": total_customers,
        "customers_by_segment": customers_by_segment,
        "s_curve_parameters": segment_year_multipliers
    }
    
    # Create directory for output
    os.makedirs('reports/optimization', exist_ok=True)
    
    # Save optimization results
    pd.DataFrame([optimization_results]).to_csv(
        'reports/optimization/annual_revenue_results.csv', index=False
    )
    
    # Generate visualization of required customer growth and revenue
    plt.figure(figsize=(12, 12))
    
    # Plot total customers over time
    plt.subplot(3, 1, 1)
    years = revenue_annual_data['year'].values
    customers = revenue_annual_data['total_ending_customers'].values
    plt.bar(years, customers)
    
    # Highlight target year
    plt.axvline(x=target_year, color='red', linestyle='--', alpha=0.7)
    plt.scatter(target_year, total_customers, s=100, color='red', zorder=5)
    plt.annotate(f"Target Year {target_year}\n{int(total_customers)} Customers", 
                xy=(target_year, total_customers), 
                xytext=(target_year+0.3, total_customers*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.title(f'Customer Growth Required to Reach ${target_annual_revenue/1000000:.2f}M Annual Revenue by Year {target_year}', 
              fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Total Customers')
    plt.grid(True, alpha=0.3)
    
    # Plot segment breakdown over time
    plt.subplot(3, 1, 2)
    segments = revenue_config['segments']
    segment_data = {}
    for segment in segments:
        col = f'{segment}_ending_customers'
        if col in revenue_annual_data.columns:
            segment_data[segment] = revenue_annual_data[col].values
    
    bottom = np.zeros(len(years))
    for segment, data in segment_data.items():
        plt.bar(years, data, bottom=bottom, label=segment)
        bottom += data
    
    plt.title('Customer Segments by Year', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Customers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot annual revenue growth
    plt.subplot(3, 1, 3)
    
    # Calculate annual revenue for each year
    annual_rev = []
    for i, year in enumerate(years):
        # Get monthly data for the year and calculate annual revenue
        year_months = revenue_model.get_monthly_data()[revenue_model.get_monthly_data()['year_number'] == year]
        yearly_revenue = year_months['total_arr'].sum() / 12 / 1000000
        annual_rev.append(yearly_revenue)
    
    plt.bar(years, annual_rev)
    
    # Highlight target year
    plt.axvline(x=target_year, color='red', linestyle='--', alpha=0.7)
    target_rev = annual_rev[target_year-1]  # List index is 0-based, year is 1-based
    plt.scatter(target_year, target_rev, s=100, color='red', zorder=5)
    plt.annotate(f"Target Year {target_year}\n${target_rev:.2f}M Revenue", 
                xy=(target_year, target_rev), 
                xytext=(target_year+0.3, target_rev*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add horizontal line at target revenue
    plt.axhline(y=target_annual_revenue/1000000, color='green', linestyle='--', alpha=0.7)
    plt.text(1, target_annual_revenue/1000000*1.05, f'Target Revenue: ${target_annual_revenue/1000000:.2f}M', 
             color='green', backgroundcolor='white', alpha=0.8)
    
    plt.title(f'Annual Revenue Growth to Reach Target', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Annual Revenue ($M)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/optimization/annual_revenue_growth.png', dpi=300, bbox_inches='tight')
    
    # Print optimization results
    print("\n" + "="*50)
    print(f"Optimization Results:")
    print(f"Target: ${target_annual_revenue/1000000:.2f}M annual revenue by year {target_year}")
    print(f"Achieved: ${achieved_annual_revenue/1000000:.2f}M annual revenue")
    print(f"Total customers required: {int(total_customers)}")
    print("\nCustomers by segment:")
    for segment, count in customers_by_segment.items():
        print(f"  - {segment}: {int(count)} customers")
        
    # Customer acquisition timeline
    print("\nCustomer Growth Year by Year:")
    
    # Create a table header
    print(f"\n{'Year':>8} {'Total':>10} ", end="")
    for segment in revenue_config['segments']:
        print(f"{segment:>12} ", end="")
    print()
    
    # Print data for each year
    for i, year in enumerate(years):
        total_cust = revenue_annual_data.iloc[i]['total_ending_customers']
        print(f"{int(year):>8} {int(total_cust):>10} ", end="")
        
        for segment in revenue_config['segments']:
            col = f'{segment}_ending_customers'
            if col in revenue_annual_data.columns:
                segment_count = revenue_annual_data.iloc[i][col]
                print(f"{int(segment_count):>12} ", end="")
        print()
        
    # Show revenue growth by year
    print("\nRevenue Growth Year by Year:")
    print(f"\n{'Year':>8} {'Annual Revenue ($M)':>20} {'YoY Growth':>15}")
    
    for i, year in enumerate(years):
        # Calculate actual annual revenue from monthly data
        # Get monthly data for this year
        year_months = revenue_model.get_monthly_data()[revenue_model.get_monthly_data()['year_number'] == year]
        # Sum monthly revenues
        yearly_revenue = year_months['total_arr'].sum() / 12 / 1000000
        
        # Calculate year-over-year growth
        if i > 0:
            # Get previous year's revenue from monthly data
            prev_year_months = revenue_model.get_monthly_data()[revenue_model.get_monthly_data()['year_number'] == year-1]
            prev_revenue = prev_year_months['total_arr'].sum() / 12 / 1000000
                
            yoy_growth = (yearly_revenue / prev_revenue - 1) * 100 if prev_revenue > 0 else float('inf')
            growth_str = f"{yoy_growth:.1f}%"
        else:
            growth_str = "N/A"
            
        print(f"{int(year):>8} {yearly_revenue:>20.2f} {growth_str:>15}")
    
    # Print original S-curve parameters
    print("\nOptimized S-curve Multipliers:")
    for segment, years_dict in segment_year_multipliers.items():
        print(f"  {segment}:")
        for year, multiplier in years_dict.items():
            print(f"    Year {year}: {multiplier:.2f}x")
    
    print("="*50)
    
    return financial_model, revenue_model, cost_model, optimization_results

def main():
    # Default values
    target_annual_revenue = 360000000  # $360M annual revenue
    target_year = 5  # Year 5
    initial_investment = 20000000  # $20M initial investment
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        target_annual_revenue = float(sys.argv[1])
    if len(sys.argv) > 2:
        target_year = int(sys.argv[2])
    if len(sys.argv) > 3:
        initial_investment = float(sys.argv[3])
    
    # Run the optimization
    financial_model, revenue_model, cost_model, _ = optimize_for_annual_revenue(
        target_annual_revenue=target_annual_revenue,
        target_year=target_year,
        initial_investment=initial_investment
    )
    
    # Generate additional reports
    if financial_model:
        # Generate additional detailed plots
        fig1 = financial_model.plot_financial_summary(figsize=(14, 8))
        fig1.savefig('reports/optimization/annual_revenue_financial_summary.png', dpi=300, bbox_inches='tight')
        
        fig2 = revenue_model.plot_growth_curves(figsize=(14, 10))
        fig2.savefig('reports/optimization/annual_revenue_growth_curves.png', dpi=300, bbox_inches='tight')
        
        fig3 = financial_model.plot_runway_and_capital(figsize=(14, 8))
        fig3.savefig('reports/optimization/annual_revenue_runway.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()