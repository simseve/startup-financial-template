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

def optimize_for_specific_revenue(target_revenue=30000000, target_month=60, initial_investment=20000000):
    """
    Optimize the financial model to achieve exactly $30M monthly revenue by month 60.
    This is a specialized version that carefully controls segment growth to avoid overshooting.
    
    Parameters:
    -----------
    target_revenue : float
        Target monthly revenue to achieve (not ARR)
    target_month : int
        Target month to achieve the revenue (1-indexed)
    initial_investment : float
        Initial investment amount
    
    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    print(f"Optimizing to achieve ${target_revenue/1000000:.2f}M monthly revenue by month {target_month}...")
    
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
    baseline_monthly_data = base_revenue_model.get_monthly_data()
    
    if target_month > len(baseline_monthly_data):
        print(f"Error: Target month {target_month} exceeds the projection period of {len(baseline_monthly_data)} months")
        return None, None, None, None
    
    baseline_monthly_revenue = baseline_monthly_data.loc[target_month-1, 'total_arr'] / 12
    print(f"Baseline model: ${baseline_monthly_revenue/1000000:.2f}M monthly revenue in month {target_month}")
    print(f"Target revenue: ${target_revenue/1000000:.2f}M")
    
    # Calculate growth needed
    growth_factor = target_revenue / baseline_monthly_revenue if baseline_monthly_revenue > 0 else 20.0
    print(f"Overall growth factor needed: {growth_factor:.2f}x")
    
    # Step 4: Create a custom segment-specific growth strategy that tapers growth over time
    # This avoids excessive growth in later years by focusing more on initial years
    
    # Get segment values from baseline
    baseline_segment_customers = {}
    for segment in revenue_config['segments']:
        if f'{segment}_customers' in baseline_monthly_data.columns:
            baseline_segment_customers[segment] = baseline_monthly_data.loc[target_month-1, f'{segment}_customers']
    
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
    custom_monthly_data = custom_model.get_monthly_data()
    
    # Check if we're close to target
    achieved_monthly_revenue = custom_monthly_data.loc[target_month-1, 'total_arr'] / 12
    relative_error = abs(achieved_monthly_revenue - target_revenue) / target_revenue
    
    print(f"\nAchieved: ${achieved_monthly_revenue/1000000:.2f}M monthly revenue at month {target_month}")
    print(f"Target: ${target_revenue/1000000:.2f}M monthly revenue")
    print(f"Error: {relative_error*100:.1f}%")
    
    # Fine-tune if we're too far from target
    tolerance = 0.05  # 5% tolerance
    if relative_error > tolerance:
        # Adjust the strategy using the error ratio
        adjustment_factor = target_revenue / achieved_monthly_revenue
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
        custom_monthly_data = custom_model.get_monthly_data()
        
        # Check results
        achieved_monthly_revenue = custom_monthly_data.loc[target_month-1, 'total_arr'] / 12
        relative_error = abs(achieved_monthly_revenue - target_revenue) / target_revenue
        
        print(f"After fine-tuning: ${achieved_monthly_revenue/1000000:.2f}M monthly revenue (Error: {relative_error*100:.1f}%)")
    
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
    revenue_monthly_data = revenue_model.get_monthly_data()
    
    # Get customer data by segment
    customers_by_segment = {}
    total_customers = revenue_monthly_data.loc[target_month-1, 'total_customers']
    
    for segment in revenue_config['segments']:
        segment_customers = revenue_monthly_data.loc[target_month-1, f'{segment}_customers'] \
            if f'{segment}_customers' in revenue_monthly_data.columns else 0
        customers_by_segment[segment] = segment_customers
    
    # Store optimization results
    optimization_results = {
        "target": f"${target_revenue/1000000:.2f}M monthly revenue",
        "target_month": target_month,
        "achieved": achieved_monthly_revenue,
        "total_customers_required": total_customers,
        "customers_by_segment": customers_by_segment
    }
    
    # Create directory for output
    os.makedirs('reports/optimization', exist_ok=True)
    
    # Save optimization results
    pd.DataFrame([optimization_results]).to_csv(
        'reports/optimization/specific_revenue_results.csv', index=False
    )
    
    # Generate visualization of required customer growth
    plt.figure(figsize=(12, 8))
    
    # Plot total customers over time
    plt.subplot(2, 1, 1)
    plt.plot(revenue_monthly_data['month_number'], revenue_monthly_data['total_customers'], 
             linewidth=2, marker='o', markersize=4)
    
    # Highlight target month
    plt.axvline(x=target_month, color='red', linestyle='--', alpha=0.7)
    plt.scatter(target_month, total_customers, s=100, color='red', zorder=5)
    plt.annotate(f"Target Month {target_month}\n{int(total_customers)} Customers", 
                xy=(target_month, total_customers), 
                xytext=(target_month+5, total_customers*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add year markers
    for year in range(2, 7):
        month = (year - 1) * 12 + 1
        plt.axvline(x=month, color='gray', linestyle='--', alpha=0.3)
        plt.text(month, plt.ylim()[1]*0.95, f'Year {year}', ha='center', 
                va='top', backgroundcolor='white', alpha=0.8)
    
    plt.title(f'Customer Growth Required to Reach ${target_revenue/1000000:.2f}M Monthly Revenue by Month {target_month}', 
              fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Total Customers')
    plt.grid(True, alpha=0.3)
    
    # Plot monthly revenue growth
    plt.subplot(2, 1, 2)
    monthly_rev = revenue_monthly_data['total_arr'] / 12 / 1000000  # Convert to millions
    plt.plot(revenue_monthly_data['month_number'], monthly_rev, 
             linewidth=2, marker='o', markersize=4)
    
    # Highlight target month
    plt.axvline(x=target_month, color='red', linestyle='--', alpha=0.7)
    target_rev = monthly_rev.iloc[target_month-1]
    plt.scatter(target_month, target_rev, s=100, color='red', zorder=5)
    plt.annotate(f"Target Month {target_month}\n${target_rev:.2f}M Revenue", 
                xy=(target_month, target_rev), 
                xytext=(target_month+5, target_rev*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Add horizontal line at target revenue
    plt.axhline(y=target_revenue/1000000, color='green', linestyle='--', alpha=0.7)
    plt.text(1, target_revenue/1000000*1.05, f'Target Revenue: ${target_revenue/1000000:.2f}M', 
             color='green', backgroundcolor='white', alpha=0.8)
    
    # Add year markers
    for year in range(2, 7):
        month = (year - 1) * 12 + 1
        plt.axvline(x=month, color='gray', linestyle='--', alpha=0.3)
        plt.text(month, plt.ylim()[1]*0.95, f'Year {year}', ha='center', 
                va='top', backgroundcolor='white', alpha=0.8)
    
    plt.title(f'Monthly Revenue Growth to Reach Target', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Monthly Revenue ($M)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/optimization/specific_revenue_customer_growth.png', dpi=300, bbox_inches='tight')
    
    # Print optimization results
    print("\n" + "="*50)
    print(f"Optimization Results:")
    print(f"Target: ${target_revenue/1000000:.2f}M monthly revenue by month {target_month}")
    print(f"Achieved: ${achieved_monthly_revenue/1000000:.2f}M monthly revenue")
    print(f"Total customers required: {int(total_customers)}")
    print("\nCustomers by segment:")
    for segment, count in customers_by_segment.items():
        print(f"  - {segment}: {int(count)} customers")
        
    # Calculate customer acquisition timeline
    print("\nCustomer Acquisition Timeline (annual):")
    years = [12, 24, 36, 48, 60, target_month] if target_month not in [12, 24, 36, 48, 60] else [12, 24, 36, 48, 60]
    years = sorted(list(set([m for m in years if m <= target_month])))
    
    # Create a table header
    print(f"\n{'Month':>10} {'Total':>10} ", end="")
    for segment in revenue_config['segments']:
        print(f"{segment:>12} ", end="")
    print()
    
    # Print data for each year
    for month in years:
        month_idx = min(month - 1, len(revenue_monthly_data) - 1)
        year_total = revenue_monthly_data.loc[month_idx, 'total_customers']
        
        print(f"{month:>10} {int(year_total):>10} ", end="")
        
        for segment in revenue_config['segments']:
            segment_count = revenue_monthly_data.loc[month_idx, f'{segment}_customers'] \
                if f'{segment}_customers' in revenue_monthly_data.columns else 0
            print(f"{int(segment_count):>12} ", end="")
        print()
    
    # Calculate and print monthly acquisition rates required
    print("\nAverage Monthly Customer Acquisition Rate:")
    
    # Calculate monthly acquisition rates for the next year or up to target month
    start_month = 0  # First month
    end_month = target_month - 1  # Target month (0-indexed)
    
    total_net_growth = total_customers - revenue_monthly_data.loc[start_month, 'total_customers']
    months_to_grow = end_month - start_month
    
    if months_to_grow > 0:
        avg_monthly_growth = total_net_growth / months_to_grow
        print(f"Overall: {avg_monthly_growth:.1f} net new customers per month")
        
        # Calculate by segment
        for segment in revenue_config['segments']:
            if f'{segment}_customers' in revenue_monthly_data.columns:
                segment_start = revenue_monthly_data.loc[start_month, f'{segment}_customers']
                segment_end = revenue_monthly_data.loc[end_month, f'{segment}_customers']
                segment_growth = segment_end - segment_start
                segment_monthly = segment_growth / months_to_grow
                print(f"  - {segment}: {segment_monthly:.1f} net new customers per month")
    
    print("="*50)
    
    return financial_model, revenue_model, cost_model, optimization_results

if __name__ == "__main__":
    # Default values
    target_revenue = 30000000  # $30M monthly revenue
    target_month = 60  # Month 60 (end of year 5)
    initial_investment = 20000000  # $20M initial investment
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        target_revenue = float(sys.argv[1])
    if len(sys.argv) > 2:
        target_month = int(sys.argv[2])
    if len(sys.argv) > 3:
        initial_investment = float(sys.argv[3])
    
    # Run the optimization
    financial_model, revenue_model, cost_model, _ = optimize_for_specific_revenue(
        target_revenue=target_revenue,
        target_month=target_month,
        initial_investment=initial_investment
    )
    
    # Generate additional reports
    if financial_model:
        # Generate additional detailed plots
        fig1 = financial_model.plot_financial_summary(figsize=(14, 8))
        fig1.savefig('reports/optimization/specific_revenue_financial_summary.png', dpi=300, bbox_inches='tight')
        
        fig2 = revenue_model.plot_growth_curves(figsize=(14, 10))
        fig2.savefig('reports/optimization/specific_revenue_growth_curves.png', dpi=300, bbox_inches='tight')
        
        fig3 = financial_model.plot_runway_and_capital(figsize=(14, 8))
        fig3.savefig('reports/optimization/specific_revenue_runway.png', dpi=300, bbox_inches='tight')