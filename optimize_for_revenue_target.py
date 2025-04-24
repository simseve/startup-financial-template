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

def optimize_for_revenue_target(target_revenue=5000000, target_month=36, growth_profile="baseline", initial_investment=20000000):
    """
    Optimize the financial model to achieve a specific revenue target by a given month.
    Calculates the required customer acquisition growth needed to reach the target.
    
    Parameters:
    -----------
    target_revenue : float
        Target monthly revenue to achieve (not ARR)
    target_month : int
        Target month to achieve the revenue (1-indexed)
    growth_profile : str
        Base growth profile to start optimization from ('baseline', 'conservative', 'aggressive', 'hypergrowth')
    initial_investment : float
        Initial investment amount
    
    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
        The three optimized model objects and a dictionary of optimization results
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
    
    # Step 3: Apply the selected growth profile to create a starting point
    if growth_profile != "custom":
        revenue_model = base_revenue_model.apply_growth_profile(growth_profile)
    else:
        revenue_model = base_revenue_model
    
    # Step 4: Perform optimization to find the growth multiplier needed to reach the target
    print("Starting optimization...")
    
    # First, test the baseline model to see how far we are from the target
    revenue_model.run_model()
    baseline_monthly_data = revenue_model.get_monthly_data()
    
    if target_month > len(baseline_monthly_data):
        print(f"Error: Target month {target_month} exceeds the projection period of {len(baseline_monthly_data)} months")
        return None, None, None, None
    
    baseline_monthly_revenue = baseline_monthly_data.loc[target_month-1, 'total_arr'] / 12
    print(f"Baseline model: ${baseline_monthly_revenue/1000000:.2f}M monthly revenue in month {target_month}")
    print(f"Target revenue: ${target_revenue/1000000:.2f}M")
    
    # Calculate initial growth factor needed
    initial_factor = target_revenue / baseline_monthly_revenue if baseline_monthly_revenue > 0 else 5.0
    print(f"Initial growth factor estimate: {initial_factor:.2f}x")
    
    # Use a more sophisticated optimization approach
    multipliers_to_try = []
    
    # Use a wider range if the estimated factor is very high
    if initial_factor > 10:
        max_factor = min(50.0, initial_factor * 1.5)  # Cap at 50x for practical reasons
        multipliers_to_try = np.linspace(initial_factor * 0.5, max_factor, 10)
    else:
        multipliers_to_try = np.linspace(0.5, max(initial_factor * 1.5, 15.0), 10)
    
    # Add some specific multipliers based on common growth scenarios
    multipliers_to_try = sorted(list(set(np.concatenate([
        multipliers_to_try, 
        [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    ]))))
    
    # Try a range of multipliers and find the one that gets closest to the target
    best_model = None
    best_distance = float('inf')
    best_multiplier = None
    best_achieved_revenue = 0
    
    print("Testing different growth multipliers...")
    
    for multiplier in multipliers_to_try:
        # Create a test model with the current multiplier
        test_model = revenue_model.apply_custom_segment_profiles({
            segment: multiplier for segment in revenue_config['segments']
        })
        
        # Run the model
        test_model.run_model()
        test_monthly_data = test_model.get_monthly_data()
        
        # Calculate monthly revenue at target month (ARR/12)
        achieved_monthly_revenue = test_monthly_data.loc[target_month-1, 'total_arr'] / 12
        distance = abs(achieved_monthly_revenue - target_revenue)
        relative_error = abs(achieved_monthly_revenue - target_revenue) / target_revenue
        
        print(f"  Testing growth multiplier {multiplier:.2f}x: Achieved ${achieved_monthly_revenue/1000000:.2f}M " +
              f"(Error: {relative_error*100:.1f}%)")
        
        # Track the best model so far
        if distance < best_distance:
            best_distance = distance
            best_model = test_model
            best_multiplier = multiplier
            best_achieved_revenue = achieved_monthly_revenue
    
    # Tolerance for acceptable solution
    tolerance = 0.05  # Within 5% of target revenue is acceptable
    target_achieved = abs(best_achieved_revenue - target_revenue) / target_revenue <= tolerance
    
    if not target_achieved:
        print(f"\nNotice: Best multiplier {best_multiplier:.2f}x achieves ${best_achieved_revenue/1000000:.2f}M " +
              f"(Target: ${target_revenue/1000000:.2f}M, Error: {(abs(best_achieved_revenue - target_revenue) / target_revenue)*100:.1f}%)")
        
        # If we're far below target, indicate that a more aggressive approach might be needed
        if best_achieved_revenue < target_revenue * 0.8:
            print(f"The revenue target might require more specific optimizations beyond a simple growth multiplier.")
            print(f"Consider modifying the S-curve parameters or segment-specific strategies.")
    else:
        print(f"\n✓ Target achieved within tolerance of {tolerance*100:.1f}%")
        print(f"  Multiplier: {best_multiplier:.2f}x achieves ${best_achieved_revenue/1000000:.2f}M monthly revenue")
    
    # If we didn't find a model within tolerance, use the best one we found
    if not best_model:
        print("Warning: Could not achieve target revenue within tolerance")
        return None, None, None, None
    
    # Step 5: Run the cost model with the optimized revenue model
    revenue_model = best_model
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
    
    # Check if target was achieved
    target_month_idx = min(target_month - 1, len(monthly_data) - 1)
    achieved_monthly_revenue = monthly_data.loc[target_month_idx, 'total_arr'] / 12
    target_achieved = abs(achieved_monthly_revenue - target_revenue) / target_revenue <= tolerance
    
    # Get customer data by segment - directly from the revenue model which has the segment data
    revenue_monthly_data = revenue_model.get_monthly_data()
    customers_by_segment = {}
    total_customers = monthly_data.loc[target_month_idx, 'total_customers']
    
    for segment in revenue_config['segments']:
        segment_customers = revenue_monthly_data.loc[target_month_idx, f'{segment}_customers'] \
            if f'{segment}_customers' in revenue_monthly_data.columns else 0
        customers_by_segment[segment] = segment_customers
    
    # Store optimization results
    optimization_results = {
        "target": f"${target_revenue/1000000:.2f}M monthly revenue",
        "target_month": target_month,
        "achieved": target_achieved,
        "achieved_revenue": achieved_monthly_revenue,
        "growth_multiplier": best_multiplier,
        "total_customers_required": total_customers,
        "customers_by_segment": customers_by_segment
    }
    
    # Create directory for output
    os.makedirs('reports/optimization', exist_ok=True)
    
    # Save optimization results
    pd.DataFrame([optimization_results]).to_csv(
        'reports/optimization/revenue_target_results.csv', index=False
    )
    
    # Generate visualization of required customer growth
    plt.figure(figsize=(12, 8))
    
    # Plot total customers over time
    plt.subplot(2, 1, 1)
    plt.plot(monthly_data['month_number'], monthly_data['total_customers'], 
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
    
    plt.title(f'Customer Growth Required to Reach ${target_revenue/1000000:.2f}M Monthly Revenue by Month {target_month}', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Total Customers')
    plt.grid(True, alpha=0.3)
    
    # Plot monthly revenue growth
    plt.subplot(2, 1, 2)
    monthly_rev = monthly_data['total_arr'] / 12 / 1000000  # Convert to millions
    plt.plot(monthly_data['month_number'], monthly_rev, 
             linewidth=2, marker='o', markersize=4)
    
    # Highlight target month
    plt.axvline(x=target_month, color='red', linestyle='--', alpha=0.7)
    target_rev = monthly_rev.iloc[target_month_idx]
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
    plt.savefig('reports/optimization/revenue_target_customer_growth.png', dpi=300, bbox_inches='tight')
    
    # Print optimization results
    print("\n" + "="*50)
    print(f"Optimization Results:")
    print(f"Target: ${target_revenue/1000000:.2f}M monthly revenue by month {target_month}")
    print(f"Achieved: ${achieved_monthly_revenue/1000000:.2f}M monthly revenue")
    print(f"Growth multiplier applied: {best_multiplier:.2f}x")
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

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        target_revenue = float(sys.argv[1])
    else:
        target_revenue = 5000000  # Default: $5M monthly revenue
    
    if len(sys.argv) > 2:
        target_month = int(sys.argv[2])
    else:
        target_month = 36  # Default: Month 36 (Year 3)
    
    if len(sys.argv) > 3:
        growth_profile = sys.argv[3]
    else:
        growth_profile = "baseline"  # Default growth profile
    
    if len(sys.argv) > 4:
        initial_investment = float(sys.argv[4])
    else:
        initial_investment = 20000000  # Default: $20M investment
    
    # Run optimization
    financial_model, revenue_model, cost_model, optimization_results = optimize_for_revenue_target(
        target_revenue=target_revenue,
        target_month=target_month,
        growth_profile=growth_profile,
        initial_investment=initial_investment
    )
    
    # If optimization was successful, generate additional visualizations
    if financial_model:
        # Generate additional detailed plots
        fig1 = financial_model.plot_financial_summary(figsize=(14, 8))
        fig1.savefig('reports/optimization/revenue_target_financial_summary.png', dpi=300, bbox_inches='tight')
        
        fig2 = revenue_model.plot_growth_curves(figsize=(14, 10))
        fig2.savefig('reports/optimization/revenue_target_growth_curves.png', dpi=300, bbox_inches='tight')
        
        fig3 = financial_model.plot_runway_and_capital(figsize=(14, 8))
        fig3.savefig('reports/optimization/revenue_target_runway.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()