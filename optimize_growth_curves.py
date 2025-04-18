import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from app import run_integrated_financial_model

def optimize_s_curves(metrics_to_optimize='valuation'):
    """
    Systematically optimize S-curve steepness parameters to maximize desired metrics.
    
    Parameters:
    -----------
    metrics_to_optimize : str
        'valuation', 'profitability', or 'overall' for combined optimization
        
    Returns:
    --------
    tuple: (best_params, results_df)
        The optimal parameters found and a dataframe with all test results
    """
    print(f"Starting S-curve steepness optimization for: {metrics_to_optimize}")
    
    # Run baseline model for comparison
    base_model, revenue_model, cost_model = run_integrated_financial_model(20000000)
    base_monthly = base_model.get_monthly_data()
    base_annual = base_model.get_annual_data()
    
    # Extract baseline metrics
    base_terminal_revenue = base_annual.iloc[-1]['annual_revenue']
    base_terminal_valuation = base_terminal_revenue * 8
    
    profitable_month_data = base_monthly[base_monthly['ebitda'] > 0]
    base_profitable_month = profitable_month_data['month_number'].min() if len(profitable_month_data) > 0 else 100
    
    base_min_capital = base_monthly['capital'].min()
    
    print(f"Baseline metrics:")
    print(f"- Terminal Valuation (8x Revenue): ${base_terminal_valuation/1000000:.2f}M")
    print(f"- Month of Profitability: {base_profitable_month}")
    print(f"- Minimum Capital Position: ${base_min_capital/1000000:.2f}M")
    
    # Define segments and steepness parameter ranges to test
    segments = ['Enterprise', 'Mid-Market', 'SMB']
    steepness_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    
    # Store results
    results = []
    
    # We'll test multiple configurations, starting with one segment at a time
    # For simplicity in this example, we'll only vary steepness for years 1-3
    # which have the biggest impact on financial outcomes
    
    # Test Enterprise segment variations
    print("\nTesting Enterprise segment variations...")
    for steepness_y1, steepness_y2, steepness_y3 in product(steepness_values[:4], steepness_values[1:5], steepness_values[2:]):
        # Skip combinations where later years have lower steepness
        if steepness_y2 < steepness_y1 or steepness_y3 < steepness_y2:
            continue
            
        print(f"Testing Enterprise: Y1={steepness_y1}, Y2={steepness_y2}, Y3={steepness_y3}")
        
        # Run model with these parameters
        financial_model, revenue_model, cost_model = run_integrated_financial_model(20000000)
        
        # Modify growth model's S-curve parameters for Enterprise
        steepness_config = {1: steepness_y1, 2: steepness_y2, 3: steepness_y3}
        tuned_growth_model = revenue_model.tune_s_curve_steepness('Enterprise', steepness_config)
        
        # Re-run the model
        tuned_growth_model.run_model()
        
        # Update financial model with new growth data
        financial_model.revenue_model = tuned_growth_model
        financial_model.run_model()
        
        # Extract metrics
        monthly_data = financial_model.get_monthly_data()
        annual_data = financial_model.get_annual_data()
        
        terminal_revenue = annual_data.iloc[-1]['annual_revenue']
        terminal_valuation = terminal_revenue * 8
        
        profitable_month_data = monthly_data[monthly_data['ebitda'] > 0]
        profitable_month = profitable_month_data['month_number'].min() if len(profitable_month_data) > 0 else 100
        
        min_capital = monthly_data['capital'].min()
        
        # Calculate score based on the metric we're optimizing
        if metrics_to_optimize == 'valuation':
            score = terminal_valuation
        elif metrics_to_optimize == 'profitability':
            score = -profitable_month  # Negative since we want to minimize months to profitability
        elif metrics_to_optimize == 'runway':
            score = min_capital
        else:  # overall score combining metrics
            score = (
                terminal_valuation / 1000000 +  # Valuation in millions
                -profitable_month / 10 +  # Months to profitability (scaled)
                min_capital / 1000000  # Min capital in millions
            )
        
        # Store result
        results.append({
            'segment': 'Enterprise',
            'steepness_y1': steepness_y1, 
            'steepness_y2': steepness_y2,
            'steepness_y3': steepness_y3,
            'terminal_revenue': terminal_revenue,
            'terminal_valuation': terminal_valuation,
            'profitable_month': profitable_month,
            'min_capital': min_capital,
            'score': score
        })
    
    # Test Mid-Market segment variations (fewer combinations for example)
    print("\nTesting Mid-Market segment variations...")
    for steepness_y1, steepness_y2 in product(steepness_values[1:5], steepness_values[2:6]):
        if steepness_y2 < steepness_y1:
            continue
            
        print(f"Testing Mid-Market: Y1={steepness_y1}, Y2={steepness_y2}")
        
        # Run model with these parameters
        financial_model, revenue_model, cost_model = run_integrated_financial_model(20000000)
        
        # Modify growth model's S-curve parameters
        steepness_config = {1: steepness_y1, 2: steepness_y2}
        tuned_growth_model = revenue_model.tune_s_curve_steepness('Mid-Market', steepness_config)
        
        # Re-run the model
        tuned_growth_model.run_model()
        
        # Update financial model with new growth data
        financial_model.revenue_model = tuned_growth_model
        financial_model.run_model()
        
        # Extract metrics
        monthly_data = financial_model.get_monthly_data()
        annual_data = financial_model.get_annual_data()
        
        terminal_revenue = annual_data.iloc[-1]['annual_revenue']
        terminal_valuation = terminal_revenue * 8
        
        profitable_month_data = monthly_data[monthly_data['ebitda'] > 0]
        profitable_month = profitable_month_data['month_number'].min() if len(profitable_month_data) > 0 else 100
        
        min_capital = monthly_data['capital'].min()
        
        # Calculate score
        if metrics_to_optimize == 'valuation':
            score = terminal_valuation
        elif metrics_to_optimize == 'profitability':
            score = -profitable_month
        elif metrics_to_optimize == 'runway':
            score = min_capital
        else:
            score = (
                terminal_valuation / 1000000 +
                -profitable_month / 10 +
                min_capital / 1000000
            )
        
        # Store result
        results.append({
            'segment': 'Mid-Market',
            'steepness_y1': steepness_y1, 
            'steepness_y2': steepness_y2,
            'steepness_y3': None,
            'terminal_revenue': terminal_revenue,
            'terminal_valuation': terminal_valuation,
            'profitable_month': profitable_month,
            'min_capital': min_capital,
            'score': score
        })
    
    # Convert to DataFrame and find best result
    results_df = pd.DataFrame(results)
    results_df['valuation_diff_pct'] = (results_df['terminal_valuation'] - base_terminal_valuation) / base_terminal_valuation * 100
    results_df['profitability_diff'] = base_profitable_month - results_df['profitable_month']
    results_df['runway_diff'] = results_df['min_capital'] - base_min_capital
    
    # Find best configurations
    best_valuation_row = results_df.loc[results_df['terminal_valuation'].idxmax()]
    best_profitability_row = results_df.loc[results_df['profitable_month'].idxmin()]
    best_runway_row = results_df.loc[results_df['min_capital'].idxmax()]
    best_overall_row = results_df.loc[results_df['score'].idxmax()]
    
    # Print best results
    print("\n==== OPTIMIZATION RESULTS ====")
    
    print("\nBest for Valuation:")
    print(f"Segment: {best_valuation_row['segment']}")
    print(f"Y1 Steepness: {best_valuation_row['steepness_y1']}")
    print(f"Y2 Steepness: {best_valuation_row['steepness_y2']}")
    if not pd.isna(best_valuation_row['steepness_y3']):
        print(f"Y3 Steepness: {best_valuation_row['steepness_y3']}")
    print(f"Terminal Valuation: ${best_valuation_row['terminal_valuation']/1000000:.2f}M")
    print(f"Improvement: +{best_valuation_row['valuation_diff_pct']:.2f}%")
    
    print("\nBest for Profitability:")
    print(f"Segment: {best_profitability_row['segment']}")
    print(f"Y1 Steepness: {best_profitability_row['steepness_y1']}")
    print(f"Y2 Steepness: {best_profitability_row['steepness_y2']}")
    if not pd.isna(best_profitability_row['steepness_y3']):
        print(f"Y3 Steepness: {best_profitability_row['steepness_y3']}")
    print(f"Month of Profitability: {best_profitability_row['profitable_month']}")
    print(f"Improvement: {best_profitability_row['profitability_diff']} months earlier")
    
    print("\nBest for Runway/Capital:")
    print(f"Segment: {best_runway_row['segment']}")
    print(f"Y1 Steepness: {best_runway_row['steepness_y1']}")
    print(f"Y2 Steepness: {best_runway_row['steepness_y2']}")
    if not pd.isna(best_runway_row['steepness_y3']):
        print(f"Y3 Steepness: {best_runway_row['steepness_y3']}")
    print(f"Minimum Capital: ${best_runway_row['min_capital']/1000000:.2f}M")
    print(f"Improvement: ${best_runway_row['runway_diff']/1000000:.2f}M more runway")
    
    # Create visualizations comparing the best results
    create_comparison_charts(best_valuation_row, best_profitability_row, best_runway_row, base_model)
    
    return best_overall_row, results_df

def create_comparison_charts(valuation_config, profitability_config, runway_config, base_model):
    """Create charts comparing the different optimized configurations"""
    # Create visualizations of the optimal scenarios
    financial_model, revenue_model, cost_model = run_integrated_financial_model(20000000)
    
    # Create scenarios dictionary
    scenarios = {
        'Baseline': base_model
    }
    
    # For valuation optimization
    if valuation_config['segment'] == 'Enterprise':
        steepness_config = {1: valuation_config['steepness_y1'], 
                            2: valuation_config['steepness_y2'],
                            3: valuation_config['steepness_y3']}
    else:
        steepness_config = {1: valuation_config['steepness_y1'], 
                            2: valuation_config['steepness_y2']}
    
    tuned_model = revenue_model.tune_s_curve_steepness(valuation_config['segment'], steepness_config)
    tuned_model.run_model()
    fin_model_valuation, _, _ = run_integrated_financial_model(20000000)
    fin_model_valuation.revenue_model = tuned_model
    fin_model_valuation.run_model()
    scenarios['Optimized for Valuation'] = fin_model_valuation
    
    # For profitability optimization
    if profitability_config['segment'] == 'Enterprise':
        steepness_config = {1: profitability_config['steepness_y1'], 
                            2: profitability_config['steepness_y2'],
                            3: profitability_config['steepness_y3']}
    else:
        steepness_config = {1: profitability_config['steepness_y1'], 
                            2: profitability_config['steepness_y2']}
    
    tuned_model = revenue_model.tune_s_curve_steepness(profitability_config['segment'], steepness_config)
    tuned_model.run_model()
    fin_model_profit, _, _ = run_integrated_financial_model(20000000)
    fin_model_profit.revenue_model = tuned_model
    fin_model_profit.run_model()
    scenarios['Optimized for Profitability'] = fin_model_profit
    
    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(15, 18))
    
    # 1. Revenue curves
    ax = axs[0]
    for name, model in scenarios.items():
        data = model.get_monthly_data()
        ax.plot(
            data['month_number'],
            data['monthly_revenue'],
            label=name,
            linewidth=2
        )
    
    ax.set_title('Monthly Revenue - Optimization Comparison', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Monthly Revenue ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. EBITDA
    ax = axs[1]
    for name, model in scenarios.items():
        data = model.get_monthly_data()
        ax.plot(
            data['month_number'],
            data['ebitda'],
            label=name,
            linewidth=2
        )
    
    # Add horizontal line at 0 for break-even
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    ax.set_title('Monthly EBITDA - Optimization Comparison', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('EBITDA ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Capital/Runway
    ax = axs[2]
    for name, model in scenarios.items():
        data = model.get_monthly_data()
        ax.plot(
            data['month_number'],
            data['capital'],
            label=name,
            linewidth=2
        )
    
    ax.set_title('Capital Runway - Optimization Comparison', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Capital ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/combined/s_curve_optimization_comparison.png', dpi=300)
    
    # Create table comparing key metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Terminal Revenue ($M)', 'Valuation ($M)', 'Month of Profitability', 'Lowest Capital ($M)'],
        'Baseline': [
            base_model.get_annual_data().iloc[-1]['annual_revenue']/1000000,
            base_model.get_annual_data().iloc[-1]['annual_revenue']*8/1000000,
            base_model.get_monthly_data()[base_model.get_monthly_data()['ebitda'] > 0]['month_number'].min(),
            base_model.get_monthly_data()['capital'].min()/1000000
        ],
        'Optimized for Valuation': [
            fin_model_valuation.get_annual_data().iloc[-1]['annual_revenue']/1000000,
            fin_model_valuation.get_annual_data().iloc[-1]['annual_revenue']*8/1000000,
            fin_model_valuation.get_monthly_data()[fin_model_valuation.get_monthly_data()['ebitda'] > 0]['month_number'].min(),
            fin_model_valuation.get_monthly_data()['capital'].min()/1000000
        ],
        'Optimized for Profitability': [
            fin_model_profit.get_annual_data().iloc[-1]['annual_revenue']/1000000,
            fin_model_profit.get_annual_data().iloc[-1]['annual_revenue']*8/1000000,
            fin_model_profit.get_monthly_data()[fin_model_profit.get_monthly_data()['ebitda'] > 0]['month_number'].min(),
            fin_model_profit.get_monthly_data()['capital'].min()/1000000
        ]
    })
    
    # Save metrics table
    metrics_df.to_csv('reports/combined/s_curve_optimization_metrics.csv', index=False)
    
    return fig

if __name__ == "__main__":
    # Run optimization
    best_params, results = optimize_s_curves('overall')
    
    # Save results
    results.to_csv('reports/combined/s_curve_optimization_results.csv', index=False)
    
    print("\nOptimization complete! Results saved to reports/combined/")