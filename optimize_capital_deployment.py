import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from app import run_integrated_financial_model


def optimize_aggressive_capital_deployment():
    """
    Optimize for aggressive but efficient capital deployment to maximize growth.
    Tests various spending configurations to accelerate growth while maintaining sustainability.
    """
    print("Optimizing for aggressive capital deployment strategy...")

    # Run baseline model with original parameters
    base_model, revenue_model, cost_model = run_integrated_financial_model(
        20000000)
    base_monthly = base_model.get_monthly_data()
    base_annual = base_model.get_annual_data()

    # Extract baseline metrics
    base_terminal_revenue = base_annual.iloc[-1]['annual_revenue']
    base_terminal_valuation = base_terminal_revenue * 8

    # Calculate baseline capital utilization metrics
    profitable_month_data = base_monthly[base_monthly['ebitda'] > 0]
    base_profitable_month = profitable_month_data['month_number'].min() if len(
        profitable_month_data) > 0 else 100

    # Calculate burn rate before profitability
    pre_profit_data = base_monthly[base_monthly['month_number']
                                   < base_profitable_month]
    negative_cash_flows = pre_profit_data[pre_profit_data['cash_flow']
                                          < 0]['cash_flow']
    base_total_burn = -negative_cash_flows.sum()

    base_min_capital = base_monthly['capital'].min()
    base_capital_efficiency = base_terminal_revenue / \
        base_total_burn if base_total_burn > 0 else float('inf')

    print(f"Baseline metrics:")
    print(f"- Terminal Revenue: ${base_terminal_revenue/1000000:.2f}M")
    print(
        f"- Terminal Valuation (8x): ${base_terminal_valuation/1000000:.2f}M")
    print(f"- Month of Profitability: {base_profitable_month}")
    print(
        f"- Total Burn Before Profitability: ${base_total_burn/1000000:.2f}M")
    print(f"- Minimum Capital Position: ${base_min_capital/1000000:.2f}M")
    print(
        f"- Capital Efficiency (Revenue/Burn): {base_capital_efficiency:.2f}x")

    # Define parameter ranges to test for aggressive spending
    cogs_ranges = {
        # Higher cloud compute for better AI performance
        'cloud_hosting': [0.15, 0.18, 0.20, 0.22, 0.25],
        'customer_support': [0.06, 0.08, 0.10, 0.12],     # More support staff
        # More third-party integrations
        'third_party_apis': [0.04, 0.06, 0.08, 0.10],
        # More implementation services
        'professional_services': [0.02, 0.03, 0.04, 0.05]
    }

    compensation_ranges = {
        'benefits_multiplier': [1.20, 1.25, 1.28],        # Better benefits
        'bonus_rate': [0.10, 0.12, 0.15],                 # Competitive bonuses
        # Strong equity incentives
        'equity_compensation': [0.15, 0.18, 0.20]
    }

    # Also vary growth curve steepness for Enterprise segment
    enterprise_steepness = {
        1: [0.6, 0.7, 0.8],  # Year 1 steepness
        2: [0.7, 0.8, 0.9],  # Year 2 steepness
        3: [0.8, 0.9, 1.0]   # Year 3 steepness
    }

    # Store results from all tested configurations
    results = []

    # Target for ideal capital burn - we want to use 60-80% of capital before profitability
    # This ensures we're investing aggressively but not running too close to zero
    target_capital_utilization = 0.7  # Aim to use 70% of capital

    # Test various combinations
    scenario_count = 0
    max_scenarios = 80  # Limit total scenarios to keep runtime reasonable

    # Generate spending combinations - use a more targeted approach than testing all combinations
    for cloud in cogs_ranges['cloud_hosting']:
        for support in [cogs_ranges['customer_support'][1], cogs_ranges['customer_support'][2]]:  # Middle values
            for apis in [cogs_ranges['third_party_apis'][1], cogs_ranges['third_party_apis'][2]]:  # Middle values
                for ps in [cogs_ranges['professional_services'][1]]:  # Just one value for PS

                    # Only test a few compensation combinations
                    # Middle value
                    for benefits in [compensation_ranges['benefits_multiplier'][1]]:
                        for bonus in [compensation_ranges['bonus_rate'][1]]:  # Middle value
                            # Middle value
                            for equity in [compensation_ranges['equity_compensation'][1]]:

                                # Test different growth curve steepness values
                                for steepness_y1 in enterprise_steepness[1]:
                                    for steepness_y2 in enterprise_steepness[2]:
                                        if steepness_y2 < steepness_y1:
                                            continue  # Skip invalid combinations

                                        for steepness_y3 in enterprise_steepness[3]:
                                            if steepness_y3 < steepness_y2:
                                                continue  # Skip invalid combinations

                                            # Limit the total scenarios we test
                                            scenario_count += 1
                                            if scenario_count > max_scenarios:
                                                break

                                            print(f"Testing scenario {scenario_count}: Cloud={cloud}, Support={support}, APIs={apis}, PS={ps}, "
                                                  f"Growth Y1={steepness_y1}, Y2={steepness_y2}, Y3={steepness_y3}")

                                            # Run model with these parameters
                                            financial_model, revenue_model, cost_model = run_integrated_financial_model(
                                                20000000)

                                            # Update cost model parameters
                                            cost_model.config['cogs']['cloud_hosting'] = cloud
                                            cost_model.config['cogs']['customer_support'] = support
                                            cost_model.config['cogs']['third_party_apis'] = apis
                                            cost_model.config['cogs']['professional_services'] = ps

                                            cost_model.config['salary']['benefits_multiplier'] = benefits
                                            cost_model.config['salary']['bonus_rate'] = bonus
                                            cost_model.config['salary']['equity_compensation'] = equity

                                            # Update growth curve
                                            steepness_config = {
                                                1: steepness_y1, 2: steepness_y2, 3: steepness_y3}
                                            tuned_growth_model = revenue_model.tune_s_curve_steepness(
                                                'Enterprise', steepness_config)
                                            tuned_growth_model.run_model()

                                            # Re-run the full model with updated cost and revenue models
                                            financial_model.revenue_model = tuned_growth_model
                                            cost_model.run_model(
                                                tuned_growth_model.monthly_data)
                                            financial_model.run_model()

                                            # Extract metrics
                                            monthly_data = financial_model.get_monthly_data()
                                            annual_data = financial_model.get_annual_data()

                                            # Calculate key metrics
                                            terminal_revenue = annual_data.iloc[-1]['annual_revenue']
                                            terminal_valuation = terminal_revenue * 8

                                            profitable_month_data = monthly_data[monthly_data['ebitda'] > 0]
                                            profitable_month = profitable_month_data['month_number'].min() if len(
                                                profitable_month_data) > 0 else 100

                                            # Calculate burn rate before profitability
                                            pre_profit_data = monthly_data[monthly_data['month_number']
                                                                           < profitable_month]
                                            negative_cash_flows = pre_profit_data[
                                                pre_profit_data['cash_flow'] < 0]['cash_flow']
                                            total_burn = -negative_cash_flows.sum()

                                            min_capital = monthly_data['capital'].min(
                                            )
                                            # What percentage of initial investment was used
                                            capital_utilization = total_burn / 20000000
                                            capital_efficiency = terminal_revenue / \
                                                total_burn if total_burn > 0 else float(
                                                    'inf')

                                            # Calculate a score that favors:
                                            # 1. Higher terminal valuation
                                            # 2. Capital utilization close to target (not too conservative, not too risky)
                                            # 3. Earlier profitability (but this is less important)

                                            capital_utilization_penalty = abs(
                                                capital_utilization - target_capital_utilization) * 100
                                            valuation_factor = terminal_valuation / 1000000  # In millions
                                            # Earlier is better, scaled
                                            profitability_factor = max(
                                                0, 72 - profitable_month) / 10

                                            score = valuation_factor - capital_utilization_penalty + profitability_factor

                                            results.append({
                                                'cloud_hosting': cloud,
                                                'customer_support': support,
                                                'third_party_apis': apis,
                                                'professional_services': ps,
                                                'benefits_multiplier': benefits,
                                                'bonus_rate': bonus,
                                                'equity_compensation': equity,
                                                'enterprise_steepness_y1': steepness_y1,
                                                'enterprise_steepness_y2': steepness_y2,
                                                'enterprise_steepness_y3': steepness_y3,
                                                'terminal_revenue': terminal_revenue,
                                                'terminal_valuation': terminal_valuation,
                                                'profitable_month': profitable_month,
                                                'total_burn': total_burn,
                                                'min_capital': min_capital,
                                                'capital_utilization': capital_utilization,
                                                'capital_efficiency': capital_efficiency,
                                                'score': score
                                            })

                                        if scenario_count > max_scenarios:
                                            break
                                    if scenario_count > max_scenarios:
                                        break
                                if scenario_count > max_scenarios:
                                    break

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find best result
    best_row = results_df.loc[results_df['score'].idxmax()]

    print("\n==== OPTIMIZATION RESULTS ====")
    print("\nBest aggressive growth configuration:")
    print(f"COGS Parameters:")
    print(f"- Cloud Hosting: {best_row['cloud_hosting']:.2%}")
    print(f"- Customer Support: {best_row['customer_support']:.2%}")
    print(f"- Third-Party APIs: {best_row['third_party_apis']:.2%}")
    print(f"- Professional Services: {best_row['professional_services']:.2%}")
    print(f"Compensation Parameters:")
    print(f"- Benefits Multiplier: {best_row['benefits_multiplier']:.2f}x")
    print(f"- Bonus Rate: {best_row['bonus_rate']:.2%}")
    print(f"- Equity Compensation: {best_row['equity_compensation']:.2%}")
    print(f"Enterprise Growth Curve Steepness:")
    print(f"- Year 1: {best_row['enterprise_steepness_y1']:.2f}")
    print(f"- Year 2: {best_row['enterprise_steepness_y2']:.2f}")
    print(f"- Year 3: {best_row['enterprise_steepness_y3']:.2f}")
    print(f"\nPerformance Metrics:")
    print(f"- Terminal Revenue: ${best_row['terminal_revenue']/1000000:.2f}M")
    print(
        f"- Terminal Valuation (8x): ${best_row['terminal_valuation']/1000000:.2f}M")
    print(f"- Month of Profitability: {best_row['profitable_month']}")
    print(
        f"- Total Burn Before Profitability: ${best_row['total_burn']/1000000:.2f}M")
    print(f"- Capital Utilization: {best_row['capital_utilization']:.2%}")
    print(
        f"- Capital Efficiency (Revenue/Burn): {best_row['capital_efficiency']:.2f}x")

    # Create a comparison chart showing baseline vs optimized spending pattern
    financial_model, revenue_model, cost_model = run_integrated_financial_model(
        20000000)

    # Configure the model with the optimal parameters
    cost_model.config['cogs']['cloud_hosting'] = best_row['cloud_hosting']
    cost_model.config['cogs']['customer_support'] = best_row['customer_support']
    cost_model.config['cogs']['third_party_apis'] = best_row['third_party_apis']
    cost_model.config['cogs']['professional_services'] = best_row['professional_services']

    cost_model.config['salary']['benefits_multiplier'] = best_row['benefits_multiplier']
    cost_model.config['salary']['bonus_rate'] = best_row['bonus_rate']
    cost_model.config['salary']['equity_compensation'] = best_row['equity_compensation']

    # Update growth curve
    steepness_config = {
        1: best_row['enterprise_steepness_y1'],
        2: best_row['enterprise_steepness_y2'],
        3: best_row['enterprise_steepness_y3']
    }
    tuned_growth_model = revenue_model.tune_s_curve_steepness(
        'Enterprise', steepness_config)
    tuned_growth_model.run_model()

    # Re-run the model
    financial_model.revenue_model = tuned_growth_model
    cost_model.run_model(tuned_growth_model.monthly_data)
    financial_model.run_model()

    # Create comparison charts between baseline and optimized models
    create_aggressive_vs_baseline_charts(financial_model, base_model)

    # Save the results
    results_df.to_csv(
        'reports/aggressive_capital_deployment_results.csv', index=False)

    return best_row, results_df


def create_aggressive_vs_baseline_charts(optimized_model, base_model):
    """Create charts comparing baseline vs optimized aggressive spending"""
    fig, axs = plt.subplots(4, 1, figsize=(15, 20))

    # Scenarios for comparison
    scenarios = {
        'Baseline': base_model,
        'Optimized Aggressive Growth': optimized_model
    }

    # 1. Monthly Burn Rate (negative cash flow)
    ax = axs[0]
    for name, model in scenarios.items():
        data = model.get_monthly_data()
        # Only plot negative cash flow as burn
        burn_data = data.copy()
        burn_data.loc[burn_data['cash_flow'] > 0, 'cash_flow'] = 0
        ax.plot(
            burn_data['month_number'],
            -burn_data['cash_flow'],  # Make positive for visualization
            label=name,
            linewidth=2
        )

    ax.set_title('Monthly Burn Rate', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Burn Rate ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cumulative Burn
    ax = axs[1]
    for name, model in scenarios.items():
        data = model.get_monthly_data()
        # Calculate cumulative burn
        burn_data = data.copy()
        burn_data.loc[burn_data['cash_flow'] > 0,
                      'cash_flow'] = 0  # Only count negative flows
        burn_data['cumulative_burn'] = -burn_data['cash_flow'].cumsum()
        ax.plot(
            burn_data['month_number'],
            burn_data['cumulative_burn'],
            label=name,
            linewidth=2
        )

    ax.set_title('Cumulative Capital Deployed', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative Spend ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Monthly Revenue
    ax = axs[2]
    for name, model in scenarios.items():
        data = model.get_monthly_data()
        ax.plot(
            data['month_number'],
            data['monthly_revenue'],
            label=name,
            linewidth=2
        )

    ax.set_title('Monthly Revenue', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Monthly Revenue ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Capital Position
    ax = axs[3]
    for name, model in scenarios.items():
        data = model.get_monthly_data()
        ax.plot(
            data['month_number'],
            data['capital'],
            label=name,
            linewidth=2
        )

    ax.set_title('Remaining Capital', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('Capital ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/aggressive_capital_deployment_comparison.png', dpi=300)

    return fig


if __name__ == "__main__":
    best_config, results = optimize_aggressive_capital_deployment()
    print("\nOptimization complete! Results saved to reports/")
