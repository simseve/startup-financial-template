import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from IPython.display import display
import argparse
import sys
import json
import os

# Import our model classes
from models.cost_model import AISaaSCostModel
from models.growth_model import SaaSGrowthModel
from models.financial_model import SaaSFinancialModel

# Load configuration from JSON files


def load_config(file_path):
    """Load a configuration from a JSON file"""
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)

        # Convert string keys to integers for dictionaries that need integer keys
        # This is needed because JSON can only have string keys
        for key in ['s_curve', 'seasonality', 'marketing_efficiency']:
            if key in config:
                if isinstance(config[key], dict):
                    # Handle s_curve which is a nested structure
                    if key == 's_curve':
                        for segment, years in config[key].items():
                            config[key][segment] = {
                                int(year): values for year, values in years.items()}
                    else:
                        # Handle flat dictionaries like seasonality and marketing_efficiency
                        config[key] = {
                            int(k): v for k, v in config[key].items()}

        # Handle headcount growth_factors
        if 'headcount' in config:
            for dept in config['headcount']:
                if 'growth_factors' in config['headcount'][dept]:
                    config['headcount'][dept]['growth_factors'] = {
                        int(k): v for k, v in config['headcount'][dept]['growth_factors'].items()
                    }

        return config
    except Exception as e:
        print(f"Error loading configuration from {file_path}: {e}")
        return None

# Configure and run the integrated model


def run_integrated_financial_model(
    initial_investment=20000000,
    growth_profile="baseline",
    segment_multipliers=None,
    optimize_target=None,
    target_month=24,
    revenue_config_path="configs/revenue_config.json",
    cost_config_path="configs/cost_config.json"
):
    """
    Run the integrated financial model for an AI SaaS business

    Parameters:
    -----------
    initial_investment : float
        Initial investment amount
    growth_profile : str
        Growth profile to use ('baseline', 'conservative', 'aggressive', 'hypergrowth', or 'custom')
    segment_multipliers : dict, optional
        Custom multipliers for each segment, if growth_profile="custom"
    optimize_target : str, optional
        Target to optimize for ('breakeven' or 'series_b')
    target_month : int, optional
        Target month for optimization goal
    revenue_config_path : str
        Path to the revenue configuration JSON file
    cost_config_path : str
        Path to the cost configuration JSON file

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    # Step 1: Load Revenue Configuration
    revenue_config = load_config(revenue_config_path)
    if not revenue_config:
        print(
            f"Failed to load revenue configuration from {revenue_config_path}. Using default configuration.")
        revenue_config = {
            # Basic parameters
            'start_date': '2025-01-01',
            'projection_months': 72,  # 6 years
            'segments': ['Enterprise', 'Mid-Market', 'SMB'],
            # ... default configuration would go here ...
        }

    # Step 2: Load Cost Configuration
    cost_config = load_config(cost_config_path)
    if not cost_config:
        print(
            f"Failed to load cost configuration from {cost_config_path}. Using default configuration.")
        cost_config = {
            # Basic parameters
            'start_date': '2025-01-01',
            'projection_months': 72,  # 6 years
            # ... default configuration would go here ...
        }

    # Step 3: Create base models
    revenue_model = SaaSGrowthModel(revenue_config)
    cost_model = AISaaSCostModel(cost_config)

    # Apply growth profile or perform optimization
    optimization_results = None

    # First, apply the selected growth profile if not optimizing
    if growth_profile != "custom" and optimize_target is None:
        # Use the built-in growth profiles
        revenue_model = revenue_model.apply_growth_profile(growth_profile)
    elif growth_profile == "custom" and segment_multipliers and optimize_target is None:
        # Apply custom segment multipliers
        revenue_model = revenue_model.apply_custom_segment_profiles(
            segment_multipliers)
    elif optimize_target is not None:
        # Prepare fixed costs for optimization
        # Extract average monthly fixed costs from the cost model
        monthly_fixed_costs = {
            'salary': cost_config['headcount']['engineering']['starting_count'] * cost_config['headcount']['engineering']['avg_salary'] / 12,
            # Initial marketing budget
            'marketing': cost_config['marketing_expenses']['paid_advertising'] * 100000 / 12,
            'facilities': 20000,  # Monthly facilities cost
            'other': 30000  # Other monthly fixed costs
        }

        # Set variable cost percentage (approx. 35% of revenue) for optimization
        variable_cost_pct = 0.35

        # Run optimization based on target
        if optimize_target == "breakeven":
            best_model, best_multiplier, achieved_month, metrics = revenue_model.optimize_for_breakeven(
                SaaSFinancialModel, monthly_fixed_costs, variable_cost_pct,
                target_month, min_multiplier=0.4, max_multiplier=3.0
            )
            revenue_model = best_model
            optimization_results = {
                "target": "breakeven",
                "target_month": target_month,
                "achieved_month": achieved_month,
                "growth_multiplier": best_multiplier
            }

        elif optimize_target == "series_b":
            # Series B typically requires $10M+ ARR with 100%+ YoY growth
            best_model, best_multiplier, achieved_month, metrics = revenue_model.optimize_for_series_b(
                SaaSFinancialModel, monthly_fixed_costs, variable_cost_pct,
                target_month, target_arr=10000000, target_growth_rate=1.0
            )
            revenue_model = best_model
            optimization_results = {
                "target": "series_b",
                "target_month": target_month,
                "achieved_month": achieved_month,
                "growth_multiplier": best_multiplier
            }

    # Step 4: Run the models
    revenue_model.run_model()

    # Pass revenue projections to the cost model
    cost_model.run_model(revenue_model)

    # Step 5: Create and run the financial model
    financial_model = SaaSFinancialModel(revenue_model, cost_model)
    financial_model.run_model()

    # Save optimization results if applicable
    if optimization_results:
        os.makedirs('reports/optimization', exist_ok=True)
        pd.DataFrame([optimization_results]).to_csv(
            'reports/optimization/optimization_results.csv', index=False)

    return financial_model, revenue_model, cost_model, optimization_results

# Execute the model


def find_breakeven_month(financial_model):
    """Find the month when the company breaks even"""
    monthly_data = financial_model.get_monthly_data()

    # Try using profit column if available
    if 'profit' in monthly_data.columns:
        for i, profit in enumerate(monthly_data['profit']):
            if profit >= 0:
                return i + 1  # Convert to 1-indexed
    # Fall back to ebitda column as proxy for profit if profit column not available
    elif 'ebitda' in monthly_data.columns:
        for i, ebitda in enumerate(monthly_data['ebitda']):
            if ebitda >= 0:
                return i + 1  # Convert to 1-indexed

    return None  # No breakeven found


def optimize_for_breakeven(target_month=24, growth_profile="baseline", initial_investment=20000000):
    """
    Optimize the model to achieve breakeven by a specific month

    Parameters:
    -----------
    target_month : int
        Target month to achieve breakeven (1-indexed)
    growth_profile : str
        Base growth profile to start optimization from
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    print(
        f"Optimizing for breakeven at month {target_month} using {growth_profile} profile as base...")

    return run_integrated_financial_model(
        growth_profile=growth_profile,
        optimize_target="breakeven",
        target_month=target_month,
        initial_investment=initial_investment
    )


def optimize_for_series_b(target_month=36, growth_profile="aggressive", initial_investment=20000000):
    """
    Optimize the model to achieve Series B qualification by a specific month
    Series B qualification typically requires:
    - $10M+ ARR
    - 100%+ YoY growth rate
    - Good unit economics

    Parameters:
    -----------
    target_month : int
        Target month to achieve Series B qualification (1-indexed)
    growth_profile : str
        Base growth profile to start optimization from
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    print(
        f"Optimizing for Series B qualification at month {target_month} using {growth_profile} profile as base...")

    return run_integrated_financial_model(
        growth_profile=growth_profile,
        optimize_target="series_b",
        target_month=target_month,
        initial_investment=initial_investment
    )


def run_with_s_curve_profile(growth_profile="baseline", initial_investment=20000000):
    """
    Run the model with a specific s-curve growth profile

    Parameters:
    -----------
    growth_profile : str
        Growth profile to use ('baseline', 'conservative', 'aggressive', 'hypergrowth')
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, None)
    """
    valid_profiles = ['baseline', 'conservative', 'aggressive', 'hypergrowth']
    if growth_profile not in valid_profiles:
        raise ValueError(
            f"Invalid profile: {growth_profile}. Must be one of: {', '.join(valid_profiles)}")

    print(f"Running model with {growth_profile} s-curve profile...")

    return run_integrated_financial_model(
        growth_profile=growth_profile,
        initial_investment=initial_investment
    )


def run_with_acceleration_strategy(target_segments=None, acceleration_years=None,
                                   deceleration_years=None, accel_multiplier=2.0,
                                   decel_multiplier=0.5, initial_investment=20000000):
    """
    Run the model with a dynamic growth acceleration/deceleration strategy

    Parameters:
    -----------
    target_segments : list, optional
        List of segments to apply the strategy to. If None, applies to all segments.
    acceleration_years : list, optional
        List of years to accelerate growth. If None, no acceleration.
    deceleration_years : list, optional
        List of years to decelerate growth. If None, no deceleration.
    accel_multiplier : float, optional
        Multiplier to apply during acceleration years.
    decel_multiplier : float, optional
        Multiplier to apply during deceleration years.
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, None)
    """
    print(f"Running model with dynamic growth acceleration strategy...")

    # Run the base integrated model
    financial_model, revenue_model, cost_model, _ = run_integrated_financial_model(
        growth_profile="baseline",
        initial_investment=initial_investment
    )

    # Create a new model with the acceleration strategy
    accelerated_model = revenue_model.create_growth_acceleration_strategy(
        target_segments=target_segments,
        acceleration_years=acceleration_years,
        deceleration_years=deceleration_years,
        accel_multiplier=accel_multiplier,
        decel_multiplier=decel_multiplier
    )

    # Run the new model
    accelerated_model.run_model()

    # Re-run the cost and financial models with the new growth model
    cost_model.run_model(accelerated_model)
    financial_model = SaaSFinancialModel(accelerated_model, cost_model)
    financial_model.run_model()

    return financial_model, accelerated_model, cost_model, None


def run_with_year_by_year_strategy(segment_year_multipliers, initial_investment=20000000):
    """
    Run the model with custom year-by-year growth multipliers for each segment

    Parameters:
    -----------
    segment_year_multipliers : dict
        Dictionary with segments as keys and year-specific multipliers as values
        Example: {
            'Enterprise': {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.5, 6: 0.3},
            'Mid-Market': {1: 1.0, 2: 1.5, 3: 2.0, 4: 1.5, 5: 1.0, 6: 0.8},
            'SMB': {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0}
        }
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, None)
    """
    print(f"Running model with custom year-by-year growth strategy...")

    # Run the base integrated model
    financial_model, revenue_model, cost_model, _ = run_integrated_financial_model(
        growth_profile="baseline",
        initial_investment=initial_investment
    )

    # Create a new model with the custom year-by-year strategy
    custom_model = revenue_model.apply_dynamic_growth_strategy(
        segment_year_multipliers)

    # Run the new model
    custom_model.run_model()

    # Re-run the cost and financial models with the new growth model
    cost_model.run_model(custom_model)
    financial_model = SaaSFinancialModel(custom_model, cost_model)
    financial_model.run_model()

    return financial_model, custom_model, cost_model, None


def run_with_monthly_pattern(segment_month_multipliers, initial_investment=20000000):
    """
    Run the model with custom monthly growth multipliers for maximum flexibility

    Parameters:
    -----------
    segment_month_multipliers : dict
        Dictionary with segments as keys and month-specific multipliers as values
        Example: {
            'Enterprise': {1: 1.5, 2: 1.6, ..., 72: 0.5},
            'Mid-Market': {1: 0.8, 2: 0.9, ..., 72: 2.0},
            'SMB': {1: 0.5, 2: 0.5, ..., 72: 3.0}
        }
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, None)
    """
    print(f"Running model with custom monthly growth pattern...")

    # Run the base integrated model
    financial_model, revenue_model, cost_model, _ = run_integrated_financial_model(
        growth_profile="baseline",
        initial_investment=initial_investment
    )

    # Create a new model with the custom monthly pattern
    custom_model = revenue_model.apply_monthly_growth_pattern(
        segment_month_multipliers)

    # Run the new model
    custom_model.run_model()

    # Re-run the cost and financial models with the new growth model
    cost_model.run_model(custom_model)
    financial_model = SaaSFinancialModel(custom_model, cost_model)
    financial_model.run_model()

    return financial_model, custom_model, cost_model, None


def run_growth_scenarios(initial_investment=20000000):
    """
    Run the model with different growth scenarios to compare outcomes

    Parameters:
    -----------
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    dict : Dictionary containing the results of each scenario
    """
    os.makedirs('reports/scenarios', exist_ok=True)

    growth_profiles = ['baseline', 'conservative', 'aggressive', 'hypergrowth']
    scenario_results = {}

    for profile in growth_profiles:
        print(f"\nRunning {profile} growth scenario...")
        fm, rm, cm, _ = run_with_s_curve_profile(
            growth_profile=profile,
            initial_investment=initial_investment
        )

        # Store the models and key metrics
        scenario_results[profile] = {
            'financial_model': fm,
            'revenue_model': rm,
            'cost_model': cm,
            'monthly_data': fm.get_monthly_data(),
            'annual_data': fm.get_annual_data(),
            'key_metrics': fm.get_key_metrics_table(),
        }

    # Compare growth scenarios visually
    plt.figure(figsize=(15, 10))

    # Plot ARR growth comparison
    plt.subplot(2, 2, 1)
    for profile, data in scenario_results.items():
        plt.plot(data['monthly_data']['month_number'],
                 data['monthly_data']['total_arr'] / 1000000,
                 label=profile.capitalize())
    plt.xlabel('Month')
    plt.ylabel('ARR ($ Millions)')
    plt.title('ARR Growth by Scenario')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot cumulative burn comparison
    plt.subplot(2, 2, 2)
    for profile, data in scenario_results.items():
        cumulative_burn = data['monthly_data']['ebitda'].cumsum()
        plt.plot(data['monthly_data']['month_number'],
                 cumulative_burn / 1000000,
                 label=profile.capitalize())
    plt.xlabel('Month')
    plt.ylabel('Cumulative Cash Flow ($ Millions)')
    plt.title('Cumulative Cash Flow by Scenario')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot breakeven comparison
    plt.subplot(2, 2, 3)
    breakeven_months = {}
    for profile, data in scenario_results.items():
        # Find breakeven month
        monthly_data = data['monthly_data']
        breakeven_month = None
        # Try using profit column if available
        if 'profit' in monthly_data.columns:
            for i, profit in enumerate(monthly_data['profit']):
                if profit >= 0:
                    breakeven_month = i + 1  # Convert to 1-indexed
                    break
        # Fall back to ebitda column as proxy for profit if profit column not available
        elif 'ebitda' in monthly_data.columns:
            for i, ebitda in enumerate(monthly_data['ebitda']):
                if ebitda >= 0:
                    breakeven_month = i + 1  # Convert to 1-indexed
                    break

        if breakeven_month:
            breakeven_months[profile] = breakeven_month
            plt.bar(profile.capitalize(), breakeven_month,
                    color=f'C{list(scenario_results.keys()).index(profile)}')

    plt.ylabel('Month Number')
    plt.title('Breakeven Month by Scenario')
    plt.ylim(0, 72)  # 6 years
    for i, (profile, month) in enumerate(breakeven_months.items()):
        plt.text(i, month + 2, f"{month}",
                 ha='center', va='bottom', fontweight='bold')

    # Plot terminal valuation comparison (5x revenue)
    plt.subplot(2, 2, 4)
    for i, (profile, data) in enumerate(scenario_results.items()):
        terminal_revenue = data['annual_data']['annual_revenue'].iloc[-1]
        valuation_5x = terminal_revenue * 5 / 1000000  # 5x revenue in $M
        plt.bar(profile.capitalize(), valuation_5x,
                color=f'C{i}', alpha=0.7, label=f"5x Revenue")

    plt.ylabel('$ Millions')
    plt.title('Terminal Valuation (5x Revenue)')
    for i, (profile, data) in enumerate(scenario_results.items()):
        terminal_revenue = data['annual_data']['annual_revenue'].iloc[-1]
        valuation_5x = terminal_revenue * 5 / 1000000
        plt.text(i, valuation_5x + 20, f"${valuation_5x:.0f}M",
                 ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('reports/scenarios/growth_scenarios_comparison.png',
                bbox_inches='tight', dpi=300)

    # Save comparison table
    comparison_table = pd.DataFrame({
        'Scenario': [],
        'Year 6 ARR': [],
        'Year 6 Customers': [],
        'Year 6 EBITDA': [],
        'Breakeven Month': [],
        '5x Revenue Valuation': [],
    })

    for profile, data in scenario_results.items():
        annual = data['annual_data']
        monthly = data['monthly_data']

        # Find breakeven month
        breakeven_month = None
        # Try using profit column if available
        if 'profit' in monthly.columns:
            for i, profit in enumerate(monthly['profit']):
                if profit >= 0:
                    breakeven_month = i + 1  # Convert to 1-indexed
                    break
        # Fall back to ebitda column as proxy for profit if profit column not available
        elif 'ebitda' in monthly.columns:
            for i, ebitda in enumerate(monthly['ebitda']):
                if ebitda >= 0:
                    breakeven_month = i + 1  # Convert to 1-indexed
                    break

        # Create a new row as a DataFrame and concatenate with comparison_table
        new_row = pd.DataFrame({
            'Scenario': [profile.capitalize()],
            'Year 6 ARR': [annual['annual_revenue'].iloc[-1] / 1000000],  # $M
            'Year 6 Customers': [annual['year_end_customers'].iloc[-1]],
            # $M
            'Year 6 EBITDA': [annual['annual_ebitda'].iloc[-1] / 1000000],
            'Breakeven Month': [breakeven_month],
            # $M
            '5x Revenue Valuation': [annual['annual_revenue'].iloc[-1] * 5 / 1000000]
        })
        comparison_table = pd.concat(
            [comparison_table, new_row], ignore_index=True)

    comparison_table.to_csv(
        'reports/scenarios/scenario_comparison.csv', index=False)
    print("Growth scenario comparison saved to reports/scenarios/growth_scenarios_comparison.png")

    return scenario_results


def compare_optimization_targets():
    """
    Compare different optimization targets (breakeven vs Series B qualification)

    Returns:
    --------
    tuple : (breakeven_results, series_b_results)
    """
    os.makedirs('reports/optimization', exist_ok=True)

    print("\nRunning breakeven optimization for month 24...")
    be_fm, be_rm, be_cm, be_results = optimize_for_breakeven(
        target_month=24)

    print("\nRunning Series B qualification optimization for month 36...")
    sb_fm, sb_rm, sb_cm, sb_results = optimize_for_series_b(
        target_month=36)

    # Compare the two optimization scenarios visually
    plt.figure(figsize=(15, 12))

    # Plot ARR growth comparison
    plt.subplot(2, 2, 1)
    plt.plot(be_fm.get_monthly_data()['month_number'],
             be_fm.get_monthly_data()['total_arr'] / 1000000,
             label='Breakeven Optimized')
    plt.plot(sb_fm.get_monthly_data()['month_number'],
             sb_fm.get_monthly_data()['total_arr'] / 1000000,
             label='Series B Optimized')
    plt.xlabel('Month')
    plt.ylabel('ARR ($ Millions)')
    plt.title('ARR Growth Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot cash position comparison
    plt.subplot(2, 2, 2)
    plt.plot(be_fm.get_monthly_data()['month_number'],
             be_fm.get_monthly_data()['capital'] / 1000000,
             label='Breakeven Optimized')
    plt.plot(sb_fm.get_monthly_data()['month_number'],
             sb_fm.get_monthly_data()['capital'] / 1000000,
             label='Series B Optimized')
    plt.xlabel('Month')
    plt.ylabel('Capital ($ Millions)')
    plt.title('Cash Position Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot customer growth comparison
    plt.subplot(2, 2, 3)
    plt.plot(be_rm.get_monthly_data()['month_number'],
             be_rm.get_monthly_data()['total_customers'],
             label='Breakeven Optimized')
    plt.plot(sb_rm.get_monthly_data()['month_number'],
             sb_rm.get_monthly_data()['total_customers'],
             label='Series B Optimized')
    plt.xlabel('Month')
    plt.ylabel('Total Customers')
    plt.title('Customer Growth Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot profit/loss comparison
    plt.subplot(2, 2, 4)
    plt.plot(be_fm.get_monthly_data()['month_number'],
             be_fm.get_monthly_data()['ebitda'] / 1000000,
             label='Breakeven Optimized')
    plt.plot(sb_fm.get_monthly_data()['month_number'],
             sb_fm.get_monthly_data()['ebitda'] / 1000000,
             label='Series B Optimized')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Month')
    plt.ylabel('Monthly EBITDA ($ Millions)')
    plt.title('EBITDA Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/optimization/optimization_targets_comparison.png',
                bbox_inches='tight', dpi=300)

    # Create comparison table
    comparison = pd.DataFrame({
        'Metric': [
            'Optimization Target',
            'Target Month',
            'Achieved Month',
            'Growth Multiplier',
            'Breakeven Month',
            'Year 3 ARR ($M)',
            'Year 3 Customers',
            'Year 6 ARR ($M)',
            'Year 6 EBITDA ($M)',
            'Year 6 Valuation (5x Revenue) ($M)',
            'Minimum Cash Position ($M)',
        ],
        'Breakeven Optimization': [
            'Breakeven',
            be_results['target_month'],
            be_results['achieved_month'],
            f"{be_results['growth_multiplier']:.2f}x",
            find_breakeven_month(be_fm),
            be_fm.get_annual_data()['annual_revenue'].iloc[2] / 1000000,
            be_rm.get_annual_data()['total_ending_customers'].iloc[2],
            be_fm.get_annual_data()['annual_revenue'].iloc[-1] / 1000000,
            be_fm.get_annual_data()['annual_ebitda'].iloc[-1] / 1000000,
            be_fm.get_annual_data()['annual_revenue'].iloc[-1] * 5 / 1000000,
            be_fm.get_monthly_data()['capital'].min() / 1000000,
        ],
        'Series B Optimization': [
            'Series B',
            sb_results['target_month'],
            sb_results['achieved_month'],
            f"{sb_results['growth_multiplier']:.2f}x",
            find_breakeven_month(sb_fm),
            sb_fm.get_annual_data()['annual_revenue'].iloc[2] / 1000000,
            sb_rm.get_annual_data()['total_ending_customers'].iloc[2],
            sb_fm.get_annual_data()['annual_revenue'].iloc[-1] / 1000000,
            sb_fm.get_annual_data()['annual_ebitda'].iloc[-1] / 1000000,
            sb_fm.get_annual_data()['annual_revenue'].iloc[-1] * 5 / 1000000,
            sb_fm.get_monthly_data()['capital'].min() / 1000000,
        ]
    })

    comparison.to_csv(
        'reports/optimization/optimization_comparison.csv', index=False)
    print("Optimization comparison saved to reports/optimization/optimization_targets_comparison.png")

    return {
        'breakeven': {
            'financial_model': be_fm,
            'revenue_model': be_rm,
            'cost_model': be_cm,
            'results': be_results
        },
        'series_b': {
            'financial_model': sb_fm,
            'revenue_model': sb_rm,
            'cost_model': sb_cm,
            'results': sb_results
        }
    }


def run_enterprise_first_strategy(initial_investment=20000000):
    """
    Run a model with a 'enterprise-first' growth strategy:
    - Focus on enterprise and mid-market in years 1-2
    - Shift to mid-market and SMB in years 3-4
    - More balanced approach in years 5-6

    Parameters:
    -----------
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, None)
    """
    print("Running 'enterprise-first' growth strategy model...")

    # Create a year-by-year strategy that prioritizes enterprise and mid-market initially
    segment_year_multipliers = {
        'Enterprise': {
            1: 2.0,  # Strong focus in year 1
            2: 1.8,  # Strong focus in year 2
            3: 1.4,  # Moderate focus in year 3
            4: 1.2,  # Less focus in year 4
            5: 1.0,  # Back to baseline in year 5
            6: 0.9,  # Slight deceleration in year 6
        },
        'Mid-Market': {
            1: 1.5,  # Good focus in year 1
            2: 1.7,  # Increased focus in year 2
            3: 1.8,  # Peak focus in year 3
            4: 1.6,  # Still strong in year 4
            5: 1.3,  # Moderate focus in year 5
            6: 1.1,  # Slight focus in year 6
        },
        'SMB': {
            1: 0.7,  # Low focus initially
            2: 0.8,  # Still low focus
            3: 1.3,  # Increased focus in year 3
            4: 1.7,  # Strong focus in year 4
            5: 2.0,  # Peak focus in year 5
            6: 2.2,  # Continued strong focus in year 6
        }
    }

    # Run the model with this strategy
    return run_with_year_by_year_strategy(
        segment_year_multipliers=segment_year_multipliers,
        initial_investment=initial_investment
    )


def run_regulatory_impact_strategy(initial_investment=20000000):
    """
    Run a model that accounts for AI regulation impact on growth:
    - Strong Enterprise focus in years 1-3 (companies with resources for compliance)
    - Strong Mid-Market in years 2-4 (following Enterprise adoption patterns)
    - Delayed SMB growth until years 5-6 (due to regulatory barriers and costs)

    Parameters:
    -----------
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, None)
    """
    print("Running 'AI regulation impact' growth strategy model...")

    # Create a year-by-year strategy accounting for AI regulations slowing SMB adoption
    segment_year_multipliers = {
        'Enterprise': {
            # Very strong focus in year 1 (early adopters with resources for compliance)
            1: 2.2,
            2: 2.0,  # Strong focus in year 2
            3: 1.6,  # Continued strong focus as regulations solidify
            4: 1.3,  # Moderate focus in year 4
            5: 1.1,  # Slightly above baseline in year 5
            6: 1.0,  # Baseline in year 6
        },
        'Mid-Market': {
            1: 1.7,  # Good focus in year 1
            2: 1.9,  # Increased focus in year 2 as mid-market follows enterprise
            3: 2.0,  # Peak focus in year 3 as mid-market adoption accelerates
            4: 1.8,  # Still strong in year 4
            5: 1.5,  # Moderate focus in year 5
            6: 1.3,  # Continued moderate focus in year 6
        },
        'SMB': {
            1: 0.3,  # Minimal focus initially due to regulatory barriers
            2: 0.4,  # Still very low focus due to compliance costs
            3: 0.6,  # Gradual increase as compliance frameworks become accessible
            4: 1.0,  # Reaching baseline as regulations stabilize
            5: 1.8,  # Accelerating as compliance becomes more standardized
            6: 2.5,  # Strong catch-up growth as barriers lower and solutions become turnkey
        }
    }

    # Run the model with this strategy
    return run_with_year_by_year_strategy(
        segment_year_multipliers=segment_year_multipliers,
        initial_investment=initial_investment
    )


def compare_dynamic_strategies(initial_investment=20000000):
    """
    Compare different dynamic growth strategies to showcase the flexibility
    of the updated growth model

    Parameters:
    -----------
    initial_investment : float
        Initial investment amount

    Returns:
    --------
    dict : Dictionary containing the results of each strategy
    """
    os.makedirs('reports/strategies', exist_ok=True)

    # Define our strategies
    strategies = {}

    # 1. Baseline S-curve strategy (for comparison)
    print("\nRunning baseline s-curve strategy...")
    fm1, rm1, cm1, _ = run_with_s_curve_profile(
        growth_profile="baseline",
        initial_investment=initial_investment
    )
    strategies['Baseline'] = {
        'financial_model': fm1,
        'revenue_model': rm1,
        'cost_model': cm1
    }

    # 2. Enterprise-first strategy
    print("\nRunning enterprise-first strategy...")
    fm2, rm2, cm2, _ = run_enterprise_first_strategy(
        initial_investment=initial_investment
    )
    strategies['Enterprise First'] = {
        'financial_model': fm2,
        'revenue_model': rm2,
        'cost_model': cm2
    }

    # 3. Acceleration/deceleration strategy
    print("\nRunning acceleration/deceleration strategy...")
    fm3, rm3, cm3, _ = run_with_acceleration_strategy(
        target_segments=['Enterprise', 'Mid-Market'],
        acceleration_years=[1, 2],
        deceleration_years=[5, 6],
        accel_multiplier=2.0,
        decel_multiplier=0.6,
        initial_investment=initial_investment
    )
    strategies['Accelerated'] = {
        'financial_model': fm3,
        'revenue_model': rm3,
        'cost_model': cm3
    }

    # Compare visually
    plt.figure(figsize=(15, 15))

    # ARR comparison
    plt.subplot(2, 2, 1)
    for name, models in strategies.items():
        plt.plot(
            models['revenue_model'].get_monthly_data()['month_number'],
            models['revenue_model'].get_monthly_data()['total_arr'] / 1000000,
            label=name
        )
    plt.title('Total ARR Comparison')
    plt.xlabel('Month')
    plt.ylabel('ARR ($ Millions)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Customer comparison
    plt.subplot(2, 2, 2)
    for name, models in strategies.items():
        plt.plot(
            models['revenue_model'].get_monthly_data()['month_number'],
            models['revenue_model'].get_monthly_data()['total_customers'],
            label=name
        )
    plt.title('Total Customers Comparison')
    plt.xlabel('Month')
    plt.ylabel('Customers')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Enterprise segment comparison
    plt.subplot(2, 2, 3)
    for name, models in strategies.items():
        plt.plot(
            models['revenue_model'].get_monthly_data()['month_number'],
            models['revenue_model'].get_monthly_data()['Enterprise_customers'],
            label=name
        )
    plt.title('Enterprise Customers Comparison')
    plt.xlabel('Month')
    plt.ylabel('Enterprise Customers')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Cash position comparison
    plt.subplot(2, 2, 4)
    for name, models in strategies.items():
        plt.plot(
            models['financial_model'].get_monthly_data()['month_number'],
            models['financial_model'].get_monthly_data()['capital'] / 1000000,
            label=name
        )
    plt.title('Cash Position Comparison')
    plt.xlabel('Month')
    plt.ylabel('Capital ($ Millions)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/strategies/dynamic_strategies_comparison.png',
                bbox_inches='tight', dpi=300)

    # Save comparison metrics
    comparison_metrics = pd.DataFrame({
        'Strategy': [],
        'Year 6 ARR ($M)': [],
        'Year 6 Customers': [],
        'Total Enterprise Customers Y6': [],
        'Total Mid-Market Customers Y6': [],
        'Total SMB Customers Y6': [],
        'Breakeven Month': [],
        'Minimum Cash ($M)': [],
        'Year 6 EBITDA ($M)': [],
    })

    for name, models in strategies.items():
        # Find breakeven month
        fm = models['financial_model']
        rm = models['revenue_model']
        monthly_data = fm.get_monthly_data()

        breakeven_month = None
        # Try using profit column if available
        if 'profit' in monthly_data.columns:
            for i, profit in enumerate(monthly_data['profit']):
                if profit >= 0:
                    breakeven_month = i + 1  # Convert to 1-indexed
                    break
        # Fall back to ebitda column as proxy for profit if profit column not available
        elif 'ebitda' in monthly_data.columns:
            for i, ebitda in enumerate(monthly_data['ebitda']):
                if ebitda >= 0:
                    breakeven_month = i + 1  # Convert to 1-indexed
                    break

        annual_data = fm.get_annual_data()
        # Create a new row as a DataFrame and concatenate with comparison_metrics
        new_row = pd.DataFrame({
            'Strategy': [name],
            'Year 6 ARR ($M)': [annual_data['annual_revenue'].iloc[-1] / 1000000],
            'Year 6 Customers': [rm.get_annual_data()['total_ending_customers'].iloc[-1]],
            'Total Enterprise Customers Y6': [rm.get_annual_data()['Enterprise_ending_customers'].iloc[-1]],
            'Total Mid-Market Customers Y6': [rm.get_annual_data()['Mid-Market_ending_customers'].iloc[-1]],
            'Total SMB Customers Y6': [rm.get_annual_data()['SMB_ending_customers'].iloc[-1]],
            'Breakeven Month': [breakeven_month],
            'Minimum Cash ($M)': [monthly_data['capital'].min() / 1000000],
            'Year 6 EBITDA ($M)': [annual_data['annual_ebitda'].iloc[-1] / 1000000],
        })
        comparison_metrics = pd.concat(
            [comparison_metrics, new_row], ignore_index=True)

    comparison_metrics.to_csv(
        'reports/strategies/dynamic_strategies_metrics.csv', index=False)
    print("Dynamic growth strategies comparison saved to reports/strategies/")

    return strategies, comparison_metrics


def optimize_for_revenue_target(target_revenue=5000000, target_month=36, growth_profile="baseline", initial_investment=20000000):
    """
    Optimize the model to achieve a specific revenue target by a given month.
    
    Parameters:
    -----------
    target_revenue : float
        Target monthly revenue to achieve (not ARR)
    target_month : int
        Target month to achieve the revenue (1-indexed)
    growth_profile : str
        Base growth profile to start optimization from
    initial_investment : float
        Initial investment amount
    
    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    print(f"Optimizing for ${target_revenue/1000000:.2f}M monthly revenue at month {target_month}...")
    
    # Load configuration
    revenue_config = load_config("configs/revenue_config.json")
    cost_config = load_config("configs/cost_config.json")
    
    # Create base models
    revenue_model = SaaSGrowthModel(revenue_config)
    cost_model = AISaaSCostModel(cost_config)
    
    # Calculate the number of customers needed to reach the target revenue
    target_arr = target_revenue * 12  # Convert monthly revenue to ARR
    
    # Use the apply_growth_profile method to set up initial growth rates
    revenue_model = revenue_model.apply_growth_profile(growth_profile)
    
    # Run the model to get segment distribution 
    revenue_model.run_model()
    
    # Get the data at the target month
    monthly_data = revenue_model.get_monthly_data()
    target_month_data = monthly_data[monthly_data['month_number'] == target_month].iloc[0]
    
    # Calculate the ratio of current ARR to target ARR
    current_arr = target_month_data['total_arr']
    arr_ratio = target_arr / current_arr if current_arr > 0 else 1.0
    
    # Scale customer counts to achieve target revenue
    segment_multipliers = {segment: arr_ratio for segment in revenue_config['segments']}
    
    # Apply the scaled customer counts
    revenue_model = revenue_model.apply_custom_segment_profiles(segment_multipliers)
    
    # Run the models
    revenue_model.run_model()
    cost_model.run_model(revenue_model)
    financial_model = SaaSFinancialModel(revenue_model, cost_model)
    financial_model.run_model()
    
    # Calculate optimization results
    monthly_data = revenue_model.get_monthly_data()
    target_month_data = monthly_data[monthly_data['month_number'] == target_month].iloc[0]
    
    # For monthly revenue, use ARR / 12 since monthly_revenue field might not exist
    achieved_arr = target_month_data['total_arr']
    achieved_revenue = achieved_arr / 12  # Convert ARR to monthly revenue
    
    optimization_results = {
        'target': 'monthly_revenue',
        'target_month': target_month,
        'target_revenue': target_revenue,
        'achieved_revenue': achieved_revenue,
        'achieved_arr': achieved_arr,
        'revenue_ratio': achieved_revenue / target_revenue,
        'customer_multiplier': arr_ratio
    }
    
    # Save optimization results
    os.makedirs('reports/optimization', exist_ok=True)
    pd.DataFrame([optimization_results]).to_csv('reports/optimization/revenue_target_results.csv', index=False)
    
    # Print optimization results
    print(f"\nRevenue Target Optimization Results:")
    print(f"Target: ${target_revenue/1000000:.2f}M monthly revenue by month {target_month}")
    print(f"Achieved: ${achieved_revenue/1000000:.2f}M monthly revenue")
    print(f"Customer scaling factor: {arr_ratio:.2f}x")
    
    return financial_model, revenue_model, cost_model, optimization_results
    
def optimize_for_annual_revenue(target_annual_revenue=360000000, target_year=5, initial_investment=20000000):
    """
    Optimize the model to achieve a specific annual revenue target by a given year.
    Uses a more sophisticated segmentation approach with year-by-year tapering.
    
    Parameters:
    -----------
    target_annual_revenue : float
        Target annual revenue to achieve
    target_year : int
        Target year to achieve the revenue (1-indexed)
    initial_investment : float
        Initial investment amount
    
    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    print(f"Optimizing for ${target_annual_revenue/1000000:.2f}M annual revenue by year {target_year}...")
    
    # Load configuration
    revenue_config = load_config("configs/revenue_config.json")
    cost_config = load_config("configs/cost_config.json")
    
    # Create base models
    revenue_model = SaaSGrowthModel(revenue_config)
    cost_model = AISaaSCostModel(cost_config)
    
    # First, determine the base scale factor needed
    # Run a baseline model to get the default annual revenue
    baseline_model = revenue_model.apply_growth_profile("baseline")
    baseline_model.run_model()
    baseline_annual = baseline_model.get_annual_data()
    
    # Find the revenue for the target year in the baseline model
    if target_year <= len(baseline_annual):
        baseline_revenue = baseline_annual.iloc[target_year-1]['total_ending_arr']
    else:
        # Use the last year if target year is beyond projection period
        baseline_revenue = baseline_annual.iloc[-1]['total_ending_arr']
    
    # Calculate the basic scaling factor needed
    scaling_factor = target_annual_revenue / baseline_revenue if baseline_revenue > 0 else 1.0
    
    # Create a year-by-year segment strategy with appropriate tapering
    # Earlier years grow faster, later years taper
    segment_year_multipliers = {}
    
    # Different strategies for different segments
    for segment in revenue_config['segments']:
        segment_year_multipliers[segment] = {}
        
        # Determine segment-specific strategies
        if segment == 'Enterprise':
            # Enterprise segments get more focus in early years, less in later years
            segment_year_multipliers[segment] = {
                1: scaling_factor * 1.8,  # Early focus
                2: scaling_factor * 1.6,
                3: scaling_factor * 1.4,
                4: scaling_factor * 1.2,
                5: scaling_factor * 1.0,
                6: scaling_factor * 0.8   # Tapered focus
            }
        elif segment == 'Mid-Market':
            # Mid-Market gets balanced focus
            segment_year_multipliers[segment] = {
                1: scaling_factor * 1.2,
                2: scaling_factor * 1.4,
                3: scaling_factor * 1.5,
                4: scaling_factor * 1.4,
                5: scaling_factor * 1.2,
                6: scaling_factor * 1.0
            }
        elif segment == 'SMB':
            # SMB gets later focus
            segment_year_multipliers[segment] = {
                1: scaling_factor * 0.8,
                2: scaling_factor * 1.0,
                3: scaling_factor * 1.2,
                4: scaling_factor * 1.5,
                5: scaling_factor * 1.8,
                6: scaling_factor * 2.0
            }
    
    # Apply the year-by-year strategy to the revenue model
    custom_model = revenue_model.apply_dynamic_growth_strategy(segment_year_multipliers)
    
    # Run the models
    custom_model.run_model()
    cost_model.run_model(custom_model)
    financial_model = SaaSFinancialModel(custom_model, cost_model)
    financial_model.run_model()
    
    # Calculate optimization results
    annual_data = custom_model.get_annual_data()
    
    # Get the achieved revenue for the target year
    if target_year <= len(annual_data):
        achieved_annual_revenue = annual_data.iloc[target_year-1]['total_ending_arr']
    else:
        # Use the last year if target year is beyond projection period
        achieved_annual_revenue = annual_data.iloc[-1]['total_ending_arr']
    
    # Calculate growth rate from previous year
    if target_year > 1 and target_year <= len(annual_data):
        previous_year_revenue = annual_data.iloc[target_year-2]['total_ending_arr']
        growth_rate = (achieved_annual_revenue / previous_year_revenue - 1) * 100 if previous_year_revenue > 0 else 0
    else:
        growth_rate = 0
    
    optimization_results = {
        'target': 'annual_revenue',
        'target_year': target_year,
        'target_annual_revenue': target_annual_revenue,
        'achieved_annual_revenue': achieved_annual_revenue,
        'revenue_ratio': achieved_annual_revenue / target_annual_revenue,
        'base_scaling_factor': scaling_factor,
        'yoy_growth_rate': growth_rate
    }
    
    # Save optimization results
    os.makedirs('reports/optimization', exist_ok=True)
    pd.DataFrame([optimization_results]).to_csv('reports/optimization/annual_revenue_results.csv', index=False)
    
    # Print optimization results
    print(f"\nAnnual Revenue Optimization Results:")
    print(f"Target: ${target_annual_revenue/1000000:.2f}M annual revenue by year {target_year}")
    print(f"Achieved: ${achieved_annual_revenue/1000000:.2f}M annual revenue")
    print(f"Base scaling factor: {scaling_factor:.2f}x")
    print(f"Year-over-Year growth rate: {growth_rate:.1f}%")
    
    # Create a visually appealing report
    plt.figure(figsize=(14, 8))
    
    # Plot the annual revenue with the target
    annual_revenue = [data['total_ending_arr'] / 1000000 for _, data in annual_data.iterrows()]
    years = [data['year'] for _, data in annual_data.iterrows()]
    
    plt.bar(years, annual_revenue, color='skyblue', alpha=0.7)
    
    # Add a target line
    if target_year <= len(years):
        plt.axhline(y=target_annual_revenue/1000000, color='red', linestyle='-', alpha=0.7)
        plt.plot(target_year, target_annual_revenue/1000000, 'ro', markersize=10)
        plt.annotate(f'Target: ${target_annual_revenue/1000000:.1f}M', 
                    xy=(target_year, target_annual_revenue/1000000),
                    xytext=(target_year-0.5, target_annual_revenue/1000000*1.1),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12)
    
    plt.title(f'Annual Revenue Optimization for Year {target_year}', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Annual Revenue ($ Millions)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add revenue amounts on top of bars
    for i, revenue in enumerate(annual_revenue):
        plt.text(years[i], revenue + 5, f'${revenue:.1f}M', ha='center', fontsize=10)
    
    # Save the figure
    os.makedirs('reports/optimization', exist_ok=True)
    plt.savefig('reports/optimization/annual_revenue_growth.png', bbox_inches='tight', dpi=300)
    
    return financial_model, custom_model, cost_model, optimization_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run financial models with different strategies')
    parser.add_argument('--strategy', type=str, default='baseline', choices=[
        'baseline',
        'conservative',
        'aggressive',
        'hypergrowth',
        'breakeven',
        'series_b',
        'series_b_revenue',
        'revenue_target',
        'annual_revenue',
        'enterprise_first',
        'regulatory_impact',
        'acceleration',
        'compare_all'
    ], help='Strategy to run')
    parser.add_argument('--investment', type=float, default=20000000,
                        help='Initial investment amount')
    parser.add_argument('--breakeven-target', type=int, default=24,
                        help='Target month to achieve breakeven (for breakeven strategy)')
    parser.add_argument('--series-b-target', type=int, default=36,
                        help='Target month to achieve Series B qualification')
    parser.add_argument('--series-b-revenue-target', type=float, default=30000000,
                        help='Target annual revenue for Series B readiness')
    parser.add_argument('--capital-burn-target', type=float, default=0.8,
                        help='Target percentage of initial capital to use (0.0-1.0)')
    parser.add_argument('--allow-over-burn', action='store_true',
                        help='Allow burning more than 100%% of initial capital (implies additional funding)')
    parser.add_argument('--revenue-target', type=float, default=5000000,
                        help='Target monthly revenue to achieve (for revenue_target strategy)')
    parser.add_argument('--revenue-target-month', type=int, default=36,
                        help='Target month to achieve revenue target')
    parser.add_argument('--annual-revenue-target', type=float, default=360000000,
                        help='Target annual revenue to achieve (for annual_revenue strategy)')
    parser.add_argument('--annual-revenue-year', type=int, default=5,
                        help='Target year to achieve annual revenue target')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Directory to save reports')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(f'{args.output_dir}/combined', exist_ok=True)
    os.makedirs(f'{args.output_dir}/growth', exist_ok=True)
    os.makedirs(f'{args.output_dir}/cost', exist_ok=True)
    os.makedirs(f'{args.output_dir}/optimization', exist_ok=True)
    os.makedirs(f'{args.output_dir}/strategies', exist_ok=True)

    # Run the selected strategy
    if args.strategy == 'compare_all':
        print("Comparing all growth strategies...")
        strategies, metrics = compare_dynamic_strategies(
            initial_investment=args.investment)
        print("\nDynamic Growth Strategy Comparison:")
        print(metrics)

        # Compare different growth scenarios
        print("\nComparing different growth scenarios...")
        scenarios = run_growth_scenarios(initial_investment=args.investment)

        # Compare optimization targets
        print("\nComparing optimization targets...")
        opt_comparison = compare_optimization_targets()

        # Return the Series B optimization results for plotting
        financial_model, revenue_model, cost_model, _ = opt_comparison['series_b'].values(
        )

    elif args.strategy == 'breakeven':
        print(f"Optimizing for breakeven at month {args.breakeven_target}...")
        financial_model, revenue_model, cost_model, optimization_results = optimize_for_breakeven(
            target_month=args.breakeven_target,
            initial_investment=args.investment
        )

        # Print optimization results
        if optimization_results:
            if 'format_type' in optimization_results and optimization_results['format_type'] == 'series_b_revenue':
                # We've already printed results for series_b_revenue strategy
                pass
            else:
                print(f"\nOptimization Results:")
                print(
                    f"Target: {optimization_results['target']} by month {optimization_results['target_month']}")
                if 'achieved_month' in optimization_results and optimization_results['achieved_month'] is not None:
                    print(
                        f"Achieved in month: {optimization_results['achieved_month']}")
                else:
                    print(f"Target not achieved within the projection period")
                if 'growth_multiplier' in optimization_results:
                    print(
                        f"Growth multiplier applied: {optimization_results['growth_multiplier']:.2f}x")

    elif args.strategy == 'series_b':
        print(
            f"Optimizing for Series B qualification at month {args.series_b_target}...")
        financial_model, revenue_model, cost_model, optimization_results = optimize_for_series_b(
            target_month=args.series_b_target,
            initial_investment=args.investment
        )
    elif args.strategy == 'series_b_revenue':
        print(f"Optimizing for Series B readiness with ${args.series_b_revenue_target/1000000:.2f}M revenue by month {args.series_b_target}...")
        
        # This functionality has been integrated. Run a variation of optimize_for_revenue_target
        # but with Series B qualification requirements
        financial_model, revenue_model, cost_model, optimization_results = optimize_for_revenue_target(
            target_revenue=args.series_b_revenue_target/12,  # Convert annual to monthly target
            target_month=args.series_b_target,
            growth_profile="aggressive",  # Use more aggressive profile for Series B
            initial_investment=args.investment
        )
        
        # Add Series B specific metrics to the results
        if optimization_results:
            # Check if we have annual data to calculate YoY growth
            annual_data = revenue_model.get_annual_data()
            target_year = (args.series_b_target // 12) + 1
            
            # Calculate YoY growth rate if we have enough data
            yoy_growth_rate = 0
            if target_year > 1 and target_year <= len(annual_data):
                current_arr = annual_data.iloc[target_year-1]['total_ending_arr']
                prev_arr = annual_data.iloc[target_year-2]['total_ending_arr'] 
                yoy_growth_rate = ((current_arr / prev_arr) - 1) * 100 if prev_arr > 0 else 0
            
            # Calculate capital utilization
            monthly_data = financial_model.get_monthly_data()
            min_capital = monthly_data['capital'].min()
            capital_burn = 1 - (min_capital / args.investment)
            capital_burn = max(0, min(1, capital_burn))  # Constrain between 0 and 1
            
            # Determine if Series B qualified (10M+ ARR, 100%+ YoY growth)
            monthly_at_target = monthly_data[monthly_data['month_number'] == args.series_b_target].iloc[0]
            arr_at_target = monthly_at_target['total_arr']
            # ARR is already annual, no need to adjust
            series_b_qualified = arr_at_target >= args.series_b_revenue_target and yoy_growth_rate >= 100
            
            # Add metrics to results
            optimization_results.update({
                'target': 'series_b_revenue',
                'yoy_growth_rate': yoy_growth_rate,
                'capital_burn_percentage': capital_burn,
                'series_b_qualified': series_b_qualified,
                'achieved_annual_revenue': arr_at_target,
                'format_type': 'series_b_revenue'
            })
            
            # Print Series B specific results
            print(f"\nSeries B Revenue Optimization Results:")
            print(f"Target: ${args.series_b_revenue_target/1000000:.2f}M annual revenue by month {args.series_b_target}")
            print(f"Achieved: ${arr_at_target/1000000:.2f}M annual revenue")
            print(f"YoY Growth Rate: {yoy_growth_rate:.1f}%")
            print(f"Capital Utilization: {capital_burn*100:.1f}% of initial investment")
            print(f"Series B Ready: {'Yes' if series_b_qualified else 'No'}")
            
            # Check if we're likely to run out of money
            if args.allow_over_burn == False and capital_burn > args.capital_burn_target:
                print("\nWARNING: This growth strategy would utilize {:.1f}% of initial capital,".format(capital_burn*100))
                print("which exceeds your specified limit of {:.1f}%.".format(args.capital_burn_target*100))
                print("Consider enabling --allow-over-burn if you plan to raise additional funding.")
        else:
            print("\nThe optimization could not find a valid solution that meets both the revenue target")
            print("and stays within capital limits. The current model indicates that the chosen targets")
            print("would require more than 100% of the initial investment.")
            print("\nPossible solutions:")
            print("1. Increase initial investment (currently ${:.1f}M)".format(args.investment/1000000))
            print("2. Reduce revenue target (currently ${:.1f}M)".format(args.series_b_revenue_target/1000000))
            print("3. Extend target timeframe (currently {} months)".format(args.series_b_target))
            print("4. Create a more efficient cost structure in configs/cost_config_efficient.json")
            print("5. Accept burning more than 100% of the initial investment (implies additional funding)")
            return None, None, None, None
                
    elif args.strategy == 'revenue_target':
        print(
            f"Optimizing for ${args.revenue_target/1000000:.2f}M monthly revenue at month {args.revenue_target_month}...")
        financial_model, revenue_model, cost_model, optimization_results = optimize_for_revenue_target(
            target_revenue=args.revenue_target,
            target_month=args.revenue_target_month,
            growth_profile="baseline",
            initial_investment=args.investment
        )
        
        # Optimization results are printed by the optimize_for_revenue_target function
        
    elif args.strategy == 'annual_revenue':
        print(
            f"Optimizing for ${args.annual_revenue_target/1000000:.2f}M annual revenue by year {args.annual_revenue_year}...")
        financial_model, revenue_model, cost_model, optimization_results = optimize_for_annual_revenue(
            target_annual_revenue=args.annual_revenue_target,
            target_year=args.annual_revenue_year,
            initial_investment=args.investment
        )
        
        # Optimization results are printed by the optimize_for_annual_revenue function

    elif args.strategy == 'enterprise_first':
        print("Running Enterprise-First strategy...")
        financial_model, revenue_model, cost_model, _ = run_enterprise_first_strategy(
            initial_investment=args.investment
        )

    elif args.strategy == 'regulatory_impact':
        print("Running Regulatory Impact strategy...")
        financial_model, revenue_model, cost_model, _ = run_regulatory_impact_strategy(
            initial_investment=args.investment
        )

    elif args.strategy == 'acceleration':
        print("Running Growth Acceleration/Deceleration strategy...")
        financial_model, revenue_model, cost_model, _ = run_with_acceleration_strategy(
            target_segments=['Enterprise', 'Mid-Market'],
            acceleration_years=[1, 2],
            deceleration_years=[5, 6],
            accel_multiplier=2.0,
            decel_multiplier=0.6,
            initial_investment=args.investment
        )

    else:
        # Default case: run with specified growth curve profile
        print(f"Running {args.strategy} growth profile...")
        financial_model, revenue_model, cost_model, _ = run_with_s_curve_profile(
            growth_profile=args.strategy,
            initial_investment=args.investment
        )

    # Get results for the selected model
    annual_data = financial_model.get_annual_data()

    # Print key metrics for the selected model
    print("\nKey Financial Metrics:")
    print(financial_model.get_key_metrics_table())

    # Save key metrics table
    metrics_table = financial_model.get_key_metrics_table()
    metrics_table.to_csv(f'{args.output_dir}/combined/key_metrics.csv')

    # Generate and save combined reports
    generate_reports(
        financial_model,
        revenue_model,
        cost_model,
        output_dir=args.output_dir
    )

    # Print profitability analysis
    print_profitability_analysis(financial_model, annual_data)

    return financial_model, revenue_model, cost_model


def generate_reports(financial_model, revenue_model, cost_model, output_dir='reports'):
    """Generate and save all reports for a financial model run"""
    # ---------- COMBINED REPORTS ----------#

    # Plot and save key charts
    fig1 = financial_model.plot_financial_summary(figsize=(14, 8))
    fig1.savefig(f'{output_dir}/combined/financial_summary.png',
                 bbox_inches='tight', dpi=300)

    fig2 = financial_model.plot_break_even_analysis(figsize=(14, 8))
    fig2.savefig(f'{output_dir}/combined/break_even_analysis.png',
                 bbox_inches='tight', dpi=300)

    fig3 = financial_model.plot_runway_and_capital(figsize=(14, 8))
    fig3.savefig(f'{output_dir}/combined/runway_and_capital.png',
                 bbox_inches='tight', dpi=300)

    fig4 = financial_model.plot_unit_economics(figsize=(14, 8))
    fig4.savefig(f'{output_dir}/combined/unit_economics.png',
                 bbox_inches='tight', dpi=300)

    # Get monthly and annual data
    monthly_data = financial_model.get_monthly_data()

    # Save cashflow data
    monthly_data[['date', 'monthly_revenue', 'total_cogs', 'total_operating_expenses',
                 'gross_profit', 'ebitda', 'cash_flow', 'capital']].to_csv(
        f'{output_dir}/combined/monthly_cashflow.csv', index=False)

    annual_cashflow = pd.DataFrame({
        'Year': financial_model.annual_data['year'],
        'Revenue': financial_model.annual_data['annual_revenue'],
        'COGS': financial_model.annual_data['annual_total_cogs'],
        'OpEx': financial_model.annual_data['annual_total_operating_expenses'],
        'EBITDA': financial_model.annual_data['annual_ebitda'],
        'EBITDA_Margin': financial_model.annual_data['annual_ebitda_margin'],
        'Year_End_Capital': financial_model.annual_data['year_end_capital']
    })
    annual_cashflow.to_csv(
        f'{output_dir}/combined/annual_cashflow.csv', index=False)

    # Save unit economics data
    unit_economics = pd.DataFrame({
        'Year': financial_model.annual_data['year'],
        'CAC': financial_model.annual_data['annual_avg_cac'],
        'LTV': financial_model.annual_data['annual_avg_ltv'],
        'LTV_CAC_Ratio': financial_model.annual_data['annual_ltv_cac_ratio'],
        'ARPU': monthly_data.groupby('year_number')['arpu'].mean().values,
        'Gross_Margin': financial_model.annual_data['annual_gross_margin']
    })
    unit_economics.to_csv(
        f'{output_dir}/combined/unit_economics.csv', index=False)

    # ---------- GROWTH MODEL REPORTS ----------#

    # Save growth model data
    revenue_monthly = revenue_model.get_monthly_data()
    revenue_annual = revenue_model.get_annual_data()

    # Save monthly growth data
    revenue_monthly[['date', 'total_arr', 'total_customers', 'total_new_customers',
                    'total_churned_customers']].to_csv(
        f'{output_dir}/growth/monthly_data.csv', index=False)

    # Save annual growth data
    revenue_annual[['year', 'total_ending_customers', 'total_ending_arr',
                   'total_new_customers', 'total_churned_customers',
                    'total_arr_growth_rate']].to_csv(
        f'{output_dir}/growth/annual_data.csv', index=False)

    # Save growth summary
    growth_summary = pd.DataFrame({
        'Year': revenue_annual['year'],
        'Customers': revenue_annual['total_ending_customers'],
        'ARR': revenue_annual['total_ending_arr'],
        'Growth_Rate': revenue_annual['total_arr_growth_rate'],
        'New_Customers': revenue_annual['total_new_customers'],
        'Churned_Customers': revenue_annual['total_churned_customers'],
    })
    growth_summary.to_csv(
        f'{output_dir}/growth/growth_summary.csv', index=False)

    # Plot and save growth charts
    plt.figure(figsize=(14, 8))
    growth_fig = revenue_model.plot_growth_curves(figsize=(14, 8))
    growth_fig.savefig(f'{output_dir}/growth/growth_curves.png',
                       bbox_inches='tight', dpi=300)

    annual_metrics_fig = revenue_model.plot_annual_metrics(figsize=(14, 8))
    annual_metrics_fig.savefig(
        f'{output_dir}/growth/annual_metrics.png', bbox_inches='tight', dpi=300)

    segment_shares_fig = revenue_model.plot_customer_segment_shares(
        figsize=(14, 8))
    segment_shares_fig.savefig(
        f'{output_dir}/growth/segment_shares.png', bbox_inches='tight', dpi=300)

    # ---------- COST MODEL REPORTS ----------#

    # Save cost model data
    cost_monthly = cost_model.get_monthly_data()
    cost_annual = cost_model.get_annual_data()

    # Save monthly cost data
    cost_monthly[['date', 'total_headcount', 'total_compensation', 'total_cogs',
                 'total_marketing_expenses', 'total_sales_expenses',
                  'total_r_and_d_expenses', 'total_g_and_a_expenses',
                  'total_operating_expenses']].to_csv(
        f'{output_dir}/cost/monthly_cost_data.csv', index=False)

    # Save annual cost data
    cost_annual[['year', 'year_end_headcount', 'total_compensation',
                 'total_cogs', 'total_marketing_expenses', 'total_sales_expenses',
                 'total_r_and_d_expenses', 'total_g_and_a_expenses',
                 'total_operating_expenses', 'total_expenses']].to_csv(
        f'{output_dir}/cost/annual_cost_data.csv', index=False)

    # Save cost summary
    cost_summary = pd.DataFrame({
        'Year': cost_annual['year'],
        'Headcount': cost_annual['year_end_headcount'],
        'Compensation': cost_annual['total_compensation'],
        'COGS': cost_annual['total_cogs'],
        'Marketing': cost_annual['total_marketing_expenses'],
        'Sales': cost_annual['total_sales_expenses'],
        'R&D': cost_annual['total_r_and_d_expenses'],
        'G&A': cost_annual['total_g_and_a_expenses'],
        'OpEx': cost_annual['total_operating_expenses'],
        'Total_Expenses': cost_annual['total_expenses']
    })
    cost_summary.to_csv(f'{output_dir}/cost/cost_summary.csv', index=False)

    # Plot and save cost charts
    expense_breakdown_fig = cost_model.plot_expense_breakdown(figsize=(14, 8))
    expense_breakdown_fig.savefig(
        f'{output_dir}/cost/expense_breakdown.png', bbox_inches='tight', dpi=300)

    headcount_growth_fig = cost_model.plot_headcount_growth(figsize=(14, 8))
    headcount_growth_fig.savefig(
        f'{output_dir}/cost/headcount_growth.png', bbox_inches='tight', dpi=300)


def print_profitability_analysis(financial_model, annual_data):
    """Print profitability analysis for a financial model"""
    monthly_data = financial_model.get_monthly_data()

    # Find month of profitability
    profitable_month_data = monthly_data[monthly_data['ebitda'] > 0]

    if len(profitable_month_data) > 0:
        profitable_month = profitable_month_data['month_number'].min()
        profitable_year = (profitable_month // 12) + 1
        profitable_month_in_year = (profitable_month % 12) or 12
        print(f"\nProfitability Analysis:")
        print(
            f"Month of profitability: Month {profitable_month} (Year {profitable_year}, Month {profitable_month_in_year})")
    else:
        print(f"\nProfitability Analysis:")
        print(f"The company does not reach profitability within the 6-year forecast period")

    # Calculate total burn before profitability
    total_burn = monthly_data[monthly_data['ebitda'] < 0]['ebitda'].sum()
    print(f"Total burn before profitability: ${abs(total_burn)/1000000:.2f}M")

    # Calculate funding adequacy
    min_capital = monthly_data['capital'].min()
    if min_capital < 0:
        print(
            f"WARNING: Funding gap detected! Additional ${abs(min_capital)/1000000:.2f}M needed")
    else:
        print(
            f"Initial funding of ${monthly_data['capital'].iloc[0]/1000000:.2f}M is adequate. Minimum capital position: ${min_capital/1000000:.2f}M")

    # Calculate terminal metrics (Year 6)
    terminal_year = annual_data.iloc[-1]
    terminal_revenue = terminal_year['annual_revenue']
    terminal_ebitda = terminal_year['annual_ebitda']
    terminal_ebitda_margin = terminal_year['annual_ebitda_margin']

    print(f"\nTerminal Metrics (Year 6):")
    print(f"Revenue: ${terminal_revenue/1000000:.2f}M")
    print(f"EBITDA: ${terminal_ebitda/1000000:.2f}M")
    print(f"EBITDA Margin: {terminal_ebitda_margin*100:.1f}%")

    # Calculate potential valuation
    print(f"\nPotential Valuation Estimates (Year 6):")
    print(f"At 5x Revenue multiple: ${(terminal_revenue*5)/1000000:.2f}M")
    print(f"At 8x Revenue multiple: ${(terminal_revenue*8)/1000000:.2f}M")
    print(f"At 12x EBITDA multiple: ${(terminal_ebitda*12)/1000000:.2f}M")
    print(f"At 18x EBITDA multiple: ${(terminal_ebitda*18)/1000000:.2f}M")


if __name__ == "__main__":
    main()
