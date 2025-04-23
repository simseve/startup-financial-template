import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from IPython.display import display

# Import our model classes
from models.cost_model import AISaaSCostModel
from models.growth_model import SaaSGrowthModel
from models.financial_model import SaaSFinancialModel

# Configure and run the integrated model


def run_integrated_financial_model(
    initial_investment=20000000,
    growth_profile="baseline",
    segment_multipliers=None,
    optimize_target=None,
    target_month=24
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

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model, optimization_results)
    """
    # Step 1: Configure Revenue Model
    revenue_config = {
        # Basic parameters
        'start_date': '2025-01-01',
        'projection_months': 72,  # 6 years
        'segments': ['Enterprise', 'Mid-Market', 'SMB'],

        # Initial ARR per customer by segment
        'initial_arr': {
            'Enterprise': 150000,  # $150K per enterprise customer
            'Mid-Market': 48000,   # $48K per mid-market customer
            'SMB': 12000,          # $12K per SMB customer
        },

        # Initial customer count by segment
        'initial_customers': {
            'Enterprise': 2,
            'Mid-Market': 1,
            'SMB': 2,
        },

        # Contract length in years by segment
        'contract_length': {
            'Enterprise': 2.0,  # 2-year contracts for enterprise
            'Mid-Market': 1.5,  # 18-month contracts for mid-market
            'SMB': 1.0,         # 1-year contracts for SMB
        },

        # Annual churn rates by segment
        'churn_rates': {
            'Enterprise': 0.08,  # 8% annual churn for enterprise
            'Mid-Market': 0.12,  # 12% annual churn for mid-market
            'SMB': 0.20,         # 20% annual churn for SMB
        },
        
        # Annual price increases by segment (as a decimal)
        'annual_price_increases': {
            'Enterprise': 0.05,  # 5% annual price increase for enterprise
            'Mid-Market': 0.04,  # 4% annual price increase for mid-market
            'SMB': 0.03,         # 3% annual price increase for SMB
        },

        # S-curve parameters for customer acquisition by segment/year
        's_curve': {
            'Enterprise': {
                # Year 1: Slower enterprise acquisition
                1: {'midpoint': 6, 'steepness': 0.5, 'max_monthly': 3},
                # Year 2: More enterprise traction
                2: {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 5},
                # Year 3: Stronger enterprise growth
                3: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 7},
                # Year 4: Continued enterprise acceleration
                4: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 10},
                # Year 5: Peak enterprise growth
                5: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 12},
                # Year 6: Stabilizing enterprise growth
                6: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 15},
            },
            'Mid-Market': {
                # Year 1: Starting with mid-market focus
                1: {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 8},
                # Year 2: Strong mid-market growth
                2: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 12},
                # Year 3: Accelerated mid-market
                3: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 18},
                # Year 4: Peak mid-market growth
                4: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 25},
                # Year 5: Strong but stabilizing mid-market
                5: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 30},
                # Year 6: Continued steady growth
                6: {'midpoint': 6, 'steepness': 0.6, 'max_monthly': 35},
            },
            'SMB': {
                # Year 1: Initial SMB acquisition
                1: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 15},
                # Year 2: Growing SMB focus
                2: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 25},
                # Year 3: Rapid SMB growth with self-service
                3: {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 40},
                # Year 4: Mass market penetration
                4: {'midpoint': 6, 'steepness': 1.0, 'max_monthly': 60},
                # Year 5: Continued SMB growth
                5: {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 80},
                # Year 6: Maturing SMB growth
                6: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 100},
            },
        },

        # Monthly seasonality factors (1.0 = average)
        'seasonality': {
            1: 0.85,   # January: Slow after holidays
            2: 0.95,   # February: Gradual recovery
            3: 1.05,   # March: End of Q1 push
            4: 1.0,    # April: Start of Q2
            5: 1.0,    # May: Steady
            6: 1.15,   # June: End of Q2 push
            7: 0.9,    # July: Summer slowdown
            8: 0.85,   # August: Summer slowdown
            9: 1.05,   # September: Back to business
            10: 1.1,   # October: Start of Q4
            11: 1.0,   # November: Pre-holiday
            12: 1.1,   # December: End of year push
        },
    }

    # Step 2: Configure Cost Model
    cost_config = {
        # Basic parameters
        'start_date': '2025-01-01',
        'projection_months': 72,  # 6 years

        # COGS Assumptions (% of ARR)
        'cogs': {
            'cloud_hosting': 0.18,  # 18% - higher for AI compute
            'customer_support': 0.08,  # 8% for support
            'third_party_apis': 0.06,  # 6% for third-party AI/ML APIs
            'professional_services': 0.03,  # 3% for PS delivery
        },

        # Headcount Assumptions (starting headcount, with growth factors)
        'headcount': {
            # Engineering including ML/AI specialists
            'engineering': {
                'starting_count': 10,  # Higher engineering count for AI company
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.5,  # Year 1: 50% growth
                    2: 1.8,  # Year 2: 80% growth
                    3: 1.6,  # Year 3: 60% growth
                    4: 1.4,  # Year 4: 40% growth
                    5: 1.3,  # Year 5: 30% growth
                    6: 1.2,  # Year 6: 20% growth
                },
                'avg_salary': 160000,  # Higher for AI engineers
            },
            # Product Management
            'product': {
                'starting_count': 3,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.5,
                    2: 1.7,
                    3: 1.5,
                    4: 1.4,
                    5: 1.3,
                    6: 1.2,
                },
                'avg_salary': 170000,
            },
            # Sales team
            'sales': {
                'starting_count': 4,
                'growth_type': 'step',
                'growth_factors': {
                    1: 2.0,  # Sales grows faster early
                    2: 1.8,
                    3: 1.6,
                    4: 1.4,
                    5: 1.3,
                    6: 1.2,
                },
                # Base salary (not including commissions)
                'avg_salary': 180000,
            },
            # Marketing team
            'marketing': {
                'starting_count': 3,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.8,
                    2: 1.7,
                    3: 1.5,
                    4: 1.4,
                    5: 1.3,
                    6: 1.2,
                },
                'avg_salary': 120000,
            },
            # Customer success/support
            'customer_success': {
                'starting_count': 2,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.5,
                    2: 1.7,
                    3: 1.6,
                    4: 1.4,
                    5: 1.3,
                    6: 1.2,
                },
                'avg_salary': 110000,
            },
            # G&A (General and Administrative)
            'g_and_a': {
                'starting_count': 4,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.4,
                    2: 1.6,
                    3: 1.4,
                    4: 1.3,
                    5: 1.2,
                    6: 1.1,
                },
                'avg_salary': 120000,
            },
            # Research team (specific to AI company)
            'research': {
                'starting_count': 4,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.5,
                    2: 1.7,
                    3: 1.5,
                    4: 1.4,
                    5: 1.3,
                    6: 1.2,
                },
                'avg_salary': 200000,  # Higher for AI researchers
            },
        },

        # Salary & Benefits Assumptions
        'salary': {
            'annual_increase': 0.05,  # 5% annual salary increase
            'benefits_multiplier': 1.28,  # Benefits are 28% of base salary
            'payroll_tax_rate': 0.09,  # 9% payroll taxes
            'bonus_rate': 0.15,  # 15% annual bonus (higher for tech)
            'equity_compensation': 0.20,  # 20% of salary as equity comp
        },

        # Marketing & Other Expenses
        'marketing_expenses': {
            # Non-headcount marketing expenses (% of ARR)
            'paid_advertising': 0.25,  # 25% of ARR
            'content_creation': 0.10,  # 10% of ARR
            'events_and_pr': 0.08,  # 8% of ARR
            'partner_marketing': 0.07,  # 7% of ARR
        },

        # Growth multiplier for marketing as company scales
        'marketing_efficiency': {
            1: 1.0,   # Year 1: Base level
            2: 0.92,  # Year 2: 8% more efficient
            3: 0.85,  # Year 3: 15% more efficient
            4: 0.80,  # Year 4: 20% more efficient
            5: 0.75,  # Year 5: 25% more efficient
            6: 0.70,  # Year 6: 30% more efficient
        },

        # Sales expenses
        'sales_expenses': {
            'commission_rate': 0.15,  # 15% commission on new ARR
            'tools_and_enablement': 0.05,  # 5% of ARR
        },

        # R&D expenses (beyond headcount) - higher for AI company
        'r_and_d_expenses': {
            'cloud_compute_for_training': 0.18,  # 18% of ARR
            'research_tools_and_data': 0.12,  # 12% of ARR
            'third_party_research': 0.08,  # 8% of ARR
        },

        # General & Admin expenses
        'g_and_a_expenses': {
            'office_and_facilities': 50000,  # Per month base cost
            'per_employee_office_cost': 1500,  # Per employee per month
            'software_and_tools': 1000,  # Per employee per month
            'legal_and_accounting': 25000,  # Per month base cost
            'insurance': 15000,  # Per month base cost
        },

        # One-time and periodic expenses
        'one_time_expenses': {
            # Format: [month_idx, category, amount, description]
            'items': [
                [3, 'office', 750000, 'Office setup and expansion'],
                [9, 'marketing', 500000, 'Major product launch campaign'],
                [15, 'software', 350000, 'Enterprise software licenses'],
                [17, 'research', 1200000, 'Major AI model training run'],
                [21, 'marketing', 600000, 'Industry conference sponsorship'],
                [24, 'office', 400000, 'Office expansion'],
                [27, 'legal', 300000, 'IP protection and legal work'],
                [36, 'office', 800000, 'New office location setup'],
                [41, 'research', 1500000, 'Advanced AI model development'],
                [48, 'infrastructure', 1000000, 'Major infrastructure upgrade'],
            ]
        }
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
        revenue_model = revenue_model.apply_custom_segment_profiles(segment_multipliers)
    elif optimize_target is not None:
        # Prepare fixed costs for optimization
        # Extract average monthly fixed costs from the cost model
        monthly_fixed_costs = {
            'salary': cost_config['headcount']['engineering']['starting_count'] * cost_config['headcount']['engineering']['avg_salary'] / 12,
            'marketing': cost_config['marketing_expenses']['paid_advertising'] * 100000 / 12,  # Initial marketing budget
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
        import os
        os.makedirs('reports/optimization', exist_ok=True)
        import pandas as pd
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
    print(f"Optimizing for breakeven at month {target_month} using {growth_profile} profile as base...")
    
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
    print(f"Optimizing for Series B qualification at month {target_month} using {growth_profile} profile as base...")
    
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
        raise ValueError(f"Invalid profile: {growth_profile}. Must be one of: {', '.join(valid_profiles)}")
    
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
    custom_model = revenue_model.apply_dynamic_growth_strategy(segment_year_multipliers)
    
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
    custom_model = revenue_model.apply_monthly_growth_pattern(segment_month_multipliers)
    
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
    import os
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
            'Year 6 EBITDA': [annual['annual_ebitda'].iloc[-1] / 1000000],  # $M
            'Breakeven Month': [breakeven_month],
            '5x Revenue Valuation': [annual['annual_revenue'].iloc[-1] * 5 / 1000000]  # $M
        })
        comparison_table = pd.concat([comparison_table, new_row], ignore_index=True)
    
    comparison_table.to_csv('reports/scenarios/scenario_comparison.csv', index=False)
    print("Growth scenario comparison saved to reports/scenarios/growth_scenarios_comparison.png")
    
    return scenario_results

def compare_optimization_targets():
    """
    Compare different optimization targets (breakeven vs Series B qualification)
    
    Returns:
    --------
    tuple : (breakeven_results, series_b_results)
    """
    import os
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
    
    comparison.to_csv('reports/optimization/optimization_comparison.csv', index=False)
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
            1: 2.2,  # Very strong focus in year 1 (early adopters with resources for compliance)
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
    import os
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
    import pandas as pd
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
        comparison_metrics = pd.concat([comparison_metrics, new_row], ignore_index=True)
    
    comparison_metrics.to_csv('reports/strategies/dynamic_strategies_metrics.csv', index=False)
    print("Dynamic growth strategies comparison saved to reports/strategies/")
    
    return strategies, comparison_metrics

def main():
    # Create necessary directories
    import os
    os.makedirs('reports/combined', exist_ok=True)
    os.makedirs('reports/growth', exist_ok=True)
    os.makedirs('reports/cost', exist_ok=True)
    os.makedirs('reports/optimization', exist_ok=True)
    os.makedirs('reports/strategies', exist_ok=True)

    # First, compare our new dynamic growth strategies
    print("Comparing dynamic growth strategies...")
    strategies, metrics = compare_dynamic_strategies(initial_investment=20000000)
    print("\nDynamic Growth Strategy Comparison:")
    print(metrics)
    
    # Run optimization for Series B qualification at month 36
    print("\nOptimizing for Series B qualification at month 36...")
    financial_model, revenue_model, cost_model, optimization_results = optimize_for_series_b(
        target_month=36,
        initial_investment=20000000
    )

    # Print optimization results
    if optimization_results:
        print(f"\nOptimization Results:")
        print(f"Target: {optimization_results['target']} by month {optimization_results['target_month']}")
        if optimization_results['achieved_month'] is not None:
            print(f"Achieved in month: {optimization_results['achieved_month']}")
        else:
            print(f"Target not achieved within the projection period")
        print(f"Growth multiplier applied: {optimization_results['growth_multiplier']:.2f}x")

        # Save optimization results
        optimization_df = pd.DataFrame({
            'Target': [optimization_results['target']],
            'Target Month': [optimization_results['target_month']],
            'Achieved Month': [optimization_results['achieved_month']],
            'Growth Multiplier': [optimization_results['growth_multiplier']]
        })
        optimization_df.to_csv('reports/optimization/series_b_optimization.csv', index=False)
        
    # Compare different growth scenarios
    print("\nComparing different growth scenarios...")
    scenarios = run_growth_scenarios(initial_investment=20000000)
    
    # Compare optimization targets
    print("\nComparing optimization targets...")
    opt_comparison = compare_optimization_targets()

    # Get results
    annual_data = financial_model.get_annual_data()

    # Print key metrics
    print("Key Financial Metrics:")
    print(financial_model.get_key_metrics_table())

    # Save key metrics table
    metrics_table = financial_model.get_key_metrics_table()
    metrics_table.to_csv('reports/combined/key_metrics.csv')

    # ---------- COMBINED REPORTS ----------#

    # Plot and save key charts
    fig1 = financial_model.plot_financial_summary(figsize=(14, 8))
    fig1.savefig('reports/combined/financial_summary.png',
                 bbox_inches='tight', dpi=300)

    fig2 = financial_model.plot_break_even_analysis(figsize=(14, 8))
    fig2.savefig('reports/combined/break_even_analysis.png',
                 bbox_inches='tight', dpi=300)

    fig3 = financial_model.plot_runway_and_capital(figsize=(14, 8))
    fig3.savefig('reports/combined/runway_and_capital.png',
                 bbox_inches='tight', dpi=300)

    fig4 = financial_model.plot_unit_economics(figsize=(14, 8))
    fig4.savefig('reports/combined/unit_economics.png',
                 bbox_inches='tight', dpi=300)

    # Get monthly and annual data
    monthly_data = financial_model.get_monthly_data()

    # Save cashflow data
    monthly_data[['date', 'monthly_revenue', 'total_cogs', 'total_operating_expenses',
                 'gross_profit', 'ebitda', 'cash_flow', 'capital']].to_csv(
        'reports/combined/monthly_cashflow.csv', index=False)

    annual_cashflow = pd.DataFrame({
        'Year': financial_model.annual_data['year'],
        'Revenue': financial_model.annual_data['annual_revenue'],
        'COGS': financial_model.annual_data['annual_total_cogs'],
        'OpEx': financial_model.annual_data['annual_total_operating_expenses'],
        'EBITDA': financial_model.annual_data['annual_ebitda'],
        'EBITDA_Margin': financial_model.annual_data['annual_ebitda_margin'],
        'Year_End_Capital': financial_model.annual_data['year_end_capital']
    })
    annual_cashflow.to_csv('reports/combined/annual_cashflow.csv', index=False)

    # Save unit economics data
    unit_economics = pd.DataFrame({
        'Year': financial_model.annual_data['year'],
        'CAC': financial_model.annual_data['annual_avg_cac'],
        'LTV': financial_model.annual_data['annual_avg_ltv'],
        'LTV_CAC_Ratio': financial_model.annual_data['annual_ltv_cac_ratio'],
        'ARPU': monthly_data.groupby('year_number')['arpu'].mean().values,
        'Gross_Margin': financial_model.annual_data['annual_gross_margin']
    })
    unit_economics.to_csv('reports/combined/unit_economics.csv', index=False)

    # ---------- GROWTH MODEL REPORTS ----------#

    # Save growth model data
    revenue_monthly = revenue_model.get_monthly_data()
    revenue_annual = revenue_model.get_annual_data()

    # Save monthly growth data
    revenue_monthly[['date', 'total_arr', 'total_customers', 'total_new_customers',
                    'total_churned_customers']].to_csv(
        'reports/growth/monthly_data.csv', index=False)

    # Save annual growth data
    revenue_annual[['year', 'total_ending_customers', 'total_ending_arr',
                   'total_new_customers', 'total_churned_customers',
                    'total_arr_growth_rate']].to_csv(
        'reports/growth/annual_data.csv', index=False)

    # Save growth summary
    growth_summary = pd.DataFrame({
        'Year': revenue_annual['year'],
        'Customers': revenue_annual['total_ending_customers'],
        'ARR': revenue_annual['total_ending_arr'],
        'Growth_Rate': revenue_annual['total_arr_growth_rate'],
        'New_Customers': revenue_annual['total_new_customers'],
        'Churned_Customers': revenue_annual['total_churned_customers'],
    })
    growth_summary.to_csv('reports/growth/growth_summary.csv', index=False)

    # Plot and save growth charts
    plt.figure(figsize=(14, 8))
    growth_fig = revenue_model.plot_growth_curves(figsize=(14, 8))
    growth_fig.savefig('reports/growth/growth_curves.png',
                       bbox_inches='tight', dpi=300)

    annual_metrics_fig = revenue_model.plot_annual_metrics(figsize=(14, 8))
    annual_metrics_fig.savefig(
        'reports/growth/annual_metrics.png', bbox_inches='tight', dpi=300)

    segment_shares_fig = revenue_model.plot_customer_segment_shares(
        figsize=(14, 8))
    segment_shares_fig.savefig(
        'reports/growth/segment_shares.png', bbox_inches='tight', dpi=300)

    # ---------- COST MODEL REPORTS ----------#

    # Save cost model data
    cost_monthly = cost_model.get_monthly_data()
    cost_annual = cost_model.get_annual_data()

    # Save monthly cost data
    cost_monthly[['date', 'total_headcount', 'total_compensation', 'total_cogs',
                 'total_marketing_expenses', 'total_sales_expenses',
                  'total_r_and_d_expenses', 'total_g_and_a_expenses',
                  'total_operating_expenses']].to_csv(
        'reports/cost/monthly_cost_data.csv', index=False)

    # Save annual cost data
    cost_annual[['year', 'year_end_headcount', 'total_compensation',
                 'total_cogs', 'total_marketing_expenses', 'total_sales_expenses',
                 'total_r_and_d_expenses', 'total_g_and_a_expenses',
                 'total_operating_expenses', 'total_expenses']].to_csv(
        'reports/cost/annual_cost_data.csv', index=False)

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
    cost_summary.to_csv('reports/cost/cost_summary.csv', index=False)

    # Plot and save cost charts
    expense_breakdown_fig = cost_model.plot_expense_breakdown(figsize=(14, 8))
    expense_breakdown_fig.savefig(
        'reports/cost/expense_breakdown.png', bbox_inches='tight', dpi=300)

    headcount_growth_fig = cost_model.plot_headcount_growth(figsize=(14, 8))
    headcount_growth_fig.savefig(
        'reports/cost/headcount_growth.png', bbox_inches='tight', dpi=300)

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
            f"Initial funding of $20M is adequate. Minimum capital position: ${min_capital/1000000:.2f}M")

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

    return financial_model, revenue_model, cost_model


if __name__ == "__main__":
    main()
