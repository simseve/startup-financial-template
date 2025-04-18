import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our model classes
from models.cost_model import AISaaSCostModel
from models.growth_model import SaaSGrowthModel
from models.financial_model import SaaSFinancialModel
from app import run_integrated_financial_model

def optimize_for_break_even():
    """
    Optimize the financial model to achieve break-even within 4 years
    while maximizing revenue. This optimization focuses on adjusting
    the S-curve parameters for customer acquisition.
    
    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model)
        The three optimized model objects used in the analysis
    """
    print("Running break-even optimization with accelerated S-curves...")
    
    # Step 1: Configure Revenue Model with optimized parameters
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

        # OPTIMIZED S-curve parameters for customer acquisition by segment/year
        # Increased steepness and max_monthly values in early years to accelerate growth
        's_curve': {
            'Enterprise': {
                # Year 1: More aggressive enterprise acquisition
                1: {'midpoint': 5, 'steepness': 0.7, 'max_monthly': 5},  # Increased from 0.5 to 0.7, max from 3 to 5
                # Year 2: Accelerated enterprise traction
                2: {'midpoint': 5, 'steepness': 0.8, 'max_monthly': 8},  # Increased from 0.6 to 0.8, max from 5 to 8
                # Year 3: Strong enterprise growth
                3: {'midpoint': 5, 'steepness': 0.9, 'max_monthly': 12}, # Increased from 0.7 to 0.9, max from 7 to 12
                # Year 4: Peak enterprise acceleration for break-even
                4: {'midpoint': 5, 'steepness': 0.9, 'max_monthly': 18}, # Increased from 0.7 to 0.9, max from 10 to 18
                # Year 5: Sustained enterprise growth
                5: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 15},
                # Year 6: Stabilizing enterprise growth
                6: {'midpoint': 6, 'steepness': 0.7, 'max_monthly': 18},
            },
            'Mid-Market': {
                # Year 1: Stronger mid-market focus to accelerate early revenue
                1: {'midpoint': 5, 'steepness': 0.8, 'max_monthly': 12},  # Increased from 0.6 to 0.8, max from 8 to 12
                # Year 2: Accelerated mid-market growth
                2: {'midpoint': 5, 'steepness': 0.9, 'max_monthly': 18},  # Increased from 0.7 to 0.9, max from 12 to 18
                # Year 3: Aggressive mid-market expansion
                3: {'midpoint': 5, 'steepness': 1.0, 'max_monthly': 30},  # Increased from 0.8 to 1.0, max from 18 to 30
                # Year 4: Peak mid-market growth for break-even
                4: {'midpoint': 5, 'steepness': 1.0, 'max_monthly': 40},  # Increased from 0.8 to 1.0, max from 25 to 40
                # Year 5: Strong but stabilizing mid-market
                5: {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 35},
                # Year 6: Continued steady growth
                6: {'midpoint': 6, 'steepness': 0.8, 'max_monthly': 40},
            },
            'SMB': {
                # Year 1: Faster SMB acquisition to drive early growth
                1: {'midpoint': 5, 'steepness': 0.9, 'max_monthly': 25},  # Increased from 0.7 to 0.9, max from 15 to 25
                # Year 2: Accelerated SMB focus
                2: {'midpoint': 5, 'steepness': 1.0, 'max_monthly': 40},  # Increased from 0.8 to 1.0, max from 25 to 40
                # Year 3: Rapid SMB growth with enhanced self-service
                3: {'midpoint': 5, 'steepness': 1.1, 'max_monthly': 65},  # Increased from 0.9 to 1.1, max from 40 to 65
                # Year 4: Massive market penetration for break-even
                4: {'midpoint': 5, 'steepness': 1.2, 'max_monthly': 90},  # Increased from 1.0 to 1.2, max from 60 to 90
                # Year 5: Strong continued SMB growth
                5: {'midpoint': 6, 'steepness': 1.0, 'max_monthly': 100},
                # Year 6: Maturing SMB growth
                6: {'midpoint': 6, 'steepness': 0.9, 'max_monthly': 120},
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

    # Step 2: Configure Cost Model - keeping the same as in app.py
    cost_config = {
        # Basic parameters
        'start_date': '2025-01-01',
        'projection_months': 72,  # 6 years

        # COGS Assumptions (% of ARR)
        'cogs': {
            'cloud_hosting': 0.18,  # 18% - higher for AI compute
            'customer_support': 0.08,  # 8% for support
            'third_party_apis': 0.02,  # 6% for third-party AI/ML APIs
            'professional_services': 0.01,  # 3% for PS delivery
        },

        # Headcount Assumptions (starting headcount, with growth factors)
        'headcount': {
            # Engineering including ML/AI specialists
            'engineering': {
                'starting_count': 5,  # Higher engineering count for AI company
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.0,  # Year 1: No change
                    2: 1.6,  # Year 2: 60% growth
                    3: 1.4,  # Year 3: 40% growth
                    4: 1.3,  # Year 4: 30% growth
                    5: 1.2,  # Year 5: 20% growth
                    6: 1.15,  # Year 6: 15% growth
                },
                'avg_salary': 100000,  # Higher for AI engineers
            },
            # Product Management
            'product': {
                'starting_count': 2,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.0,
                    2: 1.5,
                    3: 1.4,
                    4: 1.3,
                    5: 1.2,
                    6: 1.1,
                },
                'avg_salary': 130000,
            },
            # Sales team
            'sales': {
                'starting_count': 3,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.5,  # Sales grows faster early
                    2: 1.6,
                    3: 1.4,
                    4: 1.3,
                    5: 1.2,
                    6: 1.1,
                },
                # Base salary (not including commissions)
                'avg_salary': 130000,
            },
            # Marketing team
            'marketing': {
                'starting_count': 1,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.4,
                    2: 1.5,
                    3: 1.3,
                    4: 1.2,
                    5: 1.15,
                    6: 1.1,
                },
                'avg_salary': 60000,
            },
            # Customer success/support
            'customer_success': {
                'starting_count': 0,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.3,
                    2: 1.4,
                    3: 1.4,
                    4: 1.3,
                    5: 1.2,
                    6: 1.1,
                },
                'avg_salary': 90000,
            },
            # G&A (General and Administrative)
            'g_and_a': {
                'starting_count': 2,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.0,
                    2: 1.5,
                    3: 1.3,
                    4: 1.2,
                    5: 1.1,
                    6: 1.1,
                },
                'avg_salary': 80000,
            },
            # Research team (specific to AI company)
            'research': {
                'starting_count': 1,
                'growth_type': 'step',
                'growth_factors': {
                    1: 1.2,
                    2: 1.4,
                    3: 1.3,
                    4: 1.2,
                    5: 1.2,
                    6: 1.1,
                },
                'avg_salary': 150000,  # Higher for AI researchers
            },
        },

        # Salary & Benefits Assumptions
        'salary': {
            'annual_increase': 0.04,  # 4% annual salary increase
            'benefits_multiplier': 1.01,  # Benefits are 28% of base salary
            'payroll_tax_rate': 0.08,  # 8% payroll taxes
            'bonus_rate': 0.0,  # 12% annual bonus (higher for tech)
            'equity_compensation': 0.0,  # 18% of salary as equity comp
        },

        # Marketing & Other Expenses
        'marketing_expenses': {
            # Non-headcount marketing expenses (% of ARR)
            'paid_advertising': 0.18,  # 18% of ARR
            'content_creation': 0.06,  # 6% of ARR
            'events_and_pr': 0.04,  # 4% of ARR
            'partner_marketing': 0.03,  # 3% of ARR
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
            'commission_rate': 0.12,  # 12% commission on new ARR
            'tools_and_enablement': 0.03,  # 3% of ARR
        },

        # R&D expenses (beyond headcount) - higher for AI company
        'r_and_d_expenses': {
            'cloud_compute_for_training': 0.12,  # 12% of ARR
            'research_tools_and_data': 0.08,  # 8% of ARR
            'third_party_research': 0.04,  # 4% of ARR
        },

        # General & Admin expenses
        'g_and_a_expenses': {
            'office_and_facilities': 5000,  # Per month base cost
            'per_employee_office_cost': 0,  # Per employee per month
            'software_and_tools': 4000,  # Per employee per month
            'legal_and_accounting': 2000,  # Per month base cost
            'insurance': 1000,  # Per month base cost
        },

        # One-time and periodic expenses
        'one_time_expenses': {
            # Format: [month_idx, category, amount, description]
            'items': [
                [3, 'office', 0, 'Office setup and expansion'],
                [15, 'software', 0, 'Enterprise software licenses'],
                [17, 'research', 0, 'Major AI model training run'],
                [27, 'legal', 30000, 'IP protection and legal work'],
                [36, 'office', 0, 'New office location setup'],
                [41, 'research', 0, 'Advanced AI model development'],
                [48, 'infrastructure', 0, 'Major infrastructure upgrade'],
            ]
        }
    }

    # Step 3: Create models
    revenue_model = SaaSGrowthModel(revenue_config)
    cost_model = AISaaSCostModel(cost_config)

    # Step 4: Create integrated financial model
    financial_model = SaaSFinancialModel(
        revenue_model=revenue_model,
        cost_model=cost_model,
        initial_investment=20000000
    )

    # Step 5: Run the integrated model
    financial_model.run_model()
    
    # Step 6: Create necessary directories for output
    import os
    os.makedirs('reports/optimized', exist_ok=True)
    
    # Step 7: Analyze break-even point and revenue optimization results
    monthly_data = financial_model.get_monthly_data()
    annual_data = financial_model.get_annual_data()
    
    # Find month of profitability
    profitable_month_data = monthly_data[monthly_data['ebitda'] > 0]
    
    if len(profitable_month_data) > 0:
        profitable_month = profitable_month_data['month_number'].min()
        profitable_year = (profitable_month // 12) + 1
        profitable_month_in_year = (profitable_month % 12) or 12
        print(f"\nProfitability Analysis:")
        print(f"Month of profitability: Month {profitable_month} (Year {profitable_year}, Month {profitable_month_in_year})")
        
        # Check if we've met our goal of break-even in 4 years
        if profitable_year <= 4:
            print(f"SUCCESS: Break-even achieved in Year {profitable_year}, which meets the target of 4 years.")
        else:
            print(f"NOT MET: Break-even achieved in Year {profitable_year}, which does not meet the target of 4 years.")
    else:
        print(f"\nProfitability Analysis:")
        print(f"The company does not reach profitability within the 6-year forecast period")
    
    # Calculate total burn before profitability
    total_burn = monthly_data[monthly_data['ebitda'] < 0]['ebitda'].sum()
    print(f"Total burn before profitability: ${abs(total_burn)/1000000:.2f}M")
    
    # Calculate funding adequacy
    min_capital = monthly_data['capital'].min()
    if min_capital < 0:
        print(f"WARNING: Funding gap detected! Additional ${abs(min_capital)/1000000:.2f}M needed")
    else:
        print(f"Initial funding of $20M is adequate. Minimum capital position: ${min_capital/1000000:.2f}M")
    
    # Calculate terminal metrics (Year 6)
    terminal_year = annual_data.iloc[-1]
    terminal_revenue = terminal_year['annual_revenue']
    terminal_ebitda = terminal_year['annual_ebitda']
    terminal_ebitda_margin = terminal_year['annual_ebitda_margin']
    
    print(f"\nTerminal Metrics (Year 6):")
    print(f"Revenue: ${terminal_revenue/1000000:.2f}M")
    print(f"EBITDA: ${terminal_ebitda/1000000:.2f}M")
    print(f"EBITDA Margin: {terminal_ebitda_margin*100:.1f}%")
    
    # Save the S-curve optimization comparison
    fig = plt.figure(figsize=(15, 10))
    
    # Run the original model for comparison
    original_financial_model, _, _ = run_integrated_financial_model(20000000)
    original_monthly_data = original_financial_model.get_monthly_data()
    original_annual_data = original_financial_model.get_annual_data()
    
    # Compare revenue growth between original and optimized models
    plt.subplot(211)
    plt.plot(monthly_data['month_number'], monthly_data['monthly_revenue']/1000000, 
             label='Optimized Revenue ($M)', linewidth=2)
    plt.plot(original_monthly_data['month_number'], original_monthly_data['monthly_revenue']/1000000, 
             label='Original Revenue ($M)', linewidth=2, linestyle='--')
    
    # Add break-even points
    original_be_month = original_monthly_data[original_monthly_data['ebitda'] > 0]['month_number'].min()
    optimized_be_month = monthly_data[monthly_data['ebitda'] > 0]['month_number'].min()
    
    if not pd.isna(original_be_month):
        plt.axvline(x=original_be_month, color='blue', linestyle='--', alpha=0.7)
        plt.text(original_be_month+1, plt.ylim()[1]*0.9, f'Original Break-even: Month {original_be_month}',
                 color='blue', backgroundcolor='white', alpha=0.8)
    
    if not pd.isna(optimized_be_month):
        plt.axvline(x=optimized_be_month, color='green', linestyle='--', alpha=0.7)
        plt.text(optimized_be_month+1, plt.ylim()[1]*0.8, f'Optimized Break-even: Month {optimized_be_month}',
                 color='green', backgroundcolor='white', alpha=0.8)
    
    # Add year markers
    for year in range(2, 7):
        month = (year - 1) * 12 + 1
        plt.axvline(x=month, color='gray', linestyle='--', alpha=0.3)
        plt.text(month, plt.ylim()[1]*0.95, f'Year {year}', ha='center', 
                 va='top', backgroundcolor='white', alpha=0.8)
    
    plt.title('Revenue Comparison: Original vs. Optimized Model', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Monthly Revenue ($M)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare customer acquisition between original and optimized models
    plt.subplot(212)
    plt.plot(monthly_data['month_number'], monthly_data['total_customers'], 
             label='Optimized Customers', linewidth=2)
    plt.plot(original_monthly_data['month_number'], original_monthly_data['total_customers'], 
             label='Original Customers', linewidth=2, linestyle='--')
    
    # Add year markers
    for year in range(2, 7):
        month = (year - 1) * 12 + 1
        plt.axvline(x=month, color='gray', linestyle='--', alpha=0.3)
        plt.text(month, plt.ylim()[1]*0.95, f'Year {year}', ha='center', 
                 va='top', backgroundcolor='white', alpha=0.8)
    
    plt.title('Customer Growth Comparison: Original vs. Optimized Model', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Total Customers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/optimized/s_curve_optimization_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save comparison metrics to CSV
    comparison_metrics = pd.DataFrame({
        'Year': range(1, 7),
        'Original_Revenue_M': original_annual_data['annual_revenue'] / 1000000,
        'Optimized_Revenue_M': annual_data['annual_revenue'] / 1000000,
        'Revenue_Improvement_%': (annual_data['annual_revenue'] / original_annual_data['annual_revenue'] - 1) * 100,
        'Original_EBITDA_M': original_annual_data['annual_ebitda'] / 1000000,
        'Optimized_EBITDA_M': annual_data['annual_ebitda'] / 1000000,
        'Original_Customers': original_annual_data['year_end_customers'],
        'Optimized_Customers': annual_data['year_end_customers'],
        'Customer_Growth_%': (annual_data['year_end_customers'] / original_annual_data['year_end_customers'] - 1) * 100
    })
    
    # Add break-even indicators
    original_be_year = (original_be_month // 12) + 1 if not pd.isna(original_be_month) else ">6"
    optimized_be_year = (optimized_be_month // 12) + 1 if not pd.isna(optimized_be_month) else ">6"
    
    comparison_metrics.to_csv('reports/optimized/s_curve_optimization_metrics.csv', index=False)
    
    # Create a more detailed monthly comparison and save it
    monthly_comparison = pd.DataFrame({
        'Month': monthly_data['month_number'],
        'Original_Revenue_M': original_monthly_data['monthly_revenue'] / 1000000,
        'Optimized_Revenue_M': monthly_data['monthly_revenue'] / 1000000,
        'Original_EBITDA_M': original_monthly_data['ebitda'] / 1000000,
        'Optimized_EBITDA_M': monthly_data['ebitda'] / 1000000,
        'Original_Customers': original_monthly_data['total_customers'],
        'Optimized_Customers': monthly_data['total_customers']
    })
    
    monthly_comparison.to_csv('reports/optimized/s_curve_optimization_results.csv', index=False)
    
    print(f"\nBreak-even Comparison:")
    print(f"Original model break-even: Year {original_be_year}")
    print(f"Optimized model break-even: Year {optimized_be_year}")
    
    print(f"\nRevenue Optimization Results:")
    for i, year in enumerate(range(1, 7)):
        rev_improvement = comparison_metrics.loc[i, 'Revenue_Improvement_%']
        print(f"Year {year}: ${comparison_metrics.loc[i, 'Optimized_Revenue_M']:.2f}M revenue " +
              f"(+{rev_improvement:.1f}% vs. original)")
    
    return financial_model, revenue_model, cost_model

if __name__ == "__main__":
    financial_model, revenue_model, cost_model = optimize_for_break_even()
    
    # Generate additional detailed plots
    fig1 = financial_model.plot_break_even_analysis(figsize=(14, 8))
    fig1.savefig('reports/optimized/break_even_analysis.png', dpi=300, bbox_inches='tight')
    
    fig2 = financial_model.plot_financial_summary(figsize=(14, 8))
    fig2.savefig('reports/optimized/financial_summary.png', dpi=300, bbox_inches='tight')
    
    fig3 = revenue_model.plot_growth_curves(figsize=(14, 10))
    fig3.savefig('reports/optimized/growth_curves.png', dpi=300, bbox_inches='tight')