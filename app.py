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


def run_integrated_financial_model(initial_investment=20000000):
    """
    Run an integrated financial model for an AI SaaS company

    Parameters:
    -----------
    initial_investment : float, optional
        Initial investment amount in dollars

    Returns:
    --------
    tuple : (financial_model, revenue_model, cost_model)
        The three model objects used in the analysis
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
            'cloud_hosting': 0.0,  # 18% - higher for AI compute
            'customer_support': 0.0,  # 8% for support
            'third_party_apis': 0.0,  # 6% for third-party AI/ML APIs
            'professional_services': 0.0,  # 3% for PS delivery
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
                'starting_count': 1,
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
                'starting_count': 2,
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
            'commission_rate': 0.0,  # 12% commission on new ARR
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
            'office_and_facilities': 0,  # Per month base cost
            'per_employee_office_cost': 0,  # Per employee per month
            'software_and_tools': 0,  # Per employee per month
            'legal_and_accounting': 0,  # Per month base cost
            'insurance': 0,  # Per month base cost
        },

        # One-time and periodic expenses
        'one_time_expenses': {
            # Format: [month_idx, category, amount, description]
            'items': [
                [3, 'office', 0, 'Office setup and expansion'],
                [15, 'software', 0, 'Enterprise software licenses'],
                [17, 'research', 0, 'Major AI model training run'],
                [27, 'legal', 0, 'IP protection and legal work'],
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
        initial_investment=initial_investment
    )

    # Step 5: Run the integrated model
    financial_model.run_model()

    return financial_model, revenue_model, cost_model

# Execute the model


def main():
    # Run with $20M initial investment
    financial_model, revenue_model, cost_model = run_integrated_financial_model(
        20000000)

    # Create necessary directories
    import os
    os.makedirs('reports/combined', exist_ok=True)
    os.makedirs('reports/growth', exist_ok=True)
    os.makedirs('reports/cost', exist_ok=True)

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
