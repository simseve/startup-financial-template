# 2025 Financial Model Reports

This directory contains the outputs of the 2025 SaaS financial model. Reports are organized into the following directories:

## Reports Structure

```
reports/
  combined/           # Combined financial model outputs
    annual_cashflow.csv           # Annual cashflow data
    break_even_analysis.png       # Chart showing break-even analysis
    financial_summary.png         # Summary of key financial metrics
    key_metrics.csv               # Table of key financial metrics
    monthly_cashflow.csv          # Monthly cashflow data
    runway_and_capital.png        # Capital position and runway analysis
    unit_economics.csv            # Unit economics metrics
    unit_economics.png            # Visual representation of unit economics

  cost/               # Cost model outputs
    annual_cost_data.csv          # Annual cost data
    cost_summary.csv              # Summary of cost metrics
    expense_breakdown.png         # Visualization of expense categories
    headcount_growth.png          # Headcount projections by department
    monthly_cost_data.csv         # Monthly cost data

  growth/             # Growth model outputs
    annual_data.csv              # Annual growth metrics
    annual_metrics.png           # Visual representation of annual metrics
    growth_curves.png            # Customer and ARR growth curves
    growth_summary.csv           # Summary of growth metrics
    monthly_data.csv             # Monthly growth data
    segment_shares.png           # Customer segment distribution
```

## Key Metrics

The model provides the following key insights:

1. **Growth Metrics**: Customer acquisition, churn, and ARR growth over time
2. **Cost Structure**: Headcount growth, compensation, and expense breakdown
3. **Unit Economics**: CAC, LTV, LTV/CAC ratio, and payback periods
4. **Financial Performance**: Revenue, EBITDA, margins, and cash flow analysis
5. **Capital Requirements**: Runway analysis and investment needs

## Running the Model

To regenerate these reports, run:

```
python app.py
```

This will execute the model with the configured parameters and save all outputs to the reports directory.
