# AI GRC Financial Model

A comprehensive financial modeling tool for AI Governance, Risk, and Compliance (GRC) SaaS businesses. This model is designed to project financial performance from 2025 onwards, with special attention to the impact of AI regulations beginning in 2026.

## Overview

This financial model projects the growth, costs, and financial performance of an AI GRC SaaS business over a 6-year period. It incorporates customer acquisition, revenue recognition, cost structure, headcount planning, and capital requirements, with a focus on the unique dynamics of the AI GRC market.

## Key Features

- Multi-segment customer modeling (Enterprise, Mid-Market, SMB)
- S-curve based customer acquisition projections
- Detailed headcount and compensation modeling
- Cost structure appropriate for a Zurich-based SaaS company
- Unit economics calculations (LTV, CAC, payback periods)
- Capital requirement projections
- Break-even analysis
- Regulation-impact modeling
- Multiple growth profiles (baseline, conservative, aggressive, hypergrowth)

## Model Components

### Growth Model (`models/growth_model.py`)

The growth model projects customer acquisition and retention over time:

- **S-curve acquisition patterns**: Models the natural adoption curve of new technology
- **Segment-specific parameters**: Different parameters for Enterprise, Mid-Market, and SMB customers
- **Churn modeling**: Customer retention based on contract length and segment-specific churn rates
- **Annual price increases**: Built-in price escalation to model improving pricing power
- **Seasonality effects**: Accounts for seasonal variation in customer acquisition
- **Growth profiles**: Predefined acceleration/deceleration patterns for different scenarios

### Cost Model (`models/cost_model.py`)

The cost model projects all expenses associated with running the business:

- **Headcount planning**: Department-specific hiring plans with salary scales
- **Compensation modeling**: Includes base salary, benefits, payroll taxes, bonuses, and equity
- **COGS**: Cloud hosting, customer support, third-party APIs, and professional services
- **Marketing expenses**: Advertising, content creation, events, and partner marketing
- **Sales expenses**: Commission structures and sales enablement
- **R&D expenses**: Research tools, third-party research, and compute resources
- **G&A expenses**: Office costs, software tools, legal/accounting, and insurance
- **One-time expenses**: Major non-recurring costs (certifications, platform upgrades, etc.)

### Financial Model (`models/financial_model.py`)

The integrated financial model combines growth and cost projections:

- **Revenue recognition**: Monthly revenue and ARR calculations
- **Margin calculations**: Gross margin, EBITDA margin, etc.
- **Cash flow projections**: Monthly and annual cash flow
- **Capital position tracking**: Remaining capital by period
- **Runway calculations**: Cash runway under various scenarios
- **Break-even analysis**: Month of sustained profitability
- **Unit economics**: Customer acquisition cost (CAC), lifetime value (LTV), LTV/CAC ratio
- **Rule of 40**: Growth rate + profit margin (standard SaaS health metric)

## Key Assumptions

- **Start Date**: January 1, 2025
- **AI Regulations**: Major regulations go into effect in 2026 (Year 2)
- **Customer Segments**:
  - Enterprise: �175,000 ACV, 5% annual churn, 3-year contracts
  - Mid-Market: �40,000 ACV, 10% annual churn, 2-year contracts
  - SMB: �12,000 ACV, 15% annual churn, 1-year contracts
- **Zurich-based Company**:
  - Higher salary levels compared to global averages
  - Swiss cost structure (benefits, taxes, office costs)
  - European market focus
- **Capital Requirements**:
  - Baseline: �15-20M
  - Conservative: �20M+
  - Aggressive: �25M
  - Hypergrowth: �35M

## Configuration Files

### Revenue Configuration (`configs/revenue_config.json`)

Controls all aspects of customer acquisition and revenue:

```
{
  "start_date": "2025-01-01",      // Starting date for projections
  "projection_months": 72,         // Total months to project (6 years)
  "segments": ["Enterprise", "Mid-Market", "SMB"],  // Customer segments
  
  "initial_arr": {                 // Annual recurring revenue per customer
    "Enterprise": 350000,
    "Mid-Market": 120000,
    "SMB": 48000
  },
  
  "initial_customers": {           // Starting customers by segment
    "Enterprise": 2,
    "Mid-Market": 3,
    "SMB": 4
  },
  
  "contract_length": { ... },      // Length in years
  "churn_rates": { ... },          // Annual churn percentage
  "annual_price_increases": { ... }, // Annual price escalation
  "s_curve": { ... },              // Customer acquisition parameters
  "seasonality": { ... }           // Monthly seasonality factors
}
```

### Cost Configuration (`configs/cost_config.json`)

Controls all aspects of expenses and headcount:

```
{
  "start_date": "2025-01-01",
  "projection_months": 72,
  
  "cogs": { ... },                 // Cost of goods sold components
  
  "headcount": {                   // Department-specific headcount plans
    "engineering": {
      "starting_count": 6,         // Initial headcount (supports fractional)
      "growth_type": "step",       // Growth pattern (step, linear, s-curve)
      "growth_factors": { ... },   // Year-by-year growth multipliers
      "avg_salary": 150000         // Annual base salary
    },
    // Other departments...
  },
  
  "salary": { ... },               // Compensation components
  "marketing_expenses": { ... },   // Marketing budget allocation
  "marketing_efficiency": { ... }, // Year-by-year efficiency factors
  "sales_expenses": { ... },       // Sales cost components
  "r_and_d_expenses": { ... },     // R&D cost components
  "g_and_a_expenses": { ... },     // G&A fixed and variable costs
  "one_time_expenses": { ... }     // Non-recurring expenses
}
```

## Running the Model

### Basic Usage

Run the baseline model with default initial investment:

```bash
python app.py
```

### Growth Profiles

Run with different growth profiles:

```bash
# Conservative growth profile
python app.py --strategy profile --profile conservative --initial-investment 20000000

# Aggressive growth profile
python app.py --strategy profile --profile aggressive --initial-investment 25000000

# Hypergrowth profile
python app.py --strategy profile --profile hypergrowth --initial-investment 35000000
```

### Command Line Options

- `--initial-investment`: Initial capital amount (default: 5,000,000)
- `--strategy`: Business strategy (`baseline` or `profile`)
- `--profile`: Growth profile when using profile strategy (`conservative`, `baseline`, `aggressive`, `hypergrowth`)

## Interactive Dashboard

The model includes an interactive Streamlit dashboard for real-time adjustments and visualizations:

```bash
streamlit run streamlit_app.py
```

### Dashboard Features

- **Interactive S-Curve Editor**: Drag and modify growth curve parameters and see real-time changes
- **Financial Metrics**: Instantly view how parameter changes affect key metrics
- **Growth Profiles**: Apply predefined growth profiles or create custom ones with a global multiplier
- **Configuration Management**: Save, export, and import configurations
- **Visual Analysis**: Interactive charts for customers, revenue, expenses and unit economics

### Modifying Parameters

The dashboard allows you to modify:

1. **Growth Parameters**:
   - S-curve parameters (midpoint, steepness, max monthly acquisitions)
   - Segment-specific growth patterns
   - Seasonality factors
   - Global growth multiplier

2. **Financial Parameters**:
   - Initial investment
   - Initial customers and ARR by segment
   - Churn rates and contract lengths
   - Annual price increases

3. **Cost Parameters**:
   - Headcount growth by department
   - Salary levels
   - COGS percentages

## Output

The model generates several outputs:

1. **Console Summary**: Key metrics summary table with year-by-year projections
2. **Report Files**: CSV files with detailed projections saved to the `reports` directory
3. **Visualizations**: Charts and graphs saved as PNG files in the `reports` directory

### Key Output Metrics

- ARR (Annual Recurring Revenue)
- Revenue
- EBITDA and EBITDA Margin
- Customer counts
- Headcount
- CAC (Customer Acquisition Cost)
- LTV (Lifetime Value)
- LTV/CAC Ratio
- Rule of 40 Score
- Capital Position
- Break-even Point

## Calculating Unit Economics

The model uses the following approach to calculate key unit economics metrics:

- **CAC** (Customer Acquisition Cost): Total sales and marketing expenses divided by new customers acquired
- **LTV** (Lifetime Value): ARPU � Gross Margin � Customer Lifetime
  - Customer Lifetime: 1 / Churn Rate (capped at reasonable maximum)
- **LTV/CAC Ratio**: LTV divided by CAC
- **CAC Payback Period**: CAC / (Monthly contribution margin per customer)

## Interpreting Results

- **Break-even Point**: The month when the business achieves sustained profitability
- **Capital Position**: Remaining capital at the end of each period
- **Negative EBITDA**: Common in early years; check capital position to ensure sufficient runway
- **Rule of 40**: Combined growth rate and profit margin; >40 is considered strong for SaaS

## License

This financial model is proprietary and intended for internal use only.

## Acknowledgements

Financial model developed by Claude AI from Anthropic.