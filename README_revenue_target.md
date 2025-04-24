# Revenue Target Optimization Tool

This tool helps you determine how many customers you need to onboard to reach a specific monthly revenue target by a certain month, based on your current cost structure and growth parameters.

## Overview

The Revenue Target Optimization Tool calculates the required customer growth multiplier needed to achieve a specific monthly revenue goal at a target month. It analyzes different growth scenarios and finds the optimal growth strategy that hits your revenue target.

## Features

- Finds the optimal customer acquisition strategy for a specific revenue target
- Calculates the number of customers required in each segment (Enterprise, Mid-Market, SMB)
- Provides a customer acquisition timeline showing growth by year
- Calculates the average monthly customer acquisition rate needed
- Generates visualizations of the customer growth and revenue trajectory
- Works with existing cost structures to ensure financial model accuracy

## Usage

```bash
python optimize_for_revenue_target.py [target_revenue] [target_month] [growth_profile] [initial_investment]
```

### Parameters

- `target_revenue`: Target monthly revenue in dollars (default: $5,000,000)
- `target_month`: Target month to achieve the revenue, 1-indexed (default: 36, or month 36)
- `growth_profile`: Base growth profile to start from - baseline, conservative, aggressive, hypergrowth (default: baseline)
- `initial_investment`: Initial capital investment in dollars (default: $20,000,000)

### Examples

```bash
# Find customer acquisition needed for $5M monthly revenue by month 36
python optimize_for_revenue_target.py 5000000 36

# Find customer acquisition needed for $10M monthly revenue by month 48 with aggressive profile
python optimize_for_revenue_target.py 10000000 48 aggressive

# Find customer acquisition needed for $2M monthly revenue by month 24 with a different investment amount
python optimize_for_revenue_target.py 2000000 24 baseline 15000000
```

## Output

The tool provides:

1. A detailed breakdown of customers required by segment
2. Year-by-year customer acquisition timeline
3. Average monthly customer acquisition rate needed overall and by segment
4. Visualizations in the `reports/optimization/` directory showing:
   - Customer growth projection to target month
   - Monthly revenue growth to target
   - Financial summary of the optimized model
   - Growth curves by customer segment

## Limitations

This tool uses a growth multiplier approach. For extremely aggressive targets that would require more than a 20x increase over the baseline growth rate, you may need to consider:

1. More fundamental changes to the S-curve parameters in the config files
2. Adjusting segment-specific growth strategies
3. Exploring multiple growth scenarios with different profiles

## Integration with the Financial Model

The optimization script is fully integrated with the existing financial model, so it takes into account:

- Customer acquisition rates by segment
- Pricing by segment
- Churn rates
- Cost structures for determining profitability and runway