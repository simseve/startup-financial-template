# Specific Revenue Target Optimization Tool

This tool provides precise control for optimizing your customer acquisition strategy to reach a specific monthly revenue target in a given month, with more controlled and realistic growth patterns.

## Overview

The Specific Revenue Target Optimization Tool was created to solve the challenge of precisely hitting a revenue target (like $30M monthly revenue by month 60) while maintaining realistic and balanced growth across customer segments. Unlike the general revenue target optimizer, this specialized tool:

1. Uses segment-specific growth rates
2. Applies more sophisticated year-by-year tapering of growth
3. Includes fine-tuning to hit the revenue target with minimal error

## Usage

```bash
python optimize_for_specific_revenue.py [target_revenue] [target_month] [initial_investment]
```

### Parameters

- `target_revenue`: Target monthly revenue in dollars (default: $30,000,000)
- `target_month`: Target month to achieve the revenue, 1-indexed (default: 60)
- `initial_investment`: Initial capital investment in dollars (default: $20,000,000)

### Examples

```bash
# Target $30M monthly revenue by month 60 (default)
python optimize_for_specific_revenue.py

# Target $20M monthly revenue by month 48
python optimize_for_specific_revenue.py 20000000 48

# Target $40M monthly revenue by month 60 with $25M initial investment
python optimize_for_specific_revenue.py 40000000 60 25000000
```

## How It Works

The tool uses a more sophisticated approach than simple uniform scaling:

1. **Segment-Specific Growth**: Applies different growth multipliers to each customer segment
   - Enterprise: Higher multiplier (focus on high-value customers)
   - Mid-Market: Moderate multiplier
   - SMB: Lower multiplier (to prevent excessive growth)

2. **Year-by-Year Tapering**: Gradually reduces growth rates over time for more realistic growth
   - Year 1: Full multiplier strength
   - Years 2-5: Gradually decreasing multipliers
   - Year 6: Minimal additional growth

3. **Fine-Tuning Algorithm**: After initial growth strategy is applied, the tool measures the gap between achieved and target revenue, then applies a precise adjustment factor to hit the target revenue with minimal error

## Outputs

The tool provides:

1. A detailed breakdown of customers required by segment
2. Year-by-year customer acquisition timeline
3. Average monthly customer acquisition rate needed overall and by segment
4. Visualizations showing:
   - Customer growth projection to target month
   - Monthly revenue growth to target

## When to Use

Use this specialized tool when:

1. You need precise control over a specific revenue target
2. You're targeting higher revenue numbers (over $10M monthly)
3. You want more realistic, controlled growth across segments
4. You prefer segment-specific growth strategies over uniform scaling

For simpler cases or when integrated with the main app, you can continue using the regular revenue target optimization via `app.py --strategy revenue_target`.