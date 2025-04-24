#!/usr/bin/env python
import sys
from app import optimize_for_series_b

# Default values
target_month = 36
growth_profile = "aggressive"
initial_investment = 20000000

# Parse command line arguments
if len(sys.argv) > 1:
    target_month = int(sys.argv[1])
if len(sys.argv) > 2:
    growth_profile = sys.argv[2]
if len(sys.argv) > 3:
    initial_investment = float(sys.argv[3])

# Print parameters
print(f"Running Series B optimization with:")
print(f"- Target month: {target_month}")
print(f"- Growth profile: {growth_profile}")
print(f"- Initial investment: ${initial_investment/1000000:.1f}M")
print("\n" + "="*50 + "\n")

# Run the optimization
financial_model, revenue_model, cost_model, optimization_results = optimize_for_series_b(
    target_month=target_month,
    growth_profile=growth_profile,
    initial_investment=initial_investment
)

# Print optimization results
if optimization_results:
    print("\nOptimization Results:")
    print(f"Target: {optimization_results['target']} by month {optimization_results['target_month']}")
    if optimization_results['achieved_month'] is not None:
        print(f"Achieved in month: {optimization_results['achieved_month']}")
    else:
        print(f"Target not achieved within the projection period")
    print(f"Growth multiplier applied: {optimization_results['growth_multiplier']:.2f}x")