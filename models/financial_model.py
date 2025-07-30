import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import FuncFormatter


class SaaSFinancialModel:
    """
    Integrated financial model for SaaS companies that combines revenue
    projections with cost modeling to generate comprehensive financial
    forecasts, unit economics, and cashflow analysis.
    """

    def __init__(self, revenue_model=None, cost_model=None, initial_investment=0):
        """
        Initialize the financial model with revenue and cost models

        Parameters:
        -----------
        revenue_model : SaaSGrowthModel object, optional
            Revenue projection model instance
        cost_model : AISaaSCostModel object, optional
            Cost model instance
        initial_investment : float, optional
            Initial capital investment amount
        """
        self.revenue_model = revenue_model
        self.cost_model = cost_model
        self.initial_investment = initial_investment

        # Initialize results dataframes
        self.monthly_data = None
        self.annual_data = None
        self._model_run = False

    def run_model(self):
        """
        Run the integrated financial model by combining revenue and cost data

        Returns:
        --------
        tuple : (monthly_data DataFrame, annual_data DataFrame)
        """
        # Ensure both revenue and cost models are provided
        if self.revenue_model is None:
            raise ValueError("Revenue model must be provided")
        if self.cost_model is None:
            raise ValueError("Cost model must be provided")

        # Run individual models if they haven't been run
        if not hasattr(self.revenue_model, 'monthly_data') or self.revenue_model.monthly_data is None:
            self.revenue_model.run_model()

        if not hasattr(self.cost_model, 'monthly_data') or self.cost_model.monthly_data is None:
            self.cost_model.run_model(self.revenue_model)

        # Get data from individual models
        revenue_monthly = self.revenue_model.get_monthly_data()
        cost_monthly = self.cost_model.get_monthly_data()

        # Combine into integrated financial model
        self._integrate_monthly_data(revenue_monthly, cost_monthly)

        # Generate annual summary
        self._generate_annual_summary()

        # Calculate cumulative metrics
        self._calculate_cumulative_metrics()

        self._model_run = True

        return self.monthly_data, self.annual_data

    def _integrate_monthly_data(self, revenue_data, cost_data):
        """
        Integrate revenue and cost data into a unified financial model

        Parameters:
        -----------
        revenue_data : pandas.DataFrame
            Monthly revenue model data
        cost_data : pandas.DataFrame
            Monthly cost model data
        """
        # Initialize combined dataframe
        self.monthly_data = pd.DataFrame({
            'date': revenue_data['date'],
            'year': revenue_data['year'],
            'month': revenue_data['month'],
            'month_number': revenue_data['month_number'],
            'year_number': revenue_data['year_number'],
        })

        # Add revenue metrics
        revenue_columns = ['total_arr', 'total_customers',
                           'total_new_customers', 'total_churned_customers']
        for col in revenue_columns:
            if col in revenue_data.columns:
                self.monthly_data[col] = revenue_data[col]

        # Calculate monthly revenue (ARR / 12)
        self.monthly_data['monthly_revenue'] = self.monthly_data['total_arr'] / 12

        # Add cost metrics
        cost_columns = ['total_cogs', 'total_compensation', 'total_marketing_expenses',
                        'total_sales_expenses', 'total_r_and_d_expenses', 'total_g_and_a_expenses',
                        'one_time_expenses', 'total_operating_expenses', 'total_headcount']
        for col in cost_columns:
            if col in cost_data.columns:
                self.monthly_data[col] = cost_data[col]

        # Calculate financial metrics
        self.monthly_data['gross_profit'] = self.monthly_data['monthly_revenue'] - \
            self.monthly_data['total_cogs']
        
        # Prevent division by zero in gross margin calculation
        self.monthly_data['gross_margin'] = np.where(
            self.monthly_data['monthly_revenue'] > 0,
            self.monthly_data['gross_profit'] / self.monthly_data['monthly_revenue'],
            0  # Default to 0 margin when revenue is 0
        )
        
        self.monthly_data['ebitda'] = self.monthly_data['gross_profit'] - \
            self.monthly_data['total_operating_expenses']
        
        # Prevent division by zero in EBITDA margin calculation
        self.monthly_data['ebitda_margin'] = np.where(
            self.monthly_data['monthly_revenue'] > 0,
            self.monthly_data['ebitda'] / self.monthly_data['monthly_revenue'],
            0  # Default to 0 margin when revenue is 0
        )

        # Initial capital tracking
        self.monthly_data['cash_flow'] = self.monthly_data['ebitda']
        self.monthly_data.loc[0, 'capital'] = self.initial_investment

        # Calculate runway and cash position
        for i in range(1, len(self.monthly_data)):
            previous_capital = self.monthly_data.loc[i-1, 'capital']
            current_cash_flow = self.monthly_data.loc[i, 'cash_flow']
            self.monthly_data.loc[i,
                                  'capital'] = previous_capital + current_cash_flow

        # Calculate burn rate (negative cash flow)
        self.monthly_data['burn_rate'] = np.where(self.monthly_data['cash_flow'] < 0,
                                                  -self.monthly_data['cash_flow'], 0)

        # Calculate runway in months
        self.monthly_data['runway_months'] = np.nan
        
        # Define a large but finite value for "infinite" runway
        # Using a large finite number instead of inf to avoid numerical issues
        max_runway_months = 10 * 12  # 10 years as practical "infinity"
        
        for i in range(len(self.monthly_data)):
            if self.monthly_data.loc[i, 'cash_flow'] < 0:
                remaining_capital = max(0, self.monthly_data.loc[i, 'capital'])
                current_burn = -self.monthly_data.loc[i, 'cash_flow']
                
                if current_burn > 0 and remaining_capital > 0:
                    # Calculate runway with upper bound
                    runway = remaining_capital / current_burn
                    self.monthly_data.loc[i, 'runway_months'] = min(runway, max_runway_months)
                else:
                    # Handle edge case of zero burn rate
                    self.monthly_data.loc[i, 'runway_months'] = max_runway_months
            else:
                # Positive cash flow means sustainable operations
                self.monthly_data.loc[i, 'runway_months'] = max_runway_months

        # Calculate CAC
        self.monthly_data['sales_marketing_expense'] = (
            self.monthly_data['total_marketing_expenses'] +
            self.monthly_data['total_sales_expenses']
        )
        self.monthly_data['cac'] = np.nan

        # Calculate CAC on a rolling 3-month basis
        for i in range(3, len(self.monthly_data)):
            new_customers = self.monthly_data.loc[i-2:i, 'total_new_customers'].sum()
            sm_expenses = self.monthly_data.loc[i-2:i, 'sales_marketing_expense'].sum()
            
            # Set a minimum number of customers to avoid division by very small numbers
            min_customers_for_calculation = 1
            
            if new_customers >= min_customers_for_calculation:
                self.monthly_data.loc[i, 'cac'] = sm_expenses / new_customers
            else:
                # Use previous CAC or a reasonable default if no customers acquired
                if i > 3 and not pd.isna(self.monthly_data.loc[i-1, 'cac']):
                    self.monthly_data.loc[i, 'cac'] = self.monthly_data.loc[i-1, 'cac']
                else:
                    # Default CAC based on industry average or business model
                    self.monthly_data.loc[i, 'cac'] = 5000  # Reasonable default

        # Calculate LTV (simpler version) with division by zero protection
        self.monthly_data['arpu'] = np.where(
            self.monthly_data['total_customers'] > 0,
            self.monthly_data['total_arr'] / self.monthly_data['total_customers'],
            0  # Default to 0 ARPU when no customers
        )
        # Average monthly gross margin per customer (with safeguard)
        self.monthly_data['monthly_margin_per_customer'] = (
            self.monthly_data['gross_margin'] * self.monthly_data['arpu'] / 12
        )

        # Use churn rates from revenue model to calculate customer lifetime
        # Get min churn rate from the revenue model if available
        if hasattr(self.revenue_model, 'config') and 'churn_rates' in self.revenue_model.config:
            # Find the minimum churn rate from all segments (excluding _comment)
            churn_values = [v for k, v in self.revenue_model.config['churn_rates'].items() if k != "_comment"]
            min_annual_churn_rate = min(churn_values)
            # Average churn rate across all segments
            avg_churn_rate = sum(churn_values) / len(churn_values)
        else:
            # Default values if not available
            min_annual_churn_rate = 0.08  # Default minimum churn
            avg_churn_rate = 0.15  # Default average churn
            
        # Calculate max lifetime based on min churn (inversely related)
        max_lifetime_months = int(12 / min_annual_churn_rate)
        
        # Initialize customer_lifetime_months with default values
        self.monthly_data['customer_lifetime_months'] = 1 / (avg_churn_rate / 12)
        
        if 'total_churned_customers' in self.monthly_data.columns and 'total_customers' in self.monthly_data.columns:
            for i in range(12, len(self.monthly_data), 12):  # Annual calculation
                annual_churned = self.monthly_data.loc[i - 11:i, 'total_churned_customers'].sum()
                avg_customers = self.monthly_data.loc[i - 11:i, 'total_customers'].mean()
                
                # Only calculate if we have meaningful customer data
                min_customers_for_calculation = 5
                if avg_customers >= min_customers_for_calculation:
                    annual_churn_rate = annual_churned / avg_customers
                    
                    # Apply minimum churn rate to avoid infinite lifetime
                    if annual_churn_rate < min_annual_churn_rate:
                        annual_churn_rate = min_annual_churn_rate
                        
                    # Calculate lifetime in months
                    lifetime_months = 1 / (annual_churn_rate / 12)
                    
                    # Cap lifetime at maximum value
                    lifetime_months = min(lifetime_months, max_lifetime_months)
                    
                    # Apply to all months in this year
                    self.monthly_data.loc[i-11:i, 'customer_lifetime_months'] = lifetime_months
                else:
                    # Use default lifetime for early months with few customers
                    default_lifetime = min(1 / (min_annual_churn_rate / 12), max_lifetime_months)
                    self.monthly_data.loc[i-11:i, 'customer_lifetime_months'] = default_lifetime

        # Calculate LTV with reasonable bounds
        # Set a maximum LTV cap to prevent unrealistic values
        annual_arpu = self.monthly_data['arpu']
        max_ltv_multiplier = 5  # Maximum LTV can be 5x annual ARPU - more realistic for SaaS
        
        self.monthly_data['ltv'] = np.minimum(
            self.monthly_data['monthly_margin_per_customer'] * self.monthly_data['customer_lifetime_months'],
            annual_arpu * max_ltv_multiplier  # Cap LTV at reasonable multiple of ARPU
        )

        # Calculate LTV/CAC ratio with division by zero protection
        max_ratio = 10  # Maximum realistic LTV/CAC ratio for SaaS
        
        # Where CAC is available and positive, calculate ratio with cap
        self.monthly_data['ltv_cac_ratio'] = np.where(
            (self.monthly_data['cac'].notna()) & (self.monthly_data['cac'] > 0),
            np.minimum(self.monthly_data['ltv'] / self.monthly_data['cac'], max_ratio),
            np.nan  # Leave as NaN where CAC isn't available
        )

    def _generate_annual_summary(self):
        """
        Generate annual summary metrics from monthly data
        """
        # Group by year
        annual_groups = self.monthly_data.groupby('year_number')

        # Initialize annual dataframe
        self.annual_data = pd.DataFrame({
            'year': range(1, (len(self.monthly_data) // 12) + 1),
            'year_start_date': [
                self.monthly_data[self.monthly_data['year_number']
                                  == year]['date'].iloc[0]
                for year in range(1, (len(self.monthly_data) // 12) + 1)
            ],
            'year_end_date': [
                self.monthly_data[self.monthly_data['year_number']
                                  == year]['date'].iloc[-1]
                for year in range(1, (len(self.monthly_data) // 12) + 1)
            ]
        })

        # Add year-end metrics
        self.annual_data['year_end_arr'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['total_arr'].iloc[-1]
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        self.annual_data['year_end_customers'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['total_customers'].iloc[-1]
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        self.annual_data['year_end_headcount'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['total_headcount'].iloc[-1]
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        # Annual revenue
        self.annual_data['annual_revenue'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['monthly_revenue'].sum()
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        # Annual expenses
        expense_categories = [
            'total_cogs', 'total_compensation', 'total_marketing_expenses',
            'total_sales_expenses', 'total_r_and_d_expenses', 'total_g_and_a_expenses',
            'one_time_expenses', 'total_operating_expenses'
        ]

        for category in expense_categories:
            self.annual_data[f'annual_{category}'] = [
                self.monthly_data[self.monthly_data['year_number']
                                  == year][category].sum()
                for year in range(1, (len(self.monthly_data) // 12) + 1)
            ]

        # Calculate annual financial metrics
        self.annual_data['annual_gross_profit'] = self.annual_data['annual_revenue'] - \
            self.annual_data['annual_total_cogs']
        
        # Prevent division by zero in annual gross margin calculation
        self.annual_data['annual_gross_margin'] = np.where(
            self.annual_data['annual_revenue'] > 0,
            self.annual_data['annual_gross_profit'] / self.annual_data['annual_revenue'],
            0  # Default to 0 margin when revenue is 0
        )
        
        self.annual_data['annual_ebitda'] = self.annual_data['annual_gross_profit'] - \
            self.annual_data['annual_total_operating_expenses']
        
        # Prevent division by zero in annual EBITDA margin calculation
        self.annual_data['annual_ebitda_margin'] = np.where(
            self.annual_data['annual_revenue'] > 0,
            self.annual_data['annual_ebitda'] / self.annual_data['annual_revenue'],
            0  # Default to 0 margin when revenue is 0
        )

        # Calculate annual new/churned customers
        self.annual_data['annual_new_customers'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['total_new_customers'].sum()
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        self.annual_data['annual_churned_customers'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['total_churned_customers'].sum()
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        # Calculate average CAC and LTV for each year
        self.annual_data['annual_avg_cac'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['cac'].mean()
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        self.annual_data['annual_avg_ltv'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['ltv'].mean()
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        self.annual_data['annual_ltv_cac_ratio'] = self.annual_data['annual_avg_ltv'] / \
            self.annual_data['annual_avg_cac']

        # Year-end capital position
        self.annual_data['year_end_capital'] = [
            self.monthly_data[self.monthly_data['year_number']
                              == year]['capital'].iloc[-1]
            for year in range(1, (len(self.monthly_data) // 12) + 1)
        ]

        # Calculate year-over-year growth rates
        self.annual_data['arr_growth_rate'] = np.nan
        self.annual_data['revenue_growth_rate'] = np.nan

        # Set a minimum threshold for calculating growth rates to avoid division by very small numbers
        min_arr_threshold = 10000  # $10K minimum for ARR
        min_revenue_threshold = 1000  # $1K minimum for revenue
        max_growth_rate = 10.0  # 1000% as a maximum realistic growth rate

        for i in range(1, len(self.annual_data)):
            # ARR growth
            prev_arr = self.annual_data.loc[i-1, 'year_end_arr']
            current_arr = self.annual_data.loc[i, 'year_end_arr']
            
            if prev_arr >= min_arr_threshold:
                # Calculate growth rate and cap at maximum
                growth_rate = (current_arr / prev_arr) - 1
                self.annual_data.loc[i, 'arr_growth_rate'] = min(growth_rate, max_growth_rate)
            elif current_arr > 0 and prev_arr > 0:
                # For very small starting values, use a more conservative approach
                self.annual_data.loc[i, 'arr_growth_rate'] = min((current_arr - prev_arr) / max(prev_arr, min_arr_threshold), max_growth_rate)
            else:
                # Default when there's no meaningful previous value
                self.annual_data.loc[i, 'arr_growth_rate'] = np.nan

            # Revenue growth - similar approach
            prev_revenue = self.annual_data.loc[i-1, 'annual_revenue']
            current_revenue = self.annual_data.loc[i, 'annual_revenue']
            
            if prev_revenue >= min_revenue_threshold:
                # Calculate growth rate and cap at maximum
                growth_rate = (current_revenue / prev_revenue) - 1
                self.annual_data.loc[i, 'revenue_growth_rate'] = min(growth_rate, max_growth_rate)
            elif current_revenue > 0 and prev_revenue > 0:
                # For very small starting values, use a more conservative approach
                self.annual_data.loc[i, 'revenue_growth_rate'] = min((current_revenue - prev_revenue) / max(prev_revenue, min_revenue_threshold), max_growth_rate)
            else:
                # Default when there's no meaningful previous value
                self.annual_data.loc[i, 'revenue_growth_rate'] = np.nan

    def _calculate_cumulative_metrics(self):
        """
        Calculate cumulative metrics for the model
        """
        # Calculate cumulative cash flow
        self.monthly_data['cumulative_cash_flow'] = self.monthly_data['cash_flow'].cumsum(
        )

        # Calculate months to profitability
        self.monthly_data['profitable_month'] = self.monthly_data['ebitda'] > 0

        # Add cumulative metrics to annual data
        self.annual_data['cumulative_revenue'] = np.nan
        self.annual_data['cumulative_costs'] = np.nan
        self.annual_data['cumulative_ebitda'] = np.nan

        cum_revenue = 0
        cum_costs = 0
        cum_ebitda = 0

        for i in range(len(self.annual_data)):
            cum_revenue += self.annual_data.loc[i, 'annual_revenue']
            cum_costs += (self.annual_data.loc[i, 'annual_total_cogs'] +
                          self.annual_data.loc[i, 'annual_total_operating_expenses'])
            cum_ebitda += self.annual_data.loc[i, 'annual_ebitda']

            self.annual_data.loc[i, 'cumulative_revenue'] = cum_revenue
            self.annual_data.loc[i, 'cumulative_costs'] = cum_costs
            self.annual_data.loc[i, 'cumulative_ebitda'] = cum_ebitda

        # Calculate Rule of 40 metric (Growth % + Profit %)
        # Handle NaN values and set reasonable bounds
        max_rule_of_40 = 100  # Cap at 100 for extreme cases
        
        # Convert growth rate to percent
        growth_percent = self.annual_data['revenue_growth_rate'].fillna(0) * 100
        
        # Convert margin to percent
        margin_percent = self.annual_data['annual_ebitda_margin'] * 100
        
        # Calculate Rule of 40 with cap
        self.annual_data['rule_of_40'] = np.minimum(
            growth_percent + margin_percent,
            max_rule_of_40
        )

    def get_monthly_data(self):
        """
        Returns the monthly projection data as a pandas DataFrame

        Returns:
        --------
        pandas.DataFrame : Monthly model results

        Raises:
        -------
        RuntimeError : If model has not been run yet
        """
        if not self._model_run:
            raise RuntimeError(
                "Model has not been run yet. Call run_model() first.")
        return self.monthly_data.copy()

    def get_annual_data(self):
        """
        Returns the annual projection data as a pandas DataFrame

        Returns:
        --------
        pandas.DataFrame : Annual model results

        Raises:
        -------
        RuntimeError : If model has not been run yet
        """
        if not self._model_run:
            raise RuntimeError(
                "Model has not been run yet. Call run_model() first.")
        return self.annual_data.copy()

    def plot_financial_summary(self, figsize=(14, 8)):
        """
        Plot financial summary with revenue and EBITDA as side-by-side columns and customers as a line

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plot

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plot
        """
        if not hasattr(self, '_model_run') or not self._model_run:
            raise RuntimeError(
                "Model has not been run yet. Call run_model() first.")

        # Create figure with proper sizing
        fig, ax1 = plt.subplots(figsize=figsize)

        # Set up data
        years = self.annual_data['year'].values
        years_str = [f'Year {year}' for year in years]
        # Convert to millions
        revenue = self.annual_data['annual_revenue'].values / 1000000
        # Convert to millions
        ebitda = self.annual_data['annual_ebitda'].values / 1000000
        customers = self.annual_data['year_end_customers'].values

        # Set up x positions
        x = np.arange(len(years))
        bar_width = 0.35

        # Plot revenue bars in dark blue (left position)
        revenue_bars = ax1.bar(x - bar_width/2, revenue, width=bar_width,
                               label='Total revenue', color='#1a3e8c')

        # Plot EBITDA bars in light blue (right position)
        ebitda_bars = ax1.bar(x + bar_width/2, ebitda, width=bar_width,
                              label='EBITDA', color='#30a9de')

        # Configure left y-axis (revenue/EBITDA)
        ax1.set_xlabel('')
        ax1.set_ylabel('Revenue/EBITDA ($M)')

        # Set y-axis range to include padding for negative values
        # Add padding below the lowest negative value
        y_min = min(min(ebitda), 0) - 5
        y_max = max(revenue) * 1.2  # Add 20% padding above the highest revenue
        ax1.set_ylim(y_min, y_max)

        # Add horizontal grid lines with dotted style
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        ax1.set_axisbelow(True)  # Place gridlines behind the bars

        # Create secondary y-axis for customers
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of customers')
        
        # Set formatter for primary y-axis to show 2 decimal places
        from matplotlib.ticker import FormatStrFormatter
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Set the y-axis range for revenue/EBITDA
        # Ensure we have a reasonable amount of space below zero
        padding_below = max(5, abs(min(ebitda)) * 0.2)  # At least 5M or 20% of the lowest negative value
        y_min = min(min(ebitda), 0) - padding_below  
        y_max = max(revenue) * 1.2  # Add 20% padding above the highest revenue
        ax1.set_ylim(y_min, y_max)
        
        # For customer axis, directly align the 0 point with the revenue/EBITDA zero
        customer_max = max(customers) * 1.1
        
        # Get the fraction of the revenue/EBITDA y-axis occupied by negative values
        if y_min >= 0:
            # No negative values, just set both axes to start at 0
            ax2.set_ylim(0, customer_max)
        else:
            # Calculate what fraction of the axis is below zero
            fraction_below_zero = abs(y_min) / (y_max - y_min)
            
            # Create the same proportion for customer axis by adjusting the top value
            # This ensures y=0 aligns on both axes
            customer_new_max = customer_max / (1.0 - fraction_below_zero)
            ax2.set_ylim(bottom=0, top=customer_new_max)

        # Plot customers as line with markers
        customer_line = ax2.plot(x, customers, marker='o', color='#ff6347',
                                 linewidth=2, label='customers')

        # Add labels on the bars with 2 decimal places
        for i, bar in enumerate(revenue_bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                         f'{revenue[i]:.2f}', ha='center', va='bottom')

        for i, bar in enumerate(ebitda_bars):
            height = bar.get_height()
            sign = '+' if height > 0 else ''
            pos_y = height + 0.5 if height > 0 else height - 2
            ax1.text(bar.get_x() + bar.get_width()/2., pos_y,
                     f'{sign}{ebitda[i]:.2f}', ha='center', va='bottom')

        # Add customer labels
        for i, y in enumerate(customers):
            ax2.text(x[i], y + (customer_max * 0.02), f'{int(y)}',
                     ha='center', va='bottom')

        # Use actual years as x-axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(years_str)

        # Add a zero line for clarity with EBITDA values
        ax1.axhline(y=0, color='black', linestyle='-',
                    linewidth=0.5, alpha=0.5)

        # Add legends at the bottom
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                   bbox_to_anchor=(0.5, -0.05), ncol=3)

        # Add title
        plt.title('Key Financial Results and Projections',
                  fontsize=16, fontweight='bold')

        # Remove top and right spines for cleaner look
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_unit_economics(self, figsize=(12, 8)):
        """
        Plot unit economics metrics (CAC, LTV, LTV/CAC)

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plot

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plot
        """
        if not self._model_run:
            raise RuntimeError(
                "Model has not been run yet. Call run_model() first.")

        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)

        # Set up data
        years = self.annual_data['year'].values
        years_str = [f'Year {year}' for year in years]
        cac = self.annual_data['annual_avg_cac'].values
        ltv = self.annual_data['annual_avg_ltv'].values
        ltv_cac_ratio = self.annual_data['annual_ltv_cac_ratio'].values

        # Plot CAC and LTV as grouped bars
        bar_width = 0.35
        ax1.bar(years - bar_width/2, cac, bar_width, label='CAC')
        ax1.bar(years + bar_width/2, ltv, bar_width, label='LTV')

        # Configure left y-axis (CAC, LTV)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Amount ($)')

        # Create secondary y-axis for LTV/CAC ratio
        ax2 = ax1.twinx()
        ax2.set_ylabel('LTV/CAC Ratio')

        # Plot LTV/CAC ratio as line
        ax2.plot(years, ltv_cac_ratio, marker='o', color='red',
                 linewidth=2, label='LTV/CAC Ratio')

        # Add LTV/CAC target line at 3x
        ax2.axhline(y=3, color='green', linestyle='--', label='3x Target')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Add title
        plt.title('Unit Economics Metrics', fontsize=14)

        # Use actual years as x-axis labels
        ax1.set_xticks(years)
        ax1.set_xticklabels(years_str)

        plt.tight_layout()
        return fig

    def plot_runway_and_capital(self, figsize=(12, 8)):
        """
        Plot runway and capital position over time

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plot

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plot
        """
        if not self._model_run:
            raise RuntimeError(
                "Model has not been run yet. Call run_model() first.")

        # Create figure
        fig, ax1 = plt.subplots(figsize=figsize)

        # Set up data - use monthly data for more granular view
        months = self.monthly_data['month_number'].values
        # Convert to millions
        capital = self.monthly_data['capital'].values / 1000000
        # Convert to millions
        cash_flow = self.monthly_data['cash_flow'].values / 1000000

        # Plot capital position as line
        ax1.plot(months, capital, linewidth=2, label='Capital Position ($M)')

        # Plot cash flow as bars
        ax1.bar(months, cash_flow, alpha=0.4, label='Monthly Cash Flow ($M)')

        # Add zero line
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Configure y-axis
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Amount ($M)')

        # Add year markers
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax1.axvline(x=month, color='gray', linestyle='--', alpha=0.5)
            ax1.text(month, ax1.get_ylim()[1]*0.95, f'Year {year}',
                     ha='center', va='top', backgroundcolor='white', alpha=0.8)

        ax1.legend()

        # Add title
        plt.title('Capital Position and Monthly Cash Flow', fontsize=14)

        plt.tight_layout()
        return fig

    def plot_break_even_analysis(self, figsize=(12, 8)):
        """
        Plot break-even analysis showing revenue vs expenses

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plot

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plot
        """
        if not self._model_run:
            raise RuntimeError(
                "Model has not been run yet. Call run_model() first.")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Set up data
        months = self.monthly_data['month_number'].values
        # Convert to millions
        revenue = self.monthly_data['monthly_revenue'].values / 1000000
        expenses = (self.monthly_data['total_cogs'] +
                    self.monthly_data['total_operating_expenses']).values / 1000000

        # Plot revenue and expenses
        ax.plot(months, revenue, linewidth=2, label='Monthly Revenue ($M)')
        ax.plot(months, expenses, linewidth=2, label='Monthly Expenses ($M)')

        # Shade the area of loss
        for i in range(len(months)-1):
            if revenue[i] < expenses[i]:
                ax.fill_between([months[i], months[i+1]], [revenue[i], revenue[i+1]],
                                [expenses[i], expenses[i+1]], alpha=0.3, color='red')
            else:
                ax.fill_between([months[i], months[i+1]], [revenue[i], revenue[i+1]],
                                [expenses[i], expenses[i+1]], alpha=0.3, color='green')

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Configure y-axis
        ax.set_xlabel('Month')
        ax.set_ylabel('Amount ($M)')

        # Add year markers
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.5)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        # Find break-even point with sustained profitability (3+ consecutive months)
        consecutive_months_required = 3  # Number of consecutive profitable months required
        sustained_breakeven_idx = None
        
        # Find indices where revenue >= expenses
        profitable_months = revenue >= expenses
        
        # Find the first occurrence of 3+ consecutive profitable months
        for i in range(len(profitable_months) - consecutive_months_required + 1):
            if np.all(profitable_months[i:i+consecutive_months_required]):
                sustained_breakeven_idx = i
                break
                
        # If we found a sustained break-even point
        if sustained_breakeven_idx is not None and sustained_breakeven_idx < len(months):
            break_even_month = months[sustained_breakeven_idx]
            
            # Mark break-even point
            ax.plot(break_even_month,
                    revenue[sustained_breakeven_idx], 'ro', markersize=8)
            ax.annotate(f'Break-even: Month {break_even_month} (sustained)',
                        xy=(break_even_month, revenue[sustained_breakeven_idx]),
                        xytext=(break_even_month+5,
                                revenue[sustained_breakeven_idx]+0.2),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

        ax.legend()

        # Add title
        plt.title('Break-Even Analysis', fontsize=14)

        plt.tight_layout()
        return fig

    def get_key_metrics_table(self):
        """
        Returns key metrics in a formatted table

        Returns:
        --------
        pandas.DataFrame : Key metrics table
        """
        if not self._model_run:
            raise RuntimeError(
                "Model has not been run yet. Call run_model() first.")

        # Create metrics dataframe
        years = self.annual_data['year'].values
        years_str = [f'Year {year}' for year in years]

        metrics = pd.DataFrame(index=years_str)

        # Add metrics directly from the revenue model
        revenue_annual = self.revenue_model.get_annual_data()

        # Add both ARR (point-in-time) and annual revenue (sum of monthly revenue)
        metrics['ARR ($M)'] = self.annual_data['year_end_arr'].values / 1000000
        metrics['Revenue ($M)'] = self.annual_data['annual_revenue'].values / 1000000
        metrics['Customers'] = revenue_annual['total_ending_customers'].values

        # Add metrics from the cost model
        cost_annual = self.cost_model.get_annual_data()
        metrics['Headcount'] = cost_annual['year_end_headcount'].values

        # Add total expenses
        metrics['Total Expenses ($M)'] = cost_annual['total_expenses'].values / 1000000

        # Add financial metrics
        metrics['EBITDA ($M)'] = self.annual_data['annual_ebitda'].values / 1000000
        metrics['EBITDA Margin (%)'] = self.annual_data['annual_ebitda_margin'].values * 100
        metrics['Capital Position ($M)'] = self.annual_data['year_end_capital'].values / 1000000

        # Calculate unit economics metrics
        # CAC: Total sales & marketing expense divided by new customers
        # Add safety check to prevent division by zero
        safe_new_customers = np.where(
            revenue_annual['total_new_customers'].values > 0,
            revenue_annual['total_new_customers'].values,
            1  # Default to 1 to avoid division by zero
        )
        metrics['CAC ($)'] = (cost_annual['total_marketing_expenses'].values +
                              cost_annual['total_sales_expenses'].values) / safe_new_customers

        # LTV: Average revenue per user * gross margin * customer lifetime
        arpu = revenue_annual['total_ending_arr'].values / \
            revenue_annual['total_ending_customers'].values
            
        # Calculate actual gross margin from the model instead of using a hardcoded value
        annual_revenue = self.annual_data['annual_revenue'].values
        annual_cogs = self.annual_data['annual_total_cogs'].values
        
        gross_margin = np.where(
            annual_revenue > 0, 
            (annual_revenue - annual_cogs) / annual_revenue,
            0.9  # Default 90% margin when revenue is 0 (consistent with model defaults)
        )
        
        # Get min churn rate from the revenue model if available
        if hasattr(self.revenue_model, 'config') and 'churn_rates' in self.revenue_model.config:
            # Find the minimum churn rate from all segments (excluding _comment)
            churn_values = [v for k, v in self.revenue_model.config['churn_rates'].items() if k != "_comment"]
            min_annual_churn_rate = min(churn_values)
        else:
            # Default value if not available
            min_annual_churn_rate = 0.08  # Default minimum churn
            
        # Calculate max lifetime based on min churn (inversely related)
        max_lifetime_years = 1 / min_annual_churn_rate
        
        # Get annual churn rates if available, or use default
        if 'total_churn_rate' in revenue_annual.columns:
            annual_churn_rates = np.maximum(revenue_annual['total_churn_rate'].values, min_annual_churn_rate)
        else:
            annual_churn_rates = np.full(len(years), 0.15)  # Default 15% annual churn rate
            
        # Calculate customer lifetime with caps
        customer_lifetimes = np.minimum(1 / annual_churn_rates, max_lifetime_years)
        
        # Calculate LTV based on customer lifetimes (dynamic based on churn rates)
        raw_ltv = arpu * gross_margin * customer_lifetimes
        # Cap LTV at reasonable multiple of ARPU based on the max lifetime
        metrics['LTV ($)'] = raw_ltv

        # LTV/CAC ratio - allow full range based on actual values
        metrics['LTV/CAC Ratio'] = metrics['LTV ($)'].values / metrics['CAC ($)'].values

        # Rule of 40 score (Growth rate + EBITDA margin)
        # Handle NaN values and set reasonable bounds
        max_rule_of_40 = 100  # Cap at 100 for extreme cases
        min_rule_of_40 = -100  # Floor at -100 for extreme negative cases
        
        rule_of_40 = []
        
        # First year: typically only report EBITDA margin since there's no prior year for growth
        # Note: This is a standard SaaS practice for first year metrics
        first_year_r40 = min(max(metrics.loc['Year 1', 'EBITDA Margin (%)'], min_rule_of_40), max_rule_of_40)
        rule_of_40.append(first_year_r40)

        # For subsequent years, add growth rate + margin with bounds
        for i in range(1, len(years)):
            if 'total_arr_growth_rate' in revenue_annual.columns:
                growth_rate = revenue_annual.loc[i, 'total_arr_growth_rate'] * 100
                
                # Cap extreme growth rates
                growth_rate = min(growth_rate, 200)  # Cap at 200% for realistic reporting
                
                r40_score = growth_rate + metrics.loc[f'Year {i+1}', 'EBITDA Margin (%)']
                
                # Apply min/max bounds
                r40_score = min(max(r40_score, min_rule_of_40), max_rule_of_40)
                rule_of_40.append(r40_score)
            else:
                # Fallback if growth rate isn't available
                rule_of_40.append(metrics.loc[f'Year {i+1}', 'EBITDA Margin (%)'])

        metrics['Rule of 40 Score'] = rule_of_40

        # Replace any NaN values with zeros
        metrics = metrics.fillna(0)

        # Format metrics
        formatted_metrics = pd.DataFrame(index=metrics.index)

        formatted_metrics['ARR ($M)'] = [
            f"${x:.1f}M" for x in metrics['ARR ($M)'].values]
        formatted_metrics['Revenue ($M)'] = [
            f"${x:.1f}M" for x in metrics['Revenue ($M)'].values]
        formatted_metrics['Total Expenses ($M)'] = [
            f"${x:.1f}M" for x in metrics['Total Expenses ($M)'].values]
        formatted_metrics['EBITDA ($M)'] = [
            f"${x:.1f}M" if x >= 0 else f"-${abs(x):.1f}M" for x in metrics['EBITDA ($M)'].values]
        formatted_metrics['EBITDA Margin (%)'] = [
            f"{x:.1f}%" for x in metrics['EBITDA Margin (%)'].values]
        formatted_metrics['Customers'] = [
            f"{int(x):,}" for x in metrics['Customers'].values]
        formatted_metrics['Headcount'] = [
            f"{x:.1f}" for x in metrics['Headcount'].values]
        formatted_metrics['CAC ($)'] = [
            f"${int(x):,}" for x in metrics['CAC ($)'].values]
        formatted_metrics['LTV ($)'] = [
            f"${int(x):,}" for x in metrics['LTV ($)'].values]
        formatted_metrics['LTV/CAC Ratio'] = [
            f"{x:.1f}x" for x in metrics['LTV/CAC Ratio'].values]
        formatted_metrics['Rule of 40 Score'] = [
            f"{x:.1f}" for x in metrics['Rule of 40 Score'].values]
        formatted_metrics['Capital Position ($M)'] = [
            f"${x:.1f}M" if x >= 0 else f"-${abs(x):.1f}M" for x in metrics['Capital Position ($M)'].values]

        return formatted_metrics
