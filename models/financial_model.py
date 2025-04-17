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
        self.monthly_data['gross_margin'] = self.monthly_data['gross_profit'] / \
            self.monthly_data['monthly_revenue']
        self.monthly_data['ebitda'] = self.monthly_data['gross_profit'] - \
            self.monthly_data['total_operating_expenses']
        self.monthly_data['ebitda_margin'] = self.monthly_data['ebitda'] / \
            self.monthly_data['monthly_revenue']

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
        for i in range(len(self.monthly_data)):
            if self.monthly_data.loc[i, 'cash_flow'] < 0:
                remaining_capital = self.monthly_data.loc[i, 'capital']
                current_burn = -self.monthly_data.loc[i, 'cash_flow']
                if current_burn > 0:
                    self.monthly_data.loc[i,
                                          'runway_months'] = remaining_capital / current_burn
                else:
                    self.monthly_data.loc[i, 'runway_months'] = float('inf')
            else:
                self.monthly_data.loc[i, 'runway_months'] = float('inf')

        # Calculate CAC
        self.monthly_data['sales_marketing_expense'] = (
            self.monthly_data['total_marketing_expenses'] +
            self.monthly_data['total_sales_expenses']
        )
        self.monthly_data['cac'] = np.nan

        # Calculate CAC on a rolling 3-month basis
        for i in range(3, len(self.monthly_data)):
            new_customers = self.monthly_data.loc[i -
                                                  2:i, 'total_new_customers'].sum()
            sm_expenses = self.monthly_data.loc[i -
                                                2:i, 'sales_marketing_expense'].sum()
            if new_customers > 0:
                self.monthly_data.loc[i, 'cac'] = sm_expenses / new_customers

        # Calculate LTV (simpler version)
        self.monthly_data['arpu'] = self.monthly_data['total_arr'] / \
            self.monthly_data['total_customers']
        # Average monthly gross margin per customer
        self.monthly_data['monthly_margin_per_customer'] = (
            self.monthly_data['gross_margin'] * self.monthly_data['arpu'] / 12
        )

        # Use churn rates from revenue model to calculate customer lifetime
        # Assumes churn rate at segment level is tracked
        avg_churn_rate = 0.15  # Default annual churn rate if not available
        if 'total_churned_customers' in self.monthly_data.columns and 'total_customers' in self.monthly_data.columns:
            for i in range(12, len(self.monthly_data), 12):  # Annual calculation
                annual_churned = self.monthly_data.loc[i -
                                                       11:i, 'total_churned_customers'].sum()
                avg_customers = self.monthly_data.loc[i -
                                                      11:i, 'total_customers'].mean()
                if avg_customers > 0:
                    annual_churn_rate = annual_churned / avg_customers
                    if annual_churn_rate > 0:
                        self.monthly_data.loc[i-11:i, 'customer_lifetime_months'] = 1 / (
                            annual_churn_rate / 12)

        # Default lifetime if calculation fails
        if 'customer_lifetime_months' not in self.monthly_data.columns:
            self.monthly_data['customer_lifetime_months'] = 1 / \
                (avg_churn_rate / 12)

        # Calculate LTV
        self.monthly_data['ltv'] = self.monthly_data['monthly_margin_per_customer'] * \
            self.monthly_data['customer_lifetime_months']

        # Calculate LTV/CAC ratio
        self.monthly_data['ltv_cac_ratio'] = self.monthly_data['ltv'] / \
            self.monthly_data['cac']

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
        self.annual_data['annual_gross_margin'] = self.annual_data['annual_gross_profit'] / \
            self.annual_data['annual_revenue']
        self.annual_data['annual_ebitda'] = self.annual_data['annual_gross_profit'] - \
            self.annual_data['annual_total_operating_expenses']
        self.annual_data['annual_ebitda_margin'] = self.annual_data['annual_ebitda'] / \
            self.annual_data['annual_revenue']

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

        for i in range(1, len(self.annual_data)):
            # ARR growth
            prev_arr = self.annual_data.loc[i-1, 'year_end_arr']
            current_arr = self.annual_data.loc[i, 'year_end_arr']
            if prev_arr > 0:
                self.annual_data.loc[i, 'arr_growth_rate'] = (
                    current_arr / prev_arr) - 1

            # Revenue growth
            prev_revenue = self.annual_data.loc[i-1, 'annual_revenue']
            current_revenue = self.annual_data.loc[i, 'annual_revenue']
            if prev_revenue > 0:
                self.annual_data.loc[i, 'revenue_growth_rate'] = (
                    current_revenue / prev_revenue) - 1

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
        self.annual_data['rule_of_40'] = (
            (self.annual_data['revenue_growth_rate'] * 100) +
            (self.annual_data['annual_ebitda_margin'] * 100)
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
        Plot financial summary similar to the example provided

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
        years_str = [str(self.annual_data['year_start_date'].iloc[i].year)
                     for i in range(len(years))]
        # Convert to millions
        revenue = self.annual_data['annual_revenue'].values / 1000000
        # Convert to millions
        ebitda = self.annual_data['annual_ebitda'].values / 1000000
        customers = self.annual_data['year_end_customers'].values

        # Plot revenue as bars
        bar_width = 0.4
        revenue_bars = ax1.bar(years, revenue, bar_width,
                               label='Total revenue', color='#3b5998')

        # Plot EBITDA as bars
        ebitda_bars = ax1.bar(years, ebitda, bar_width,
                              label='EBITDA', color='#4ECDC4')

        # Configure left y-axis (revenue/EBITDA)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Revenue/EBITDA ($M)')

        # Create secondary y-axis for customers
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of customers')

        # Plot customers as line
        customer_line = ax2.plot(
            years, customers, marker='o', color='#FF6B6B', linewidth=2, label='Customers')

        # Set labels on the bars
        for i, bar in enumerate(revenue_bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                         f'${revenue[i]:.0f}M', ha='center', va='bottom')

        for i, bar in enumerate(ebitda_bars):
            height = bar.get_height()
            sign = '+' if height > 0 else ''
            ax1.text(bar.get_x() + bar.get_width()/2.,
                     height + 0.3 if height > 0 else height - 1.5,
                     f'{sign}${ebitda[i]:.0f}M', ha='center', va='bottom')

        # Add customer count labels
        for i, y in enumerate(customers):
            ax2.text(years[i], y + 5, f'{int(y)}', ha='center', va='bottom')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Add title
        plt.title('Key Financial Results and Projections\nIllustrative example',
                  fontsize=18, fontweight='bold')

        # Use actual years as x-axis labels
        ax1.set_xticks(years)
        ax1.set_xticklabels(years_str)

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
        years_str = [str(self.annual_data['year_start_date'].iloc[i].year)
                     for i in range(len(years))]
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

        # Find break-even point
        break_even_idx = np.argmax(revenue >= expenses)
        if break_even_idx > 0 and break_even_idx < len(months) and revenue[break_even_idx] >= expenses[break_even_idx]:
            break_even_month = months[break_even_idx]

            # Mark break-even point
            ax.plot(break_even_month,
                    revenue[break_even_idx], 'ro', markersize=8)
            ax.annotate(f'Break-even: Month {break_even_month}',
                        xy=(break_even_month, revenue[break_even_idx]),
                        xytext=(break_even_month+5,
                                revenue[break_even_idx]+0.2),
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
        # Convert ARR to monthly revenue
        metrics['Revenue ($M)'] = revenue_annual['total_ending_arr'].values / \
            12 / 1000000
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
        metrics['CAC ($)'] = (cost_annual['total_marketing_expenses'].values +
                              cost_annual['total_sales_expenses'].values) / revenue_annual['total_new_customers'].values

        # LTV: Average revenue per user * gross margin * customer lifetime
        arpu = revenue_annual['total_ending_arr'].values / \
            revenue_annual['total_ending_customers'].values
        gross_margin = 0.65  # Assumed 65% gross margin from COGS
        customer_lifetime = 1 / 0.15  # Assuming 15% annual churn rate
        metrics['LTV ($)'] = arpu * gross_margin * customer_lifetime

        # LTV/CAC ratio
        metrics['LTV/CAC Ratio'] = metrics['LTV ($)'].values / \
            metrics['CAC ($)'].values

        # Rule of 40 score (Growth rate + EBITDA margin)
        rule_of_40 = []
        # First year just use EBITDA margin
        rule_of_40.append(metrics.loc['Year 1', 'EBITDA Margin (%)'])

        # For subsequent years, add growth rate
        for i in range(1, len(years)):
            growth_rate = revenue_annual.loc[i, 'total_arr_growth_rate'] * 100
            rule_of_40.append(
                growth_rate + metrics.loc[f'Year {i+1}', 'EBITDA Margin (%)'])

        metrics['Rule of 40 Score'] = rule_of_40

        # Replace any NaN values with zeros
        metrics = metrics.fillna(0)

        # Format metrics
        formatted_metrics = pd.DataFrame(index=metrics.index)

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
            f"{int(x)}" for x in metrics['Headcount'].values]
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
