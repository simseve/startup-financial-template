import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class AISaaSCostModel:
    """
    Cost modeling class for AI SaaS companies with detailed tracking of
    COGS, headcount, salaries, R&D, marketing, and other operating expenses.
    
    This class can be used standalone or integrated with a revenue model
    like SaaSGrowthModel to provide comprehensive financial projections.
    """
    
    def __init__(self, config=None):
        """
        Initialize the AI SaaS cost model with configuration parameters
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the cost model
        """
        # Set default configuration if none provided
        if config is None:
            self.config = self._get_default_config()
        else:
            self.config = config
            
        # Initialize results dataframes
        self.monthly_data = None
        self.annual_data = None
        self._model_run = False
        
    def _get_default_config(self):
        """
        Returns a default configuration with typical AI SaaS cost structures
        """
        return {
            # Basic parameters
            'start_date': '2025-01-01',
            'projection_months': 72,  # 6 years
            
            # COGS Assumptions (% of ARR)
            'cogs': {
                'cloud_hosting': 0.12,  # 12% of ARR for cloud infrastructure
                'customer_support': 0.08,  # 8% of ARR for support
                'third_party_apis': 0.05,  # 5% of ARR for third-party AI/ML APIs
                'professional_services': 0.03,  # 3% of ARR for PS delivery
            },
            
            # Headcount Assumptions (starting headcount, with growth factors)
            'headcount': {
                # Engineering including ML/AI specialists
                'engineering': {
                    'starting_count': 10,
                    'growth_type': 'step',  # 'linear', 'step', or 's_curve'
                    'growth_factors': {
                        1: 1.0,  # Year 1: No change
                        2: 1.5,  # Year 2: 50% growth
                        3: 1.3,  # Year 3: 30% growth
                        4: 1.2,  # Year 4: 20% growth
                        5: 1.2,  # Year 5: 20% growth
                        6: 1.1,  # Year 6: 10% growth
                    },
                    'avg_salary': 150000,  # Average annual salary
                },
                # Product Management
                'product': {
                    'starting_count': 3,
                    'growth_type': 'step',
                    'growth_factors': {
                        1: 1.0,
                        2: 1.3,
                        3: 1.3,
                        4: 1.2,
                        5: 1.2,
                        6: 1.1,
                    },
                    'avg_salary': 140000,
                },
                # Sales team
                'sales': {
                    'starting_count': 5,
                    'growth_type': 'step',
                    'growth_factors': {
                        1: 1.4,  # Sales grows faster early
                        2: 1.5,
                        3: 1.3,
                        4: 1.2,
                        5: 1.2,
                        6: 1.1,
                    },
                    'avg_salary': 120000,  # Base salary (not including commissions)
                },
                # Marketing team
                'marketing': {
                    'starting_count': 3,
                    'growth_type': 'step',
                    'growth_factors': {
                        1: 1.3,
                        2: 1.4,
                        3: 1.2,
                        4: 1.2,
                        5: 1.1,
                        6: 1.1,
                    },
                    'avg_salary': 110000,
                },
                # Customer success/support
                'customer_success': {
                    'starting_count': 4,
                    'growth_type': 'step',
                    'growth_factors': {
                        1: 1.2,
                        2: 1.3,
                        3: 1.3,
                        4: 1.2,
                        5: 1.2,
                        6: 1.1,
                    },
                    'avg_salary': 85000,
                },
                # G&A (General and Administrative)
                'g_and_a': {
                    'starting_count': 3,
                    'growth_type': 'step',
                    'growth_factors': {
                        1: 1.0,
                        2: 1.3,
                        3: 1.2,
                        4: 1.2,
                        5: 1.1,
                        6: 1.1,
                    },
                    'avg_salary': 100000,
                },
            },
            
            # Salary & Benefits Assumptions
            'salary': {
                'annual_increase': 0.03,  # 3% annual salary increase
                'benefits_multiplier': 1.25,  # Benefits are 25% of base salary
                'payroll_tax_rate': 0.08,  # 8% payroll taxes
                'bonus_rate': 0.10,  # 10% annual bonus
                'equity_compensation': 0.15,  # 15% of salary as equity comp
            },
            
            # Marketing & Other Expenses
            'marketing_expenses': {
                # Non-headcount marketing expenses (% of ARR)
                'paid_advertising': 0.15,  # 15% of ARR
                'content_creation': 0.05,  # 5% of ARR
                'events_and_pr': 0.03,  # 3% of ARR
                'partner_marketing': 0.02,  # 2% of ARR
            },
            
            # Growth multiplier for marketing as company scales (efficiency/economies of scale)
            'marketing_efficiency': {
                1: 1.0,  # Year 1: Base level
                2: 0.95,  # Year 2: 5% more efficient
                3: 0.90,  # Year 3: 10% more efficient
                4: 0.85,  # Year 4: 15% more efficient
                5: 0.80,  # Year 5: 20% more efficient
                6: 0.75,  # Year 6: 25% more efficient
            },
            
            # Sales expenses
            'sales_expenses': {
                'commission_rate': 0.10,  # 10% commission on new ARR
                'tools_and_enablement': 0.02,  # 2% of ARR
            },
            
            # R&D expenses (beyond headcount)
            'r_and_d_expenses': {
                'cloud_compute_for_training': 0.08,  # 8% of ARR
                'research_tools_and_data': 0.05,  # 5% of ARR
                'third_party_research': 0.02,  # 2% of ARR
            },
            
            # General & Admin expenses
            'g_and_a_expenses': {
                'office_and_facilities': 10000,  # Per month base cost
                'per_employee_office_cost': 500,  # Per employee per month
                'software_and_tools': 300,  # Per employee per month
                'legal_and_accounting': 20000,  # Per month base cost
                'insurance': 5000,  # Per month base cost
            },
            
            # One-time and periodic expenses
            'one_time_expenses': {
                # Format: [month_idx, category, amount, description]
                'items': [
                    [3, 'office', 100000, 'Office setup and expansion'],
                    [15, 'software', 50000, 'Enterprise software licenses'],
                    [27, 'legal', 75000, 'IP protection and legal work'],
                    [36, 'office', 150000, 'New office location setup'],
                    [48, 'infrastructure', 200000, 'Major infrastructure upgrade'],
                ]
            }
        }
        
    def set_config(self, config):
        """
        Update the model configuration
        
        Parameters:
        -----------
        config : dict
            New configuration parameters for the model
        """
        self.config = config
        self._model_run = False  # Reset run flag since config changed
        
    def get_config(self):
        """
        Returns the current model configuration
        
        Returns:
        --------
        dict : Current model configuration
        """
        return self.config.copy()
    
    def run_model(self, revenue_model=None):
        """
        Run the cost model for the configured time period
        
        Parameters:
        -----------
        revenue_model : object, optional
            Revenue model instance with monthly_data containing ARR data.
            If provided, costs will be calculated based on the revenue model's ARR.
            If not provided, all costs that depend on ARR will be zero.
        
        Returns:
        --------
        tuple : (monthly_data DataFrame, annual_data DataFrame)
        """
        self._initialize_dataframes()
        
        # Get ARR data if revenue model is provided
        if revenue_model is not None and hasattr(revenue_model, 'monthly_data'):
            self.monthly_data['total_arr'] = revenue_model.monthly_data['total_arr']
        else:
            # If no revenue model, set ARR to 0 (pure cost model)
            self.monthly_data['total_arr'] = 0
        
        # Calculate headcount growth over time
        self._calculate_headcount()
        
        # Calculate salary and benefit expenses
        self._calculate_salary_expenses()
        
        # Calculate COGS
        self._calculate_cogs()
        
        # Calculate marketing expenses
        self._calculate_marketing_expenses()
        
        # Calculate sales expenses
        self._calculate_sales_expenses()
        
        # Calculate R&D expenses
        self._calculate_r_and_d_expenses()
        
        # Calculate G&A expenses
        self._calculate_g_and_a_expenses()
        
        # Add one-time expenses
        self._add_one_time_expenses()
        
        # Calculate totals and ratios
        self._calculate_expense_totals()
        
        # Generate annual summary
        self._generate_annual_summary()
        
        self._model_run = True
        
        return self.monthly_data, self.annual_data
    
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
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
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
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
        return self.annual_data.copy()
    
    def _initialize_dataframes(self):
        """
        Initialize dataframes to store model results
        """
        # Date range for projections
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        date_range = pd.date_range(
            start=start_date, 
            periods=self.config['projection_months'], 
            freq='MS'  # Month Start
        )
        
        # Initialize monthly dataframe
        self.monthly_data = pd.DataFrame({
            'date': date_range,
            'year': [d.year for d in date_range],
            'month': [d.month for d in date_range],
            'month_number': range(1, len(date_range) + 1),
            'year_number': [(i // 12) + 1 for i in range(len(date_range))],
        })
        
        # Add total ARR column (will be populated by revenue model if provided)
        self.monthly_data['total_arr'] = 0.0
        
        # Add department headcount columns
        for department in self.config['headcount']:
            self.monthly_data[f'{department}_headcount'] = 0
        
        # Add total headcount column
        self.monthly_data['total_headcount'] = 0
        
        # Add salary & benefits columns
        for department in self.config['headcount']:
            self.monthly_data[f'{department}_base_salary'] = 0.0
            self.monthly_data[f'{department}_benefits'] = 0.0
            self.monthly_data[f'{department}_payroll_tax'] = 0.0
            self.monthly_data[f'{department}_bonus'] = 0.0
            self.monthly_data[f'{department}_equity'] = 0.0
            self.monthly_data[f'{department}_total_comp'] = 0.0
        
        # Add total compensation columns
        self.monthly_data['total_base_salary'] = 0.0
        self.monthly_data['total_benefits'] = 0.0
        self.monthly_data['total_payroll_tax'] = 0.0
        self.monthly_data['total_bonus'] = 0.0
        self.monthly_data['total_equity'] = 0.0
        self.monthly_data['total_compensation'] = 0.0
        
        # Add COGS columns
        for cogs_category in self.config['cogs']:
            self.monthly_data[f'cogs_{cogs_category}'] = 0.0
        self.monthly_data['total_cogs'] = 0.0
        
        # Add Marketing expense columns
        for marketing_category in self.config['marketing_expenses']:
            self.monthly_data[f'marketing_{marketing_category}'] = 0.0
        self.monthly_data['total_marketing_expenses'] = 0.0
        
        # Add Sales expense columns
        for sales_category in self.config['sales_expenses']:
            self.monthly_data[f'sales_{sales_category}'] = 0.0
        self.monthly_data['total_sales_expenses'] = 0.0
        
        # Add R&D expense columns
        for r_and_d_category in self.config['r_and_d_expenses']:
            self.monthly_data[f'r_and_d_{r_and_d_category}'] = 0.0
        self.monthly_data['total_r_and_d_expenses'] = 0.0
        
        # Add G&A expense columns
        for g_and_a_category in self.config['g_and_a_expenses']:
            self.monthly_data[f'g_and_a_{g_and_a_category}'] = 0.0
        self.monthly_data['total_g_and_a_expenses'] = 0.0
        
        # Add one-time expense column
        self.monthly_data['one_time_expenses'] = 0.0
        
        # Add total expense columns
        self.monthly_data['total_expenses'] = 0.0
        
        # Add high-level category totals
        self.monthly_data['total_operating_expenses'] = 0.0  # OpEx
        
        # Add expense ratios
        self.monthly_data['cogs_as_percent_of_arr'] = 0.0
        self.monthly_data['opex_as_percent_of_arr'] = 0.0
    
    def _calculate_headcount(self):
        """
        Calculate headcount growth for each department over time
        """
        for dept, config in self.config['headcount'].items():
            starting_count = config['starting_count']
            growth_type = config.get('growth_type', 'step')
            
            # Set initial headcount
            self.monthly_data.loc[0, f'{dept}_headcount'] = starting_count
            
            if growth_type == 'step':
                # Step function growth (changes at year boundaries)
                for month_idx in range(1, self.config['projection_months']):
                    year_number = self.monthly_data.loc[month_idx, 'year_number']
                    prev_year_number = self.monthly_data.loc[month_idx - 1, 'year_number']
                    
                    if year_number != prev_year_number:
                        # Year boundary - apply growth factor
                        growth_factor = config['growth_factors'].get(year_number, 1.0)
                        prev_headcount = self.monthly_data.loc[month_idx - 1, f'{dept}_headcount']
                        new_headcount = round(prev_headcount * growth_factor)
                        self.monthly_data.loc[month_idx, f'{dept}_headcount'] = new_headcount
                    else:
                        # Same year - maintain headcount
                        self.monthly_data.loc[month_idx, f'{dept}_headcount'] = self.monthly_data.loc[month_idx - 1, f'{dept}_headcount']
            
            elif growth_type == 'linear':
                # Linear growth throughout the year
                for year_number in range(1, 7):
                    growth_factor = config['growth_factors'].get(year_number, 1.0)
                    
                    # Calculate monthly growth rate
                    if growth_factor == 1.0:
                        monthly_growth = 0
                    else:
                        year_start_idx = (year_number - 1) * 12
                        year_end_idx = year_number * 12 - 1
                        
                        if year_start_idx >= len(self.monthly_data):
                            continue
                            
                        start_headcount = self.monthly_data.loc[year_start_idx, f'{dept}_headcount']
                        target_headcount = round(start_headcount * growth_factor)
                        monthly_growth = (target_headcount - start_headcount) / 12
                        
                        # Apply linear growth through the year
                        for month_idx in range(year_start_idx + 1, min(year_end_idx + 1, len(self.monthly_data))):
                            month_in_year = month_idx - year_start_idx
                            incremental_growth = round(monthly_growth * month_in_year)
                            self.monthly_data.loc[month_idx, f'{dept}_headcount'] = start_headcount + incremental_growth
            
            elif growth_type == 's_curve':
                # S-curve growth (faster in the middle of the year)
                for year_number in range(1, 7):
                    growth_factor = config['growth_factors'].get(year_number, 1.0)
                    
                    if growth_factor == 1.0:
                        continue
                        
                    year_start_idx = (year_number - 1) * 12
                    year_end_idx = year_number * 12 - 1
                    
                    if year_start_idx >= len(self.monthly_data):
                        continue
                        
                    start_headcount = self.monthly_data.loc[year_start_idx, f'{dept}_headcount']
                    target_headcount = round(start_headcount * growth_factor)
                    total_growth = target_headcount - start_headcount
                    
                    # Apply S-curve growth
                    for month_idx in range(year_start_idx + 1, min(year_end_idx + 1, len(self.monthly_data))):
                        month_in_year = month_idx - year_start_idx
                        # S-curve formula: position = 1 / (1 + e^(-steepness * (x - midpoint)))
                        steepness = 0.8
                        midpoint = 6
                        s_curve_position = 1 / (1 + np.exp(-steepness * (month_in_year - midpoint)))
                        incremental_growth = round(total_growth * s_curve_position)
                        self.monthly_data.loc[month_idx, f'{dept}_headcount'] = start_headcount + incremental_growth
        
        # Calculate total headcount across all departments
        departments = self.config['headcount'].keys()
        self.monthly_data['total_headcount'] = self.monthly_data[[f'{dept}_headcount' for dept in departments]].sum(axis=1)
    
    def _calculate_salary_expenses(self):
        """
        Calculate salary and benefits expenses for each department
        """
        salary_config = self.config['salary']
        annual_increase = salary_config['annual_increase']
        benefits_multiplier = salary_config['benefits_multiplier']
        payroll_tax_rate = salary_config['payroll_tax_rate']
        bonus_rate = salary_config['bonus_rate']
        equity_rate = salary_config['equity_compensation']
        
        for dept, config in self.config['headcount'].items():
            base_salary = config['avg_salary']
            
            for month_idx in range(self.config['projection_months']):
                year_number = self.monthly_data.loc[month_idx, 'year_number']
                
                # Apply annual salary increases
                adjusted_salary = base_salary * (1 + annual_increase) ** (year_number - 1)
                
                # Calculate monthly salary expense based on headcount
                headcount = self.monthly_data.loc[month_idx, f'{dept}_headcount']
                monthly_base_salary = (headcount * adjusted_salary) / 12
                
                # Calculate benefits and additional compensation components
                monthly_benefits = monthly_base_salary * (benefits_multiplier - 1)  # Subtract 1 because benefits are additional to base
                monthly_payroll_tax = monthly_base_salary * payroll_tax_rate
                monthly_bonus = monthly_base_salary * bonus_rate
                monthly_equity = monthly_base_salary * equity_rate
                
                # Total monthly compensation
                monthly_total_comp = monthly_base_salary + monthly_benefits + monthly_payroll_tax + monthly_bonus + monthly_equity
                
                # Store values in dataframe
                self.monthly_data.loc[month_idx, f'{dept}_base_salary'] = monthly_base_salary
                self.monthly_data.loc[month_idx, f'{dept}_benefits'] = monthly_benefits
                self.monthly_data.loc[month_idx, f'{dept}_payroll_tax'] = monthly_payroll_tax
                self.monthly_data.loc[month_idx, f'{dept}_bonus'] = monthly_bonus
                self.monthly_data.loc[month_idx, f'{dept}_equity'] = monthly_equity
                self.monthly_data.loc[month_idx, f'{dept}_total_comp'] = monthly_total_comp
        
        # Calculate total compensation across all departments
        departments = self.config['headcount'].keys()
        self.monthly_data['total_base_salary'] = self.monthly_data[[f'{dept}_base_salary' for dept in departments]].sum(axis=1)
        self.monthly_data['total_benefits'] = self.monthly_data[[f'{dept}_benefits' for dept in departments]].sum(axis=1)
        self.monthly_data['total_payroll_tax'] = self.monthly_data[[f'{dept}_payroll_tax' for dept in departments]].sum(axis=1)
        self.monthly_data['total_bonus'] = self.monthly_data[[f'{dept}_bonus' for dept in departments]].sum(axis=1)
        self.monthly_data['total_equity'] = self.monthly_data[[f'{dept}_equity' for dept in departments]].sum(axis=1)
        self.monthly_data['total_compensation'] = self.monthly_data[[f'{dept}_total_comp' for dept in departments]].sum(axis=1)
    
    def _calculate_cogs(self):
        """
        Calculate Cost of Goods Sold (COGS) based on ARR
        """
        for month_idx in range(self.config['projection_months']):
            arr = self.monthly_data.loc[month_idx, 'total_arr']
            
            # Calculate each COGS component
            for cogs_category, rate in self.config['cogs'].items():
                monthly_cogs = (arr * rate) / 12  # Convert to monthly
                self.monthly_data.loc[month_idx, f'cogs_{cogs_category}'] = monthly_cogs
            
            # Calculate total COGS
            cogs_categories = self.config['cogs'].keys()
            self.monthly_data.loc[month_idx, 'total_cogs'] = self.monthly_data.loc[month_idx, [f'cogs_{cat}' for cat in cogs_categories]].sum()
    
    def _calculate_marketing_expenses(self):
        """
        Calculate marketing expenses based on ARR and applying efficiency factors
        """
        for month_idx in range(self.config['projection_months']):
            arr = self.monthly_data.loc[month_idx, 'total_arr']
            year_number = self.monthly_data.loc[month_idx, 'year_number']
            
            # Apply efficiency factor for the year
            efficiency_factor = self.config['marketing_efficiency'].get(year_number, 1.0)
            
            # Calculate each marketing expense component
            for marketing_category, rate in self.config['marketing_expenses'].items():
                monthly_expense = (arr * rate * efficiency_factor) / 12  # Convert to monthly
                self.monthly_data.loc[month_idx, f'marketing_{marketing_category}'] = monthly_expense
            
            # Calculate total marketing expenses
            marketing_categories = self.config['marketing_expenses'].keys()
            self.monthly_data.loc[month_idx, 'total_marketing_expenses'] = self.monthly_data.loc[month_idx, [f'marketing_{cat}' for cat in marketing_categories]].sum()
    
    def _calculate_sales_expenses(self):
        """
        Calculate sales expenses including commissions and enablement costs
        """
        for month_idx in range(self.config['projection_months']):
            arr = self.monthly_data.loc[month_idx, 'total_arr']
            
            # Calculate commission on new ARR if available
            commission_rate = self.config['sales_expenses']['commission_rate']
            if month_idx > 0 and 'total_new_arr' in self.monthly_data.columns:
                new_arr = self.monthly_data.loc[month_idx, 'total_new_arr']
                commission = new_arr * commission_rate
            else:
                # Estimate new ARR as a portion of total ARR for commission calculation
                estimated_new_arr = arr * 0.05  # Assume 5% of ARR is new
                commission = estimated_new_arr * commission_rate
            
            self.monthly_data.loc[month_idx, 'sales_commission_rate'] = commission
            
            # Tools and enablement as percentage of ARR
            tools_rate = self.config['sales_expenses']['tools_and_enablement']
            monthly_tools = (arr * tools_rate) / 12  # Convert to monthly
            self.monthly_data.loc[month_idx, 'sales_tools_and_enablement'] = monthly_tools
            
            # Calculate total sales expenses
            sales_categories = self.config['sales_expenses'].keys()
            self.monthly_data.loc[month_idx, 'total_sales_expenses'] = self.monthly_data.loc[month_idx, [f'sales_{cat}' for cat in sales_categories]].sum()
    
    def _calculate_r_and_d_expenses(self):
        """
        Calculate R&D expenses based on ARR and headcount
        """
        for month_idx in range(self.config['projection_months']):
            arr = self.monthly_data.loc[month_idx, 'total_arr']
            
            # Calculate each R&D expense component
            for r_and_d_category, rate in self.config['r_and_d_expenses'].items():
                monthly_expense = (arr * rate) / 12  # Convert to monthly
                self.monthly_data.loc[month_idx, f'r_and_d_{r_and_d_category}'] = monthly_expense
            
            # Calculate total R&D expenses
            r_and_d_categories = self.config['r_and_d_expenses'].keys()
            self.monthly_data.loc[month_idx, 'total_r_and_d_expenses'] = self.monthly_data.loc[month_idx, [f'r_and_d_{cat}' for cat in r_and_d_categories]].sum()
    
    def _calculate_g_and_a_expenses(self):
        """
        Calculate General & Administrative expenses based on headcount and base costs
        """
        for month_idx in range(self.config['projection_months']):
            headcount = self.monthly_data.loc[month_idx, 'total_headcount']
            
            # Calculate each G&A expense component
            g_and_a_config = self.config['g_and_a_expenses']
            
            # Office and facilities (base cost + per employee cost)
            office_base = g_and_a_config['office_and_facilities']
            per_employee_office = g_and_a_config['per_employee_office_cost'] * headcount
            self.monthly_data.loc[month_idx, 'g_and_a_office_and_facilities'] = office_base + per_employee_office
            
            # Software and tools (per employee)
            software_per_employee = g_and_a_config['software_and_tools'] * headcount
            self.monthly_data.loc[month_idx, 'g_and_a_software_and_tools'] = software_per_employee
            
            # Legal and accounting (base cost)
            self.monthly_data.loc[month_idx, 'g_and_a_legal_and_accounting'] = g_and_a_config['legal_and_accounting']
            
            # Insurance (base cost)
            self.monthly_data.loc[month_idx, 'g_and_a_insurance'] = g_and_a_config['insurance']
            
            # Calculate total G&A expenses
            g_and_a_categories = ['office_and_facilities', 'software_and_tools', 'legal_and_accounting', 'insurance']
            self.monthly_data.loc[month_idx, 'total_g_and_a_expenses'] = self.monthly_data.loc[month_idx, [f'g_and_a_{cat}' for cat in g_and_a_categories]].sum()
    
    def _add_one_time_expenses(self):
        """
        Add one-time expenses to specific months
        """
        for item in self.config['one_time_expenses']['items']:
            month_idx, category, amount, description = item
            
            if month_idx < len(self.monthly_data):
                self.monthly_data.loc[month_idx, 'one_time_expenses'] = self.monthly_data.loc[month_idx, 'one_time_expenses'] + amount
    
    def _calculate_expense_totals(self):
        """
        Calculate overall expense totals and ratios
        """
        for month_idx in range(self.config['projection_months']):
            # Operating Expenses (excluding COGS)
            self.monthly_data.loc[month_idx, 'total_operating_expenses'] = (
                self.monthly_data.loc[month_idx, 'total_compensation'] +
                self.monthly_data.loc[month_idx, 'total_marketing_expenses'] +
                self.monthly_data.loc[month_idx, 'total_sales_expenses'] +
                self.monthly_data.loc[month_idx, 'total_r_and_d_expenses'] +
                self.monthly_data.loc[month_idx, 'total_g_and_a_expenses'] +
                self.monthly_data.loc[month_idx, 'one_time_expenses']
            )
            
            # Total Expenses (COGS + OpEx)
            self.monthly_data.loc[month_idx, 'total_expenses'] = (
                self.monthly_data.loc[month_idx, 'total_cogs'] +
                self.monthly_data.loc[month_idx, 'total_operating_expenses']
            )
            
            # Calculate ratios
            arr = self.monthly_data.loc[month_idx, 'total_arr']
            if arr > 0:
                monthly_arr = arr / 12  # Convert to monthly
                self.monthly_data.loc[month_idx, 'cogs_as_percent_of_arr'] = (
                    self.monthly_data.loc[month_idx, 'total_cogs'] / monthly_arr * 100
                )
                self.monthly_data.loc[month_idx, 'opex_as_percent_of_arr'] = (
                    self.monthly_data.loc[month_idx, 'total_operating_expenses'] / monthly_arr * 100
                )
            else:
                self.monthly_data.loc[month_idx, 'cogs_as_percent_of_arr'] = float('nan')
                self.monthly_data.loc[month_idx, 'opex_as_percent_of_arr'] = float('nan')
    
    def _generate_annual_summary(self):
        """
        Generate annual summary metrics
        """
        # Group by year and aggregate
        annual_groups = self.monthly_data.groupby('year_number')
        
        # Initialize annual dataframe
        self.annual_data = pd.DataFrame({
            'year': range(1, (self.config['projection_months'] // 12) + 1),
            'year_start_date': [
                self.monthly_data[self.monthly_data['year_number'] == year]['date'].iloc[0]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ],
            'year_end_date': [
                self.monthly_data[self.monthly_data['year_number'] == year]['date'].iloc[-1]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]
        })
        
        # Average ARR for each year
        self.annual_data['avg_arr'] = [
            self.monthly_data[self.monthly_data['year_number'] == year]['total_arr'].mean()
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        # Year-end headcount
        self.annual_data['year_end_headcount'] = [
            self.monthly_data[self.monthly_data['year_number'] == year]['total_headcount'].iloc[-1]
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        # Department headcounts at year-end
        for dept in self.config['headcount']:
            self.annual_data[f'{dept}_year_end_headcount'] = [
                self.monthly_data[self.monthly_data['year_number'] == year][f'{dept}_headcount'].iloc[-1]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]
        
        # Annual expense totals
        expense_categories = [
            'total_compensation', 'total_cogs', 'total_marketing_expenses',
            'total_sales_expenses', 'total_r_and_d_expenses', 
            'total_g_and_a_expenses', 'one_time_expenses',
            'total_operating_expenses', 'total_expenses'
        ]
        
        for category in expense_categories:
            self.annual_data[category] = [
                self.monthly_data[self.monthly_data['year_number'] == year][category].sum()
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]
        
        # Calculate annual ratios
        for year_idx in range(len(self.annual_data)):
            annual_arr = self.annual_data.loc[year_idx, 'avg_arr']
            if annual_arr > 0:
                self.annual_data.loc[year_idx, 'cogs_as_percent_of_arr'] = (
                    self.annual_data.loc[year_idx, 'total_cogs'] / annual_arr * 100
                )
                self.annual_data.loc[year_idx, 'opex_as_percent_of_arr'] = (
                    self.annual_data.loc[year_idx, 'total_operating_expenses'] / annual_arr * 100
                )
                self.annual_data.loc[year_idx, 'expenses_as_percent_of_arr'] = (
                    self.annual_data.loc[year_idx, 'total_expenses'] / annual_arr * 100
                )
            else:
                self.annual_data.loc[year_idx, 'cogs_as_percent_of_arr'] = float('nan')
                self.annual_data.loc[year_idx, 'opex_as_percent_of_arr'] = float('nan')
                self.annual_data.loc[year_idx, 'expenses_as_percent_of_arr'] = float('nan')
                
            # Calculate expense breakdown percentages
            total_expenses = self.annual_data.loc[year_idx, 'total_expenses']
            if total_expenses > 0:
                self.annual_data.loc[year_idx, 'compensation_percent_of_total'] = (
                    self.annual_data.loc[year_idx, 'total_compensation'] / total_expenses * 100
                )
                self.annual_data.loc[year_idx, 'cogs_percent_of_total'] = (
                    self.annual_data.loc[year_idx, 'total_cogs'] / total_expenses * 100
                )
                self.annual_data.loc[year_idx, 'marketing_percent_of_total'] = (
                    self.annual_data.loc[year_idx, 'total_marketing_expenses'] / total_expenses * 100
                )
                self.annual_data.loc[year_idx, 'sales_percent_of_total'] = (
                    self.annual_data.loc[year_idx, 'total_sales_expenses'] / total_expenses * 100
                )
                self.annual_data.loc[year_idx, 'r_and_d_percent_of_total'] = (
                    self.annual_data.loc[year_idx, 'total_r_and_d_expenses'] / total_expenses * 100
                )
                self.annual_data.loc[year_idx, 'g_and_a_percent_of_total'] = (
                    self.annual_data.loc[year_idx, 'total_g_and_a_expenses'] / total_expenses * 100
                )
    
    def plot_expense_breakdown(self, figsize=(12, 8)):
        """
        Plot expense breakdown by category
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plot
            
        Returns:
        --------
        matplotlib.figure : Figure object
        """
        if not self._model_run:
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get expense categories and their annual totals
        expense_categories = {
            'COGS': self.annual_data['total_cogs'],
            'Compensation': self.annual_data['total_compensation'],
            'Marketing': self.annual_data['total_marketing_expenses'],
            'Sales': self.annual_data['total_sales_expenses'],
            'R&D': self.annual_data['total_r_and_d_expenses'],
            'G&A': self.annual_data['total_g_and_a_expenses'],
            'One-time': self.annual_data['one_time_expenses']
        }
        
        # Create stacked bar chart
        bottom = np.zeros(len(self.annual_data))
        for label, values in expense_categories.items():
            ax.bar(self.annual_data['year'], values, bottom=bottom, label=label)
            bottom += values
        
        ax.set_title('Annual Expense Breakdown', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Expenses ($)')
        ax.legend()
        
        # Add total expense labels
        for i, year in enumerate(self.annual_data['year']):
            total = self.annual_data.loc[i, 'total_expenses']
            ax.text(year, total * 1.02, f"${total/1000000:.1f}M", ha='center', fontweight='bold')
        
        return fig
    
    def plot_expense_ratios(self, figsize=(12, 8)):
        """
        Plot expense ratios as percentage of ARR
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plot
            
        Returns:
        --------
        matplotlib.figure : Figure object
        """
        if not self._model_run:
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot expense ratios
        ax.plot(self.annual_data['year'], self.annual_data['cogs_as_percent_of_arr'], 
                marker='o', linewidth=2, label='COGS % of ARR')
        ax.plot(self.annual_data['year'], self.annual_data['opex_as_percent_of_arr'], 
                marker='s', linewidth=2, label='OpEx % of ARR')
        ax.plot(self.annual_data['year'], self.annual_data['expenses_as_percent_of_arr'], 
                marker='^', linewidth=3, label='Total Expenses % of ARR')
        
        ax.set_title('Expense Ratios as Percentage of ARR', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of ARR (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add ratio labels for total expenses
        for i, year in enumerate(self.annual_data['year']):
            ratio = self.annual_data.loc[i, 'expenses_as_percent_of_arr']
            if not np.isnan(ratio):
                ax.text(year, ratio * 1.02, f"{ratio:.1f}%", ha='center', fontweight='bold')
        
        return fig
    
    def plot_headcount_growth(self, figsize=(12, 8)):
        """
        Plot headcount growth by department
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plot
            
        Returns:
        --------
        matplotlib.figure : Figure object
        """
        if not self._model_run:
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot stacked bar chart of headcount by department
        departments = list(self.config['headcount'].keys())
        bottom = np.zeros(len(self.annual_data))
        
        for dept in departments:
            values = self.annual_data[f'{dept}_year_end_headcount']
            ax.bar(self.annual_data['year'], values, bottom=bottom, label=dept.capitalize())
            bottom += values
        
        ax.set_title('Year-End Headcount by Department', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Headcount')
        ax.legend()
        
        # Add total headcount labels
        for i, year in enumerate(self.annual_data['year']):
            total = self.annual_data.loc[i, 'year_end_headcount']
            ax.text(year, total * 1.02, f"{int(total)}", ha='center', fontweight='bold')
        
        return fig
    
    def calculate_cac(self, revenue_model):
        """
        Calculate Customer Acquisition Cost (CAC) metrics
        
        Parameters:
        -----------
        revenue_model : object
            Revenue model instance with monthly_data containing customer acquisition data
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with CAC metrics
        """
        if not self._model_run:
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
            
        if not hasattr(revenue_model, 'monthly_data') or 'total_new_customers' not in revenue_model.monthly_data.columns:
            raise ValueError("Revenue model must contain monthly_data with total_new_customers column")
        
        # Create CAC metrics dataframe
        cac_data = pd.DataFrame({
            'year': range(1, (self.config['projection_months'] // 12) + 1),
        })
        
        # Calculate annual new customers
        cac_data['new_customers'] = [
            revenue_model.monthly_data[revenue_model.monthly_data['year_number'] == year]['total_new_customers'].sum()
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        # Calculate sales and marketing expenses
        cac_data['sales_expenses'] = self.annual_data['total_sales_expenses']
        cac_data['marketing_expenses'] = self.annual_data['total_marketing_expenses']
        cac_data['sales_marketing_expenses'] = cac_data['sales_expenses'] + cac_data['marketing_expenses']
        
        # Calculate CAC metrics
        cac_data['cac'] = cac_data['sales_marketing_expenses'] / cac_data['new_customers']
        
        # Calculate CAC payback period (in months) if LTV data is available
        if 'total_arr' in revenue_model.monthly_data.columns:
            # Calculate average ARR per customer
            cac_data['avg_arr_per_customer'] = [
                revenue_model.monthly_data[revenue_model.monthly_data['year_number'] == year]['total_arr'].iloc[-1] / 
                revenue_model.monthly_data[revenue_model.monthly_data['year_number'] == year]['total_customers'].iloc[-1]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]
            
            # Use gross margin to calculate contribution margin
            cac_data['gross_margin_percent'] = 1 - (self.annual_data['total_cogs'] / self.annual_data['avg_arr'])
            
            # Calculate CAC payback period (in months)
            cac_data['cac_payback_months'] = (
                cac_data['cac'] / (cac_data['avg_arr_per_customer'] * cac_data['gross_margin_percent'] / 12)
            )
        
        return cac_data
    
    def calculate_cashflow(self, revenue_model):
        """
        Calculate monthly and annual cash flow
        
        Parameters:
        -----------
        revenue_model : object
            Revenue model instance with monthly_data containing ARR data
            
        Returns:
        --------
        tuple : (monthly_cashflow DataFrame, annual_cashflow DataFrame)
        """
        if not self._model_run:
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
            
        if not hasattr(revenue_model, 'monthly_data') or 'total_arr' not in revenue_model.monthly_data.columns:
            raise ValueError("Revenue model must contain monthly_data with total_arr column")
        
        # Create monthly cash flow dataframe
        monthly_cashflow = self.monthly_data.copy()
        
        # Calculate monthly revenue (ARR / 12)
        monthly_cashflow['monthly_revenue'] = monthly_cashflow['total_arr'] / 12
        
        # Calculate gross profit
        monthly_cashflow['gross_profit'] = monthly_cashflow['monthly_revenue'] - monthly_cashflow['total_cogs']
        monthly_cashflow['gross_margin_percent'] = monthly_cashflow['gross_profit'] / monthly_cashflow['monthly_revenue'] * 100
        
        # Calculate operating income
        monthly_cashflow['operating_income'] = monthly_cashflow['gross_profit'] - monthly_cashflow['total_operating_expenses']
        
        # Calculate cash flow
        monthly_cashflow['cash_flow'] = monthly_cashflow['operating_income']  # Simplified (ignoring taxes, capex, etc.)
        monthly_cashflow['cumulative_cash_flow'] = monthly_cashflow['cash_flow'].cumsum()
        
        # Create annual cash flow summary
        annual_cashflow = pd.DataFrame({
            'year': range(1, (self.config['projection_months'] // 12) + 1),
        })
        
        # Annual revenue
        annual_cashflow['annual_revenue'] = [
            monthly_cashflow[monthly_cashflow['year_number'] == year]['monthly_revenue'].sum()
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        # Annual gross profit
        annual_cashflow['annual_gross_profit'] = [
            monthly_cashflow[monthly_cashflow['year_number'] == year]['gross_profit'].sum()
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        # Annual operating income
        annual_cashflow['annual_operating_income'] = [
            monthly_cashflow[monthly_cashflow['year_number'] == year]['operating_income'].sum()
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        # Annual cash flow
        annual_cashflow['annual_cash_flow'] = [
            monthly_cashflow[monthly_cashflow['year_number'] == year]['cash_flow'].sum()
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        # Calculate margins
        annual_cashflow['gross_margin_percent'] = annual_cashflow['annual_gross_profit'] / annual_cashflow['annual_revenue'] * 100
        annual_cashflow['operating_margin_percent'] = annual_cashflow['annual_operating_income'] / annual_cashflow['annual_revenue'] * 100
        
        # Year-end cumulative cash flow
        annual_cashflow['year_end_cumulative_cash_flow'] = [
            monthly_cashflow[monthly_cashflow['year_number'] == year]['cumulative_cash_flow'].iloc[-1]
            for year in range(1, (self.config['projection_months'] // 12) + 1)
        ]
        
        return monthly_cashflow, annual_cashflow
    
    def plot_cashflow(self, monthly_cashflow, annual_cashflow, figsize=(15, 10)):
        """
        Plot cash flow metrics
        
        Parameters:
        -----------
        monthly_cashflow : pandas.DataFrame
            Monthly cash flow data
        annual_cashflow : pandas.DataFrame
            Annual cash flow data
        figsize : tuple, optional
            Figure size for the plots
            
        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plots
        """
        # Create figure with 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        
        # Plot monthly operating income and cumulative cash flow
        ax = axs[0]
        ax.bar(monthly_cashflow['month_number'], monthly_cashflow['operating_income'], 
              alpha=0.6, label='Monthly Operating Income')
        ax.set_xlabel('Month')
        ax.set_ylabel('Monthly Operating Income ($)')
        ax.grid(True, alpha=0.3)
        
        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                   ha='center', va='top', backgroundcolor='white', alpha=0.8)
        
        # Add cumulative cash flow on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(monthly_cashflow['month_number'], monthly_cashflow['cumulative_cash_flow'], 
                color='red', linewidth=2, label='Cumulative Cash Flow')
        ax2.set_ylabel('Cumulative Cash Flow ($)', color='red')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot annual metrics
        ax = axs[1]
        
        # Create stacked bar chart for revenue, gross profit, and operating income
        width = 0.35
        ax.bar(annual_cashflow['year'], annual_cashflow['annual_revenue'], 
              width, label='Revenue')
        ax.bar(annual_cashflow['year'], annual_cashflow['annual_gross_profit'], 
              width, label='Gross Profit')
        ax.bar(annual_cashflow['year'], annual_cashflow['annual_operating_income'], 
              width, label='Operating Income')
        
        ax.set_title('Annual Financial Performance', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Amount ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add operating margin on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(annual_cashflow['year'], annual_cashflow['operating_margin_percent'], 
                marker='o', color='red', linewidth=2, label='Operating Margin')
        ax2.set_ylabel('Operating Margin (%)', color='red')
        ax2.set_ylim([-50, 50])
        
        # Add margin labels
        for i, year in enumerate(annual_cashflow['year']):
            margin = annual_cashflow.loc[i, 'operating_margin_percent']
            ax2.text(year, margin + np.sign(margin)*2, f"{margin:.1f}%", 
                    ha='center', color='red', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def display_summary_metrics(self):
        """
        Display summary metrics as a table
        
        Returns:
        --------
        pandas.DataFrame : Formatted summary table
        """
        if not self._model_run:
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
        
        # Create a summary dataframe
        summary = pd.DataFrame(
            index=['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5', 'Year 6'])
            
        # Add key metrics
        summary['Headcount'] = self.annual_data['year_end_headcount'].values
        summary['Comp Expenses ($M)'] = (self.annual_data['total_compensation'] / 1000000).values
        summary['COGS ($M)'] = (self.annual_data['total_cogs'] / 1000000).values
        summary['Marketing ($M)'] = (self.annual_data['total_marketing_expenses'] / 1000000).values
        summary['R&D ($M)'] = (self.annual_data['total_r_and_d_expenses'] / 1000000).values
        summary['Total Expenses ($M)'] = (self.annual_data['total_expenses'] / 1000000).values
        summary['COGS % of ARR'] = self.annual_data['cogs_as_percent_of_arr'].values
        summary['OpEx % of ARR'] = self.annual_data['opex_as_percent_of_arr'].values
        
        # Format the summary table
        formatted_summary = summary.copy()
        for col in ['Comp Expenses ($M)', 'COGS ($M)', 'Marketing ($M)', 'R&D ($M)', 'Total Expenses ($M)']:
            formatted_summary[col] = formatted_summary[col].map('${:,.1f}M'.format)
            
        for col in ['COGS % of ARR', 'OpEx % of ARR']:
            formatted_summary[col] = formatted_summary[col].map('{:,.1f}%'.format)
            
        return formatted_summary
    
    def calculate_unit_economics(self, revenue_model):
        """
        Calculate unit economics metrics including LTV, CAC, and LTV/CAC ratio
        
        Parameters:
        -----------
        revenue_model : object
            Revenue model instance with customer and ARR data
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with unit economics metrics
        """
        if not self._model_run:
            raise RuntimeError("Model has not been run yet. Call run_model() first.")
            
        if not hasattr(revenue_model, 'monthly_data'):
            raise ValueError("Revenue model must contain monthly_data")
            
        # Create unit economics dataframe
        unit_econ = pd.DataFrame({
            'year': range(1, (self.config['projection_months'] // 12) + 1),
        })
        
        # Calculate ARPU (Average Revenue Per User)
        for i, year in enumerate(unit_econ['year']):
            # Get year-end values
            year_data = revenue_model.monthly_data[revenue_model.monthly_data['year_number'] == year]
            if len(year_data) > 0 and year_data['total_customers'].iloc[-1] > 0:
                unit_econ.loc[i, 'arpu'] = (
                    year_data['total_arr'].iloc[-1] / year_data['total_customers'].iloc[-1]
                )
            else:
                unit_econ.loc[i, 'arpu'] = 0
        
        # Calculate gross margin
        unit_econ['gross_margin_percent'] = [
            (1 - self.annual_data['total_cogs'].iloc[i] / 
            (self.annual_data['avg_arr'].iloc[i] if self.annual_data['avg_arr'].iloc[i] > 0 else 1)) * 100
            for i in range(len(unit_econ))
        ]
        
        # Calculate contribution margin per customer
        unit_econ['contribution_margin'] = unit_econ['arpu'] * unit_econ['gross_margin_percent'] / 100
        
        # Get churn rate from revenue model if available
        if hasattr(revenue_model, 'annual_data') and 'total_churn_rate' in revenue_model.annual_data.columns:
            unit_econ['churn_rate_percent'] = revenue_model.annual_data['total_churn_rate'] * 100
        else:
            # Estimate churn rate from monthly data if possible
            try:
                annual_churn_rates = []
                for year in range(1, 7):
                    year_data = revenue_model.monthly_data[revenue_model.monthly_data['year_number'] == year]
                    if 'total_churned_customers' in year_data.columns and 'total_customers' in year_data.columns:
                        total_churned = year_data['total_churned_customers'].sum()
                        avg_customers = year_data['total_customers'].mean()
                        if avg_customers > 0:
                            annual_churn_rates.append((total_churned / avg_customers) * 100)
                        else:
                            annual_churn_rates.append(0)
                    else:
                        annual_churn_rates.append(np.nan)
                unit_econ['churn_rate_percent'] = annual_churn_rates[:len(unit_econ)]
            except:
                # Default churn rate if calculation fails
                unit_econ['churn_rate_percent'] = 15  # Assume 15% annual churn
        
        # Calculate LTV (Lifetime Value)
        unit_econ['customer_lifetime_years'] = 1 / (unit_econ['churn_rate_percent'] / 100)
        unit_econ['ltv'] = unit_econ['contribution_margin'] * unit_econ['customer_lifetime_years']
        
        # Calculate CAC (Customer Acquisition Cost)
        cac_data = self.calculate_cac(revenue_model)
        unit_econ['cac'] = cac_data['cac']
        
        # Calculate LTV/CAC ratio
        unit_econ['ltv_cac_ratio'] = unit_econ['ltv'] / unit_econ['cac']
        
        # Calculate CAC payback period (in months)
        unit_econ['cac_payback_months'] = (
            unit_econ['cac'] / (unit_econ['contribution_margin'] / 12)
        )
        
        return unit_econ
    
    def plot_unit_economics(self, unit_econ, figsize=(15, 10)):
        """
        Plot unit economics metrics
        
        Parameters:
        -----------
        unit_econ : pandas.DataFrame
            Unit economics data
        figsize : tuple, optional
            Figure size for the plots
            
        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plots
        """
        # Create figure with 2 subplots
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        
        # Plot LTV and CAC
        ax = axs[0]
        ax.bar(unit_econ['year'] - 0.2, unit_econ['ltv'], width=0.4, label='LTV')
        ax.bar(unit_econ['year'] + 0.2, unit_econ['cac'], width=0.4, label='CAC')
        
        ax.set_title('Customer Lifetime Value vs. Acquisition Cost', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Amount ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add LTV/CAC ratio on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(unit_econ['year'], unit_econ['ltv_cac_ratio'], 
                marker='o', color='red', linewidth=2, label='LTV/CAC Ratio')
        ax2.set_ylabel('LTV/CAC Ratio', color='red')
        
        # Add ratio labels
        for i, year in enumerate(unit_econ['year']):
            ratio = unit_econ.loc[i, 'ltv_cac_ratio']
            ax2.text(year, ratio + 0.2, f"{ratio:.1f}x", 
                    ha='center', color='red', fontweight='bold')
        
        # Plot CAC Payback Period
        ax = axs[1]
        ax.bar(unit_econ['year'], unit_econ['cac_payback_months'], 
              alpha=0.6, label='CAC Payback Period (Months)')
        
        ax.set_title('CAC Payback Period', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Months')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add ideal threshold line (e.g., 12 months)
        ax.axhline(y=12, color='green', linestyle='--', label='12-Month Threshold')
        
        # Add payback period labels
        for i, year in enumerate(unit_econ['year']):
            months = unit_econ.loc[i, 'cac_payback_months']
            ax.text(year, months + 0.5, f"{months:.1f}", 
                   ha='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def display_unit_economics_table(self, unit_econ):
        """
        Display unit economics metrics as a table
        
        Parameters:
        -----------
        unit_econ : pandas.DataFrame
            Unit economics data
            
        Returns:
        --------
        pandas.DataFrame : Formatted summary table
        """
        # Create a summary dataframe
        summary = pd.DataFrame(
            index=['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5', 'Year 6'][:len(unit_econ)])
            
        # Add key metrics
        summary['ARPU ($)'] = unit_econ['arpu'].values
        summary['Gross Margin (%)'] = unit_econ['gross_margin_percent'].values
        summary['Churn Rate (%)'] = unit_econ['churn_rate_percent'].values
        summary['Customer Lifetime (Years)'] = unit_econ['customer_lifetime_years'].values
        summary['LTV ($)'] = unit_econ['ltv'].values
        summary['CAC ($)'] = unit_econ['cac'].values
        summary['LTV/CAC Ratio'] = unit_econ['ltv_cac_ratio'].values
        summary['CAC Payback (Months)'] = unit_econ['cac_payback_months'].values
        
        # Format the summary table
        formatted_summary = summary.copy()
        formatted_summary['ARPU ($)'] = formatted_summary['ARPU ($)'].map('${:,.0f}'.format)
        formatted_summary['Gross Margin (%)'] = formatted_summary['Gross Margin (%)'].map('{:,.1f}%'.format)
        formatted_summary['Churn Rate (%)'] = formatted_summary['Churn Rate (%)'].map('{:,.1f}%'.format)
        formatted_summary['Customer Lifetime (Years)'] = formatted_summary['Customer Lifetime (Years)'].map('{:,.1f}'.format)
        formatted_summary['LTV ($)'] = formatted_summary['LTV ($)'].map('${:,.0f}'.format)
        formatted_summary['CAC ($)'] = formatted_summary['CAC ($)'].map('${:,.0f}'.format)
        formatted_summary['LTV/CAC Ratio'] = formatted_summary['LTV/CAC Ratio'].map('{:,.1f}x'.format)
        formatted_summary['CAC Payback (Months)'] = formatted_summary['CAC Payback (Months)'].map('{:,.1f}'.format)
        
        return formatted_summary


# Example usage
if __name__ == "__main__":
    # Create a standalone cost model with default configuration
    cost_model = AISaaSCostModel()
    
    # Run the model (without revenue data)
    monthly_data, annual_data = cost_model.run_model()
    
    # Plot expense breakdown
    expense_fig = cost_model.plot_expense_breakdown()
    plt.savefig('expense_breakdown.png')
    
    # Plot headcount growth
    headcount_fig = cost_model.plot_headcount_growth()
    plt.savefig('headcount_growth.png')
    
    # Display summary metrics
    summary = cost_model.display_summary_metrics()
    print(summary)
    
    # Save data
    monthly_data.to_csv('monthly_cost_data.csv', index=False)
    annual_data.to_csv('annual_cost_data.csv', index=False)