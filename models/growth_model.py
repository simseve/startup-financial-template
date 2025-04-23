import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from IPython.display import display


class SaaSGrowthModel:
    """
    SaaS Growth Model based on S-curve acquisition patterns.
    Models customer acquisition, churn, and ARR for multiple segments over time.
    """

    def __init__(self, config=None):
        """
        Initialize the SaaS growth model with configuration parameters

        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for the model
        """
        # Set default configuration if none provided
        if config is None:
            self.config = self._get_default_config()
        else:
            self.config = config

        # Initialize results dataframe
        self.monthly_data = None
        self.annual_data = None
        self._model_run = False

    # Output accessors - added for modular design
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

    def get_config(self):
        """
        Returns the current model configuration

        Returns:
        --------
        dict : Current model configuration
        """
        return self.config.copy()

    def set_config(self, config):
        """
        Update the model configuration

        Parameters:
        -----------
        config : dict
            New configuration parameters for the model

        Notes:
        ------
        Requires re-running the model after configuration changes
        """
        self.config = config
        self._model_run = False  # Reset run flag since config changed

    def _get_default_config(self):
        """
        Returns a minimal default configuration. For a more comprehensive configuration,
        it's recommended to pass a custom configuration when initializing the model.
        """
        return {
            # Basic parameters
            'start_date': '2025-01-01',
            'projection_months': 72,  # 6 years
            'segments': ['Enterprise', 'Mid-Market', 'SMB'],

            # Empty dictionaries for segment-specific parameters
            # These would need to be populated for a functional model
            'initial_arr': {},
            'initial_customers': {},
            'contract_length': {},
            'churn_rates': {},
            's_curve': {},
            'seasonality': {},
            
            # Annual price increases (% increase per year)
            'annual_price_increases': {
                'Enterprise': 0.0,  # Default 0% annual increase
                'Mid-Market': 0.0,  # Default 0% annual increase
                'SMB': 0.0          # Default 0% annual increase
            }
        }

    def run_model(self):
        """
        Run the growth model for the configured time period
        """
        self._initialize_dataframes()

        # Set initial values
        for segment in self.config['segments']:
            self.monthly_data.loc[0,
                                  f'{segment}_customers'] = self.config['initial_customers'][segment]
            self.monthly_data.loc[0, f'{segment}_arr'] = (
                self.config['initial_customers'][segment] *
                self.config['initial_arr'][segment]
            )

        # Initialize cohort tracking
        cohort_data = {}
        for segment in self.config['segments']:
            cohort_data[segment] = {}
            # Start with initial customers as month 0 cohort
            cohort_data[segment][0] = self.config['initial_customers'][segment]

        # Calculate totals for first month
        self._calculate_totals(0)

        # Run the model month by month
        for month_idx in range(1, self.config['projection_months']):
            year_number = (month_idx // 12) + 1
            month_of_year = (month_idx % 12) + 1

            # Process each segment
            for segment in self.config['segments']:
                # Apply s-curve growth model for new customers
                new_customers = self._calculate_new_customers(
                    segment, year_number, month_of_year, month_idx)

                # Track new customers as a cohort
                if month_idx not in cohort_data[segment]:
                    cohort_data[segment][month_idx] = new_customers

                # Calculate churn based on contract duration for previous cohorts
                churned_customers = 0
                # Check each previous cohort
                for cohort_start, cohort_size in list(cohort_data[segment].items()):
                    contract_months = int(
                        self.config['contract_length'][segment] * 12)
                    # If this cohort is up for renewal (contract period has elapsed)
                    if cohort_start < month_idx and (month_idx - cohort_start) % contract_months == 0:
                        # Apply churn rate to the cohort
                        churn_rate = self.config['churn_rates'][segment]
                        cohort_churn = int(np.round(cohort_size * churn_rate))
                        churned_customers += cohort_churn
                        # Update the cohort size after churn
                        cohort_data[segment][cohort_start] -= cohort_churn

                # Get the current year and check if it's the start of a new year
                current_year = self.monthly_data.loc[month_idx, 'year_number']
                prev_year = self.monthly_data.loc[month_idx - 1, 'year_number']
                is_new_year = current_year > prev_year
                
                # Get the current price per customer
                # For existing customers, we'll derive from previous values
                prev_customers = self.monthly_data.loc[month_idx - 1, f'{segment}_customers']
                prev_arr = self.monthly_data.loc[month_idx - 1, f'{segment}_arr']
                
                # Calculate current price per customer
                current_price = self.config['initial_arr'][segment]
                if prev_customers > 0:
                    current_price = prev_arr / prev_customers
                
                # Apply annual price increase if it's the start of a new year (except year 1)
                price_increase = self.config['annual_price_increases'].get(segment, 0.0)
                if is_new_year and current_year > 1 and price_increase > 0:
                    current_price = current_price * (1 + price_increase)
                
                # Calculate ARR changes
                new_arr = new_customers * current_price
                churned_arr = churned_customers * current_price
                
                # Update customer counts
                current_customers = prev_customers + new_customers - churned_customers
                
                # Update ARR with price adjustments
                if is_new_year and current_year > 1 and price_increase > 0:
                    # Apply price increase to all existing customers
                    existing_customers = prev_customers - churned_customers
                    price_adjustment = existing_customers * (current_price - (prev_arr / prev_customers)) if prev_customers > 0 else 0
                    current_arr = prev_arr + new_arr - churned_arr + price_adjustment
                else:
                    # No price adjustment needed
                    current_arr = prev_arr + new_arr - churned_arr

                # Store results
                self.monthly_data.loc[month_idx,
                                      f'{segment}_new_customers'] = new_customers
                self.monthly_data.loc[month_idx,
                                      f'{segment}_churned_customers'] = churned_customers
                self.monthly_data.loc[month_idx,
                                      f'{segment}_customers'] = current_customers
                self.monthly_data.loc[month_idx,
                                      f'{segment}_new_arr'] = new_arr
                self.monthly_data.loc[month_idx,
                                      f'{segment}_churned_arr'] = churned_arr
                self.monthly_data.loc[month_idx,
                                      f'{segment}_arr'] = current_arr

                # Store cohort distribution
                cohort_counts = []
                for c in sorted(cohort_data[segment].keys()):
                    cohort_counts.append(cohort_data[segment][c])
                self.monthly_data.loc[month_idx, f'{segment}_cohort_distribution'] = str(
                    cohort_counts)

            # Calculate totals
            self._calculate_totals(month_idx)

        # Generate annual summary
        self._generate_annual_summary()

        self._model_run = True  # Mark model as run

        return self.monthly_data, self.annual_data

    def _initialize_dataframes(self):
        """
        Initialize dataframes to store model results
        """
        # Date range for projections
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        date_range = [start_date + timedelta(days=30*i)
                      for i in range(self.config['projection_months'])]

        # Initialize monthly dataframe
        self.monthly_data = pd.DataFrame({
            'date': date_range,
            'year': [d.year for d in date_range],
            'month': [d.month for d in date_range],
            'month_number': range(1, len(date_range) + 1),
            'year_number': [(i // 12) + 1 for i in range(len(date_range))],
        })

        # Add segment-specific columns
        segments = self.config['segments']
        for segment in segments:
            # Customer counts
            self.monthly_data[f'{segment}_customers'] = 0
            self.monthly_data[f'{segment}_new_customers'] = 0
            self.monthly_data[f'{segment}_churned_customers'] = 0

            # Revenue
            self.monthly_data[f'{segment}_new_arr'] = 0.0
            self.monthly_data[f'{segment}_churned_arr'] = 0.0
            self.monthly_data[f'{segment}_arr'] = 0.0

            # Cohort tracking
            self.monthly_data[f'{segment}_cohort_distribution'] = ''

        # Total columns
        self.monthly_data['total_customers'] = 0
        self.monthly_data['total_new_customers'] = 0
        self.monthly_data['total_churned_customers'] = 0
        self.monthly_data['total_arr'] = 0.0
        self.monthly_data['total_new_arr'] = 0.0
        self.monthly_data['total_churned_arr'] = 0.0

    def _calculate_new_customers(self, segment, year_number, month_of_year, month_idx):
        """
        Calculate new customers for a segment using S-curve growth model

        Parameters:
        -----------
        segment : str
            Customer segment name
        year_number : int
            Current year (1-6)
        month_of_year : int
            Current month (1-12)
        month_idx : int
            Overall month index

        Returns:
        --------
        int : Number of new customers
        """
        # Get S-curve parameters for this year and segment
        s_params = self.config['s_curve'][segment][year_number]

        # Get month position in the year (0-11)
        month_position = month_of_year - 1

        # Midpoint of the S-curve (when growth is at half the maximum)
        midpoint = s_params['midpoint'] - 1  # Convert to 0-indexed

        # Steepness of the curve
        steepness = s_params['steepness']

        # Maximum monthly new customers at peak of S-curve
        max_monthly = s_params['max_monthly']

        # Apply S-curve formula:
        # new_customers = max_monthly / (1 + e^(-steepness * (month - midpoint)))
        s_curve_value = max_monthly / \
            (1 + np.exp(-steepness * (month_position - midpoint)))

        # Apply seasonality factor
        seasonality_factor = self.config['seasonality'][month_of_year]
        seasonal_value = s_curve_value * seasonality_factor

        # Convert to integer and ensure non-negative
        new_customers = max(0, int(np.round(seasonal_value)))

        return new_customers

    def _calculate_totals(self, month_idx):
        """
        Calculate totals across all segments for a given month

        Parameters:
        -----------
        month_idx : int
            Month index in the projection
        """
        # List of metrics to sum
        metrics = [
            'customers', 'new_customers', 'churned_customers',
            'arr', 'new_arr', 'churned_arr'
        ]

        # Sum each metric across segments
        for metric in metrics:
            self.monthly_data.loc[month_idx, f'total_{metric}'] = sum(
                self.monthly_data.loc[month_idx, f'{segment}_{metric}']
                for segment in self.config['segments']
            )

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
                self.monthly_data[self.monthly_data['year_number']
                                  == year]['date'].iloc[0]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ],
            'year_end_date': [
                self.monthly_data[self.monthly_data['year_number']
                                  == year]['date'].iloc[-1]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]
        })

        # Calculate annual metrics for each segment
        for segment in self.config['segments']:
            # Year-end values
            self.annual_data[f'{segment}_ending_customers'] = [
                self.monthly_data[self.monthly_data['year_number']
                                  == year][f'{segment}_customers'].iloc[-1]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]

            self.annual_data[f'{segment}_ending_arr'] = [
                self.monthly_data[self.monthly_data['year_number']
                                  == year][f'{segment}_arr'].iloc[-1]
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]

            # Annual new and churned
            self.annual_data[f'{segment}_new_customers'] = [
                self.monthly_data[self.monthly_data['year_number']
                                  == year][f'{segment}_new_customers'].sum()
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]

            self.annual_data[f'{segment}_churned_customers'] = [
                self.monthly_data[self.monthly_data['year_number']
                                  == year][f'{segment}_churned_customers'].sum()
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]

            # ARR metrics
            self.annual_data[f'{segment}_new_arr'] = [
                self.monthly_data[self.monthly_data['year_number']
                                  == year][f'{segment}_new_arr'].sum()
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]

            self.annual_data[f'{segment}_churned_arr'] = [
                self.monthly_data[self.monthly_data['year_number']
                                  == year][f'{segment}_churned_arr'].sum()
                for year in range(1, (self.config['projection_months'] // 12) + 1)
            ]

            # Calculate growth rates
            self.annual_data[f'{segment}_customer_growth_rate'] = np.zeros(
                len(self.annual_data))
            for i in range(len(self.annual_data)):
                if i == 0:
                    # First year growth over initial customers
                    initial_customers = self.config['initial_customers'][segment]
                    ending_customers = self.annual_data.loc[i,
                                                            f'{segment}_ending_customers']
                    if initial_customers > 0:
                        growth_rate = (ending_customers /
                                       initial_customers) - 1
                    else:
                        growth_rate = float(
                            'inf') if ending_customers > 0 else 0
                else:
                    # Year-over-year growth
                    prev_ending = self.annual_data.loc[i -
                                                       1, f'{segment}_ending_customers']
                    current_ending = self.annual_data.loc[i,
                                                          f'{segment}_ending_customers']
                    if prev_ending > 0:
                        growth_rate = (current_ending / prev_ending) - 1
                    else:
                        growth_rate = float('inf') if current_ending > 0 else 0

                self.annual_data.loc[i,
                                     f'{segment}_customer_growth_rate'] = growth_rate

            # Calculate ARR growth rates
            self.annual_data[f'{segment}_arr_growth_rate'] = np.zeros(
                len(self.annual_data))
            for i in range(len(self.annual_data)):
                if i == 0:
                    # First year growth over initial ARR
                    initial_arr = self.config['initial_customers'][segment] * \
                        self.config['initial_arr'][segment]
                    ending_arr = self.annual_data.loc[i,
                                                      f'{segment}_ending_arr']
                    if initial_arr > 0:
                        growth_rate = (ending_arr / initial_arr) - 1
                    else:
                        growth_rate = float('inf') if ending_arr > 0 else 0
                else:
                    # Year-over-year growth
                    prev_ending = self.annual_data.loc[i -
                                                       1, f'{segment}_ending_arr']
                    current_ending = self.annual_data.loc[i,
                                                          f'{segment}_ending_arr']
                    if prev_ending > 0:
                        growth_rate = (current_ending / prev_ending) - 1
                    else:
                        growth_rate = float('inf') if current_ending > 0 else 0

                self.annual_data.loc[i,
                                     f'{segment}_arr_growth_rate'] = growth_rate

        # Calculate totals
        metrics_to_total = [
            'ending_customers', 'ending_arr', 'new_customers', 'churned_customers',
            'new_arr', 'churned_arr'
        ]

        for metric in metrics_to_total:
            self.annual_data[f'total_{metric}'] = sum(
                self.annual_data[f'{segment}_{metric}']
                for segment in self.config['segments']
            )

        # Calculate total growth rates
        self.annual_data['total_customer_growth_rate'] = np.zeros(
            len(self.annual_data))
        self.annual_data['total_arr_growth_rate'] = np.zeros(
            len(self.annual_data))

        for i in range(len(self.annual_data)):
            # Customer growth rate
            if i == 0:
                initial_total = sum(self.config['initial_customers'].values())
                ending_total = self.annual_data.loc[i,
                                                    'total_ending_customers']
                if initial_total > 0:
                    growth_rate = (ending_total / initial_total) - 1
                else:
                    growth_rate = float('inf') if ending_total > 0 else 0
            else:
                prev_ending = self.annual_data.loc[i -
                                                   1, 'total_ending_customers']
                current_ending = self.annual_data.loc[i,
                                                      'total_ending_customers']
                if prev_ending > 0:
                    growth_rate = (current_ending / prev_ending) - 1
                else:
                    growth_rate = float('inf') if current_ending > 0 else 0

            self.annual_data.loc[i, 'total_customer_growth_rate'] = growth_rate

            # ARR growth rate
            if i == 0:
                initial_arr = sum(
                    self.config['initial_customers'][segment] *
                    self.config['initial_arr'][segment]
                    for segment in self.config['segments']
                )
                ending_arr = self.annual_data.loc[i, 'total_ending_arr']
                if initial_arr > 0:
                    growth_rate = (ending_arr / initial_arr) - 1
                else:
                    growth_rate = float('inf') if ending_arr > 0 else 0
            else:
                prev_ending = self.annual_data.loc[i-1, 'total_ending_arr']
                current_ending = self.annual_data.loc[i, 'total_ending_arr']
                if prev_ending > 0:
                    growth_rate = (current_ending / prev_ending) - 1
                else:
                    growth_rate = float('inf') if current_ending > 0 else 0

            self.annual_data.loc[i, 'total_arr_growth_rate'] = growth_rate

    def apply_growth_profile(self, profile_name):
        """
        Apply a predefined growth profile to the model and return a new model instance.
        
        Parameters:
        -----------
        profile_name : str
            The name of the growth profile to apply ('baseline', 'conservative', 
            'aggressive', 'hypergrowth')
            
        Returns:
        --------
        SaaSGrowthModel : A new instance with the updated configuration
        
        Notes:
        ------
        This method modifies the max_monthly parameters of the s-curve for all segments
        to match the selected growth profile.
        """
        if profile_name not in ['baseline', 'conservative', 'aggressive', 'hypergrowth']:
            raise ValueError(
                f"Invalid profile name: {profile_name}. Must be one of: 'baseline', 'conservative', 'aggressive', 'hypergrowth'")
        
        # Create a deep copy of the current configuration
        import copy
        new_config = copy.deepcopy(self.config)
        
        # Define multipliers for each profile relative to baseline
        profile_multipliers = {
            'baseline': 1.0,
            'conservative': 0.7,  # 30% slower growth than baseline
            'aggressive': 1.5,    # 50% faster growth than baseline
            'hypergrowth': 2.5    # 150% faster growth than baseline
        }
        
        # Get the multiplier for the selected profile
        multiplier = profile_multipliers[profile_name]
        
        # Apply the multiplier to max_monthly for each segment and year
        for segment in new_config['segments']:
            for year in range(1, 7):
                # Scale the max_monthly parameter by the multiplier
                new_config['s_curve'][segment][year]['max_monthly'] = int(
                    round(new_config['s_curve'][segment][year]['max_monthly'] * multiplier)
                )
        
        # Create a new model instance with the updated configuration
        new_model = SaaSGrowthModel(new_config)
        return new_model
        
    def apply_custom_segment_profiles(self, segment_multipliers):
        """
        Apply custom growth multipliers to different segments
        
        Parameters:
        -----------
        segment_multipliers : dict
            Dictionary with segments as keys and multipliers as values
            Example: {'Enterprise': 1.2, 'Mid-Market': 0.9, 'SMB': 1.5}
            
        Returns:
        --------
        SaaSGrowthModel : A new instance with the updated configuration
        """
        # Validate the segment multipliers
        for segment in segment_multipliers:
            if segment not in self.config['segments']:
                raise ValueError(f"Invalid segment: {segment}")
            
            if not isinstance(segment_multipliers[segment], (int, float)) or segment_multipliers[segment] <= 0:
                raise ValueError(f"Multiplier for {segment} must be a positive number")
        
        # Create a deep copy of the current configuration
        import copy
        new_config = copy.deepcopy(self.config)
        
        # Apply the multipliers to max_monthly for each segment and year
        for segment, multiplier in segment_multipliers.items():
            for year in range(1, 7):
                # Scale the max_monthly parameter by the multiplier
                new_config['s_curve'][segment][year]['max_monthly'] = int(
                    round(new_config['s_curve'][segment][year]['max_monthly'] * multiplier)
                )
        
        # Create a new model instance with the updated configuration
        new_model = SaaSGrowthModel(new_config)
        return new_model
        
    def apply_dynamic_growth_strategy(self, segment_year_multipliers):
        """
        Apply different growth multipliers for each segment by year
        
        Parameters:
        -----------
        segment_year_multipliers : dict
            Dictionary with segments as keys and year-specific multipliers as values
            Example: {
                'Enterprise': {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.8, 5: 0.5, 6: 0.3},
                'Mid-Market': {1: 1.0, 2: 1.5, 3: 2.0, 4: 1.5, 5: 1.0, 6: 0.8},
                'SMB': {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0}
            }
            
        Returns:
        --------
        SaaSGrowthModel : A new instance with the updated configuration
        """
        # Validate the segment-year multipliers
        for segment, year_multipliers in segment_year_multipliers.items():
            if segment not in self.config['segments']:
                raise ValueError(f"Invalid segment: {segment}")
                
            for year, multiplier in year_multipliers.items():
                if year not in range(1, 7):
                    raise ValueError(f"Year {year} not valid. Must be between 1 and 6.")
                if not isinstance(multiplier, (int, float)) or multiplier < 0:
                    raise ValueError(f"Multiplier for {segment} year {year} must be a non-negative number")
        
        # Create a deep copy of the current configuration
        import copy
        new_config = copy.deepcopy(self.config)
        
        # Apply the year-specific multipliers for each segment
        for segment, year_multipliers in segment_year_multipliers.items():
            for year, multiplier in year_multipliers.items():
                # Scale the max_monthly parameter by the multiplier
                new_config['s_curve'][segment][year]['max_monthly'] = int(
                    round(new_config['s_curve'][segment][year]['max_monthly'] * multiplier)
                )
        
        # Create a new model instance with the updated configuration
        new_model = SaaSGrowthModel(new_config)
        return new_model

    def tune_s_curve_steepness(self, segment, steepness_values=None):
        """
        Update the steepness parameters of the S-curve for a specific segment
        and return a copy of the model for analysis.

        Parameters:
        -----------
        segment : str
            The segment to tune ('Enterprise', 'Mid-Market', 'SMB')
        steepness_values : dict, optional
            Dictionary with years as keys and steepness values as values.
            Example: {1: 0.7, 2: 0.8}. If a year is not specified, the 
            original steepness value is retained.

        Returns:
        --------
        SaaSGrowthModel : A new instance with the updated configuration

        Examples:
        ---------
        >>> model = SaaSGrowthModel()
        >>> tuned_model = model.tune_s_curve_steepness('Enterprise', {1: 0.7, 2: 0.8})
        >>> tuned_model.run_model()
        >>> tuned_model.plot_growth_curves()
        """
        if segment not in self.config['segments']:
            raise ValueError(
                f"Segment '{segment}' not found in model configuration")

        # Create a deep copy of the current configuration
        import copy
        new_config = copy.deepcopy(self.config)

        # Update steepness values for the specified segment
        if steepness_values:
            for year, steepness in steepness_values.items():
                if year not in new_config['s_curve'][segment]:
                    raise ValueError(
                        f"Year {year} not found in model configuration")
                new_config['s_curve'][segment][year]['steepness'] = steepness

        # Create a new model instance with the updated configuration
        new_model = SaaSGrowthModel(new_config)
        return new_model

    def compare_steepness_scenarios(self, segment, scenario_steepness, figsize=(15, 10)):
        """
        Compare different steepness scenarios for a segment by visualizing
        the impact on customer acquisition and ARR.

        Parameters:
        -----------
        segment : str
            The segment to analyze ('Enterprise', 'Mid-Market', 'SMB')
        scenario_steepness : dict
            Dictionary with scenario names as keys and steepness configurations as values.
            Example: {
                'Base': {1: 0.5, 2: 0.6},
                'Aggressive': {1: 0.7, 2: 0.8}
            }
        figsize : tuple, optional
            Figure size for the plots

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the comparison plots
        """
        if segment not in self.config['segments']:
            raise ValueError(
                f"Segment '{segment}' not found in model configuration")

        # Run the base model if it hasn't been run yet
        if not self._model_run:
            self.run_model()

        # Create scenarios
        scenarios = {'Current': self}
        for name, steepness_config in scenario_steepness.items():
            model = self.tune_s_curve_steepness(segment, steepness_config)
            model.run_model()
            scenarios[name] = model

        # Create comparison plots
        fig, axs = plt.subplots(2, 1, figsize=figsize)

        # 1. Monthly customer acquisition for the segment
        ax = axs[0]

        for name, model in scenarios.items():
            ax.plot(
                model.monthly_data['month_number'],
                model.monthly_data[f'{segment}_new_customers'],
                label=f'{name}'
            )

        ax.set_title(
            f'Monthly New {segment} Customer Acquisition - Steepness Comparison', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('New Customers')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        # 2. Cumulative customers for the segment
        ax = axs[1]

        for name, model in scenarios.items():
            ax.plot(
                model.monthly_data['month_number'],
                model.monthly_data[f'{segment}_customers'],
                label=f'{name}'
            )

        ax.set_title(
            f'Total {segment} Customers - Steepness Comparison', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Customer Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        plt.tight_layout()
        return fig

    def get_s_curve_parameters(self, segment=None):
        """
        Display the current S-curve parameters for all segments or a specific segment

        Parameters:
        -----------
        segment : str, optional
            If provided, only display parameters for this segment

        Returns:
        --------
        pd.DataFrame : DataFrame with S-curve parameters
        """
        segments = [segment] if segment else self.config['segments']

        # Validate segment
        for seg in segments:
            if seg not in self.config['segments']:
                raise ValueError(
                    f"Segment '{seg}' not found in model configuration")

        # Create a DataFrame to hold the parameters
        rows = []
        for seg in segments:
            for year in range(1, 7):
                params = self.config['s_curve'][seg][year]
                rows.append({
                    'Segment': seg,
                    'Year': year,
                    'Midpoint': params['midpoint'],
                    'Steepness': params['steepness'],
                    'Max Monthly': params['max_monthly']
                })

        return pd.DataFrame(rows)

    def apply_monthly_growth_pattern(self, segment_month_multipliers):
        """
        Apply different growth multipliers for each segment by month for maximum flexibility
        
        Parameters:
        -----------
        segment_month_multipliers : dict
            Dictionary with segments as keys and month-specific multipliers as values
            Example: {
                'Enterprise': {1: 1.5, 2: 1.6, ..., 72: 0.5},
                'Mid-Market': {1: 0.8, 2: 0.9, ..., 72: 2.0},
                'SMB': {1: 0.5, 2: 0.5, ..., 72: 3.0}
            }
            
        Returns:
        --------
        SaaSGrowthModel : A new instance with the updated configuration
        """
        # Create a deep copy of the current configuration
        import copy
        new_config = copy.deepcopy(self.config)
        
        # Create a custom s-curve parameter set that will override the standard s-curve parameters
        if 'custom_monthly_multipliers' not in new_config:
            new_config['custom_monthly_multipliers'] = {}
        
        # Store the custom multipliers
        for segment, month_multipliers in segment_month_multipliers.items():
            if segment not in self.config['segments']:
                raise ValueError(f"Invalid segment: {segment}")
                
            new_config['custom_monthly_multipliers'][segment] = month_multipliers
        
        # Create a new model instance with the updated configuration
        new_model = SaaSGrowthModel(new_config)
        
        # Override the _calculate_new_customers method to use custom multipliers when available
        def custom_calculate_new_customers(self, segment, year_number, month_of_year, month_idx):
            """Custom method that uses monthly multipliers when available"""
            # Check if we have custom monthly multipliers for this segment and month
            if ('custom_monthly_multipliers' in self.config and 
                segment in self.config['custom_monthly_multipliers'] and 
                month_idx + 1 in self.config['custom_monthly_multipliers'][segment]):
                
                # Get the original calculation
                original_result = SaaSGrowthModel._calculate_new_customers(
                    self, segment, year_number, month_of_year, month_idx)
                
                # Apply the custom multiplier
                custom_multiplier = self.config['custom_monthly_multipliers'][segment][month_idx + 1]
                return max(0, int(np.round(original_result * custom_multiplier)))
            else:
                # Use the original method if no custom multiplier exists
                return SaaSGrowthModel._calculate_new_customers(
                    self, segment, year_number, month_of_year, month_idx)
        
        # Replace the method in the new model instance
        import types
        new_model._calculate_new_customers = types.MethodType(custom_calculate_new_customers, new_model)
        
        return new_model
    
    def create_growth_acceleration_strategy(self, target_segments=None, acceleration_years=None, 
                                         deceleration_years=None, accel_multiplier=2.0, 
                                         decel_multiplier=0.5):
        """
        Create a growth strategy with acceleration and deceleration phases
        
        Parameters:
        -----------
        target_segments : list, optional
            List of segments to apply the strategy to. If None, applies to all segments.
        acceleration_years : list, optional
            List of years to accelerate growth. If None, no acceleration.
        deceleration_years : list, optional
            List of years to decelerate growth. If None, no deceleration.
        accel_multiplier : float, optional
            Multiplier to apply during acceleration years.
        decel_multiplier : float, optional
            Multiplier to apply during deceleration years.
            
        Returns:
        --------
        SaaSGrowthModel : A new instance with the updated configuration
        
        Example:
        --------
        # Accelerate Enterprise and Mid-Market in years 1-2, decelerate in years 5-6
        model.create_growth_acceleration_strategy(
            target_segments=['Enterprise', 'Mid-Market'],
            acceleration_years=[1, 2],
            deceleration_years=[5, 6],
            accel_multiplier=2.5,
            decel_multiplier=0.4
        )
        """
        # Use all segments if none specified
        if target_segments is None:
            target_segments = self.config['segments']
            
        # Validate segments
        for segment in target_segments:
            if segment not in self.config['segments']:
                raise ValueError(f"Invalid segment: {segment}")
                
        # Initialize year multipliers for each segment
        segment_year_multipliers = {}
        for segment in target_segments:
            segment_year_multipliers[segment] = {}
            
            # Set default multiplier of 1.0 for all years
            for year in range(1, 7):
                segment_year_multipliers[segment][year] = 1.0
                
            # Apply acceleration multipliers
            if acceleration_years:
                for year in acceleration_years:
                    if 1 <= year <= 6:
                        segment_year_multipliers[segment][year] = accel_multiplier
                        
            # Apply deceleration multipliers
            if deceleration_years:
                for year in deceleration_years:
                    if 1 <= year <= 6:
                        segment_year_multipliers[segment][year] = decel_multiplier
        
        # Use the dynamic growth strategy method to apply these multipliers
        return self.apply_dynamic_growth_strategy(segment_year_multipliers)
        
    def plot_growth_curves(self, figsize=(15, 15), highlight_customizations=False):
        """
        Plot growth curves for customers and ARR

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plots
        highlight_customizations : bool, optional
            If True, highlight areas with custom growth multipliers

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plots
        """
        if self.monthly_data is None:
            raise ValueError(
                "Model has not been run yet. Call run_model() first.")

        # Create figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=figsize)

        # 1. Monthly customer acquisition
        ax = axs[0]

        for segment in self.config['segments']:
            ax.plot(
                self.monthly_data['month_number'],
                self.monthly_data[f'{segment}_new_customers'],
                label=f'{segment} New'
            )
            
            # Highlight customized months if requested
            if highlight_customizations and 'custom_monthly_multipliers' in self.config and segment in self.config['custom_monthly_multipliers']:
                for month, multiplier in self.config['custom_monthly_multipliers'][segment].items():
                    if 1 <= month <= len(self.monthly_data):
                        if multiplier > 1.1:  # Accelerated
                            ax.axvspan(month-0.5, month+0.5, alpha=0.2, color='green')
                        elif multiplier < 0.9:  # Decelerated
                            ax.axvspan(month-0.5, month+0.5, alpha=0.2, color='red')

        ax.set_title(
            'Monthly New Customer Acquisition by Segment', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('New Customers')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        # 2. Total customers over time
        ax = axs[1]

        for segment in self.config['segments']:
            ax.plot(
                self.monthly_data['month_number'],
                self.monthly_data[f'{segment}_customers'],
                label=segment
            )

        # Add total line
        ax.plot(
            self.monthly_data['month_number'],
            self.monthly_data['total_customers'],
            label='Total',
            linewidth=2,
            color='black'
        )

        ax.set_title('Total Customers by Segment Over Time', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Customer Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        # 3. Monthly ARR
        ax = axs[2]

        for segment in self.config['segments']:
            ax.plot(
                self.monthly_data['month_number'],
                self.monthly_data[f'{segment}_arr'] /
                1000000,  # Convert to millions
                label=segment
            )

        # Add total line
        ax.plot(
            self.monthly_data['month_number'],
            self.monthly_data['total_arr'] / 1000000,  # Convert to millions
            label='Total',
            linewidth=2,
            color='black'
        )

        ax.set_title('Monthly ARR by Segment Over Time', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('ARR ($ Millions)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        # 4. Churn analysis
        ax = axs[3]

        for segment in self.config['segments']:
            # Plot churn as bars
            ax.bar(
                self.monthly_data['month_number'],
                self.monthly_data[f'{segment}_churned_customers'],
                alpha=0.6,
                label=f'{segment} Churn'
            )

        ax.set_title('Monthly Customer Churn by Segment', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Customers Churned')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        plt.tight_layout()
        return fig

    def plot_annual_metrics(self, figsize=(15, 10)):
        """
        Plot annual summary metrics

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plots

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plots
        """
        if self.annual_data is None:
            raise ValueError(
                "Model has not been run yet. Call run_model() first.")

        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=figsize)

        # 1. Annual ending customers by segment
        ax = axs[0]

        # Create stacked bar chart
        bottoms = np.zeros(len(self.annual_data))

        for segment in self.config['segments']:
            ax.bar(
                self.annual_data['year'],
                self.annual_data[f'{segment}_ending_customers'],
                bottom=bottoms,
                label=segment
            )
            bottoms += self.annual_data[f'{segment}_ending_customers']

        # Add customer count labels
        for i, year in enumerate(self.annual_data['year']):
            ax.text(
                year,
                self.annual_data.loc[i, 'total_ending_customers'] * 1.02,
                f"{int(self.annual_data.loc[i, 'total_ending_customers'])}",
                ha='center',
                fontweight='bold'
            )

        ax.set_title('Year-End Customers by Segment', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Customers')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Annual ending ARR by segment
        ax = axs[1]

        # Create stacked bar chart
        bottoms = np.zeros(len(self.annual_data))

        for segment in self.config['segments']:
            ax.bar(
                self.annual_data['year'],
                self.annual_data[f'{segment}_ending_arr'] /
                1000000,  # Convert to millions
                bottom=bottoms,
                label=segment
            )
            bottoms += self.annual_data[f'{segment}_ending_arr'] / 1000000

        # Add ARR labels
        for i, year in enumerate(self.annual_data['year']):
            ax.text(
                year,
                self.annual_data.loc[i, 'total_ending_arr'] / 1000000 * 1.02,
                f"${self.annual_data.loc[i, 'total_ending_arr']/1000000:.1f}M",
                ha='center',
                fontweight='bold'
            )

        ax.set_title('Year-End ARR by Segment ($ Millions)', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('ARR ($ Millions)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Growth rates
        ax = axs[2]

        # Plot growth rates
        years = self.annual_data['year']

        # Plot growth rates as lines
        for segment in self.config['segments']:
            ax.plot(
                years,
                # Convert to percentage
                self.annual_data[f'{segment}_arr_growth_rate'] * 100,
                marker='o',
                label=f'{segment} ARR Growth'
            )

        # Add total growth rate
        ax.plot(
            years,
            self.annual_data['total_arr_growth_rate'] * 100,
            marker='s',
            linewidth=3,
            color='black',
            label='Total ARR Growth'
        )

        # Add growth rate labels for total
        for i, year in enumerate(years):
            ax.text(
                year,
                self.annual_data.loc[i, 'total_arr_growth_rate'] * 100 * 1.05,
                f"{self.annual_data.loc[i, 'total_arr_growth_rate']*100:.1f}%",
                ha='center',
                fontweight='bold'
            )

        ax.set_title('Annual ARR Growth Rate by Segment', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('Growth Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_customer_segment_shares(self, figsize=(12, 6)):
        """
        Plot customer segment shares over time

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size for the plots

        Returns:
        --------
        fig : matplotlib figure
            Matplotlib figure with the plots
        """
        if self.monthly_data is None:
            raise ValueError(
                "Model has not been run yet. Call run_model() first.")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate percentage shares
        segment_shares = pd.DataFrame(index=self.monthly_data.index)

        for segment in self.config['segments']:
            segment_shares[segment] = (
                self.monthly_data[f'{segment}_customers'] /
                self.monthly_data['total_customers'] * 100
            )

        # Plot area chart
        ax.stackplot(
            self.monthly_data['month_number'],
            [segment_shares[segment] for segment in self.config['segments']],
            labels=self.config['segments'],
            alpha=0.7
        )

        ax.set_title('Customer Segment Share Over Time', fontsize=14)
        ax.set_xlabel('Month')
        ax.set_ylabel('Share (%)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add year boundaries
        for year in range(2, 7):
            month = (year - 1) * 12 + 1
            ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
            ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                    ha='center', va='top', backgroundcolor='white', alpha=0.8)

        return fig

    def display_summary_metrics(self):
        """
        Display summary metrics as a table
        """
        if self.annual_data is None:
            raise ValueError(
                "Model has not been run yet. Call run_model() first.")

        # Create a summary dataframe
        summary = pd.DataFrame(
            index=['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5', 'Year 6'])

        # Add key metrics
        summary['Total Customers'] = self.annual_data['total_ending_customers'].values
        summary['Total ARR ($M)'] = (
            self.annual_data['total_ending_arr'] / 1000000).values
        summary['ARR Growth (%)'] = (
            self.annual_data['total_arr_growth_rate'] * 100).values
        summary['New Customers'] = self.annual_data['total_new_customers'].values
        summary['Churned Customers'] = self.annual_data['total_churned_customers'].values
        summary['Churn Rate (%)'] = (self.annual_data['total_churned_customers'] /
                                     self.annual_data['total_ending_customers'].shift(1).fillna(
            sum(self.config['initial_customers'].values(
            ))
        ) * 100).values

        # Format the summary table
        formatted_summary = summary.copy()
        formatted_summary['Total ARR ($M)'] = formatted_summary['Total ARR ($M)'].map(
            '${:,.1f}M'.format)
        formatted_summary['ARR Growth (%)'] = formatted_summary['ARR Growth (%)'].map(
            '{:,.1f}%'.format)
        formatted_summary['Churn Rate (%)'] = formatted_summary['Churn Rate (%)'].map(
            '{:,.1f}%'.format)

        return formatted_summary

    def optimize_for_breakeven(self, financial_model_class, monthly_fixed_costs, variable_cost_pct, target_month, min_multiplier=0.6, max_multiplier=3.0, steps=10):
        """
        Optimize the growth profile to achieve breakeven by a target month
        
        Parameters:
        -----------
        financial_model_class : class
            Financial model class to use for optimization
        monthly_fixed_costs : dict
            Dictionary of fixed monthly costs
        variable_cost_pct : float
            Variable costs as a percentage of revenue (0-1)
        target_month : int
            Target month to achieve breakeven
        min_multiplier : float, optional
            Minimum growth multiplier to try
        max_multiplier : float, optional
            Maximum growth multiplier to try
        steps : int, optional
            Number of multiplier steps to try
            
        Returns:
        --------
        tuple : (best_model, best_multiplier, achieved_month, metrics)
            best_model : SaaSGrowthModel
                Model with the best growth profile
            best_multiplier : float
                Growth multiplier that achieved breakeven closest to target
            achieved_month : int
                Month when breakeven was achieved
            metrics : dict
                Dictionary of optimization metrics
        """
        if not isinstance(monthly_fixed_costs, dict):
            raise ValueError("monthly_fixed_costs must be a dictionary")
        
        if not 0 <= variable_cost_pct <= 1:
            raise ValueError("variable_cost_pct must be between 0 and 1")
        
        if not 1 <= target_month <= self.config['projection_months']:
            raise ValueError(f"target_month must be between 1 and {self.config['projection_months']}")
        
        # Calculate multipliers to try
        multipliers = np.linspace(min_multiplier, max_multiplier, steps)
        
        # Track results
        results = []
        
        print(f"Optimizing for breakeven at month {target_month}...")
        print(f"Trying {steps} growth multipliers from {min_multiplier:.2f}x to {max_multiplier:.2f}x")
        
        # Try each multiplier
        for multiplier in multipliers:
            # Apply growth multiplier to all segments
            profile_model = self.apply_custom_segment_profiles({
                segment: multiplier for segment in self.config['segments']
            })
            
            # Run the growth model
            profile_model.run_model()
            
            # Create and run a minimal financial model to check for breakeven
            revenue_data = profile_model.get_monthly_data()
            
            # Calculate costs from the growth model's revenue projections
            cost_data = pd.DataFrame({
                'month_number': revenue_data['month_number'],
                'monthly_revenue': revenue_data['total_arr'] / 12,
                'variable_costs': (revenue_data['total_arr'] / 12) * variable_cost_pct,
                'fixed_costs': sum(monthly_fixed_costs.values())
            })
            
            # Calculate profit/loss
            cost_data['profit'] = cost_data['monthly_revenue'] - cost_data['variable_costs'] - cost_data['fixed_costs']
            
            # Find breakeven month
            profitable_months = cost_data[cost_data['profit'] >= 0]
            
            if len(profitable_months) > 0:
                breakeven_month = profitable_months['month_number'].min()
                
                # Store the result
                results.append({
                    'multiplier': multiplier,
                    'breakeven_month': breakeven_month,
                    'model': profile_model,
                    'distance': abs(breakeven_month - target_month)
                })
            else:
                # No breakeven within projection period
                results.append({
                    'multiplier': multiplier,
                    'breakeven_month': float('inf'),
                    'model': profile_model,
                    'distance': float('inf')
                })
        
        # Find the best result (closest to target month)
        if not results:
            raise ValueError("No valid results found during optimization")
        
        # Sort by distance to target month
        results.sort(key=lambda x: x['distance'])
        best_result = results[0]
        
        print(f"Optimization complete!")
        print(f"Best multiplier: {best_result['multiplier']:.2f}x")
        
        if best_result['breakeven_month'] == float('inf'):
            print(f"No breakeven achieved within the projection period")
            achieved_month = None
        else:
            print(f"Breakeven achieved at month {best_result['breakeven_month']} (target: {target_month})")
            achieved_month = best_result['breakeven_month']
        
        # Return the best model, multiplier, achieved month, and all metrics
        return best_result['model'], best_result['multiplier'], achieved_month, results
    
    def optimize_for_series_b(self, financial_model_class, monthly_fixed_costs, variable_cost_pct, 
                             target_month, target_arr=10000000, target_growth_rate=1.0,
                             min_multiplier=0.6, max_multiplier=4.0, steps=10):
        """
        Optimize the growth profile to achieve Series B qualification by a target month
        
        Parameters:
        -----------
        financial_model_class : class
            Financial model class to use for optimization
        monthly_fixed_costs : dict
            Dictionary of fixed monthly costs
        variable_cost_pct : float
            Variable costs as a percentage of revenue (0-1)
        target_month : int
            Target month to achieve Series B qualification
        target_arr : float, optional
            Target ARR for Series B qualification (default: $10M)
        target_growth_rate : float, optional
            Target ARR growth rate for Series B qualification (default: 100%)
        min_multiplier : float, optional
            Minimum growth multiplier to try
        max_multiplier : float, optional
            Maximum growth multiplier to try
        steps : int, optional
            Number of multiplier steps to try
            
        Returns:
        --------
        tuple : (best_model, best_multiplier, achieved_month, metrics)
            best_model : SaaSGrowthModel
                Model with the best growth profile
            best_multiplier : float
                Growth multiplier that achieved Series B qualification closest to target
            achieved_month : int
                Month when Series B qualification was achieved
            metrics : dict
                Dictionary of optimization metrics
        """
        if not isinstance(monthly_fixed_costs, dict):
            raise ValueError("monthly_fixed_costs must be a dictionary")
        
        if not 0 <= variable_cost_pct <= 1:
            raise ValueError("variable_cost_pct must be between 0 and 1")
        
        if not 1 <= target_month <= self.config['projection_months']:
            raise ValueError(f"target_month must be between 1 and {self.config['projection_months']}")
        
        # Calculate multipliers to try
        multipliers = np.linspace(min_multiplier, max_multiplier, steps)
        
        # Track results
        results = []
        
        print(f"Optimizing for Series B qualification at month {target_month}...")
        print(f"Target ARR: ${target_arr/1000000:.1f}M, Target growth rate: {target_growth_rate*100:.0f}%")
        print(f"Trying {steps} growth multipliers from {min_multiplier:.2f}x to {max_multiplier:.2f}x")
        
        # Try each multiplier
        for multiplier in multipliers:
            # Apply growth multiplier to all segments
            profile_model = self.apply_custom_segment_profiles({
                segment: multiplier for segment in self.config['segments']
            })
            
            # Run the growth model
            profile_model.run_model()
            
            # Get the data
            revenue_data = profile_model.get_monthly_data()
            
            # We need at least 12 months of data to calculate YoY growth
            if target_month < 13:
                continue
                
            # Check if we've met the ARR target
            arr_at_target = revenue_data.loc[target_month-1, 'total_arr']
            
            # Calculate YoY growth rate
            if target_month >= 13:  # Need at least 12 months for YoY growth
                arr_year_ago = revenue_data.loc[target_month-13, 'total_arr']
                yoy_growth_rate = (arr_at_target / arr_year_ago) - 1 if arr_year_ago > 0 else float('inf')
            else:
                yoy_growth_rate = float('inf')  # Assume infinite growth rate if less than a year
            
            # Check if we've met Series B criteria
            qualified = arr_at_target >= target_arr and yoy_growth_rate >= target_growth_rate
            
            # Find the month when the criteria are met
            qualification_month = float('inf')
            
            for month in range(12, len(revenue_data)):  # Start from month 13 to calculate YoY
                arr = revenue_data.loc[month, 'total_arr']
                arr_year_ago = revenue_data.loc[month-12, 'total_arr']
                growth = (arr / arr_year_ago) - 1 if arr_year_ago > 0 else float('inf')
                
                if arr >= target_arr and growth >= target_growth_rate:
                    qualification_month = month + 1  # Convert to 1-indexed
                    break
            
            # Store the result
            results.append({
                'multiplier': multiplier,
                'arr_at_target': arr_at_target,
                'growth_rate_at_target': yoy_growth_rate,
                'qualified_at_target': qualified,
                'qualification_month': qualification_month,
                'model': profile_model,
                'distance': abs(qualification_month - target_month) if qualification_month != float('inf') else float('inf')
            })
        
        # Find the best result (closest to target month)
        qualified_results = [r for r in results if r['qualification_month'] != float('inf')]
        
        if not qualified_results:
            # No qualifying profiles, return the one that gets closest
            results.sort(key=lambda x: (
                # First, sort by ARR as percentage of target
                -(x['arr_at_target'] / target_arr), 
                # Then by growth rate as percentage of target
                -(x['growth_rate_at_target'] / target_growth_rate) if target_growth_rate > 0 else 0
            ))
            best_result = results[0]
            achieved_month = None
        else:
            # Find the profile that hits the target closest to the desired month
            qualified_results.sort(key=lambda x: x['distance'])
            best_result = qualified_results[0]
            achieved_month = best_result['qualification_month']
        
        print(f"Optimization complete!")
        print(f"Best multiplier: {best_result['multiplier']:.2f}x")
        
        if achieved_month is None:
            print(f"Series B qualification criteria not met within the projection period")
            print(f"Best ARR at target month: ${best_result['arr_at_target']/1000000:.2f}M ({best_result['arr_at_target']/target_arr*100:.1f}% of target)")
            print(f"Growth rate at target month: {best_result['growth_rate_at_target']*100:.1f}%")
        else:
            print(f"Series B criteria met at month {achieved_month} (target: {target_month})")
        
        # Return the best model, multiplier, achieved month, and all metrics
        return best_result['model'], best_result['multiplier'], achieved_month, results
        
    def display_configuration(self):
        """
        Display the current model configuration
        """
        # Create a formatted representation of the configuration
        config_display = {}

        # Basic parameters
        config_display['General Parameters'] = {
            'Start Date': self.config['start_date'],
            'Projection Months': self.config['projection_months'],
            'Segments': ', '.join(self.config['segments'])
        }

        # Segment-specific parameters
        for segment in self.config['segments']:
            config_display[f'{segment} Parameters'] = {
                'Initial ARR': f"${self.config['initial_arr'][segment]:,}",
                'Initial Customers': self.config['initial_customers'][segment],
                'Contract Length (years)': self.config['contract_length'][segment],
                'Churn Rate': f"{self.config['churn_rates'][segment]*100:.1f}%",
                'Annual Price Increase': f"{self.config['annual_price_increases'].get(segment, 0.0)*100:.1f}%"
            }

            # Add S-curve parameters
            config_display[f'{segment} S-Curve Parameters'] = {}
            for year in range(1, 7):
                params = self.config['s_curve'][segment][year]
                config_display[f'{segment} S-Curve Parameters'][f'Year {year}'] = (
                    f"Mid: {params['midpoint']}, "
                    f"Steep: {params['steepness']}, "
                    f"Max: {params['max_monthly']} per month"
                )

        # Seasonality
        config_display['Seasonality Factors'] = {
            month: factor for month, factor in self.config['seasonality'].items()
        }

        return config_display


# Example usage
if __name__ == "__main__":
    # Create model with default configuration
    model = SaaSGrowthModel()

    # Run the model
    monthly_data, annual_data = model.run_model()

    # # Check current parameters
    # params_df = model.get_s_curve_parameters('Enterprise')
    # display(params_df)

    # # Create a tuned model for Enterprise with steeper curves in years 1-2
    # tuned_model = model.tune_s_curve_steepness('Enterprise', {1: 0.8, 2: 0.9})
    # tuned_model.run_model()

    # # Compare different steepness scenarios
    # comparison = model.compare_steepness_scenarios('Enterprise', {
    #     'Aggressive': {1: 0.8, 2: 0.9, 3: 1.0},
    #     'Conservative': {1: 0.3, 2: 0.4, 3: 0.5}
    # })
    # plt.show()

    # Plot results
    growth_fig = model.plot_growth_curves()
    plt.savefig('growth_curves.png')

    annual_fig = model.plot_annual_metrics()
    plt.savefig('annual_metrics.png')

    segment_fig = model.plot_customer_segment_shares()
    plt.savefig('segment_shares.png')

    # Display summary metrics
    summary = model.display_summary_metrics()
    print(summary)

    # Save data
    monthly_data.to_csv('monthly_data.csv', index=False)
    annual_data.to_csv('annual_data.csv', index=False)
