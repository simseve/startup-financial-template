import os
import json
import copy
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from io import BytesIO

from models.growth_model import SaaSGrowthModel
from models.cost_model import AISaaSCostModel
from models.financial_model import SaaSFinancialModel

# Load default configurations
def load_default_configs():
    with open(os.path.join('configs', 'revenue_config.json'), 'r') as f:
        revenue_config = json.load(f)
    
    with open(os.path.join('configs', 'cost_config.json'), 'r') as f:
        cost_config = json.load(f)
        
    # Convert string keys to integers for certain nested dictionaries
    # S-curves
    for segment in revenue_config['segments']:
        revenue_config['s_curve'][segment] = {
            int(year): params for year, params in revenue_config['s_curve'][segment].items()
        }
    
    # Seasonality
    revenue_config['seasonality'] = {
        int(month): factor for month, factor in revenue_config['seasonality'].items()
    }
    
    # Headcount growth factors
    for dept in cost_config['headcount']:
        cost_config['headcount'][dept]['growth_factors'] = {
            int(year): factor for year, factor in cost_config['headcount'][dept]['growth_factors'].items()
        }
    
    # Marketing efficiency
    cost_config['marketing_efficiency'] = {
        int(year): factor for year, factor in cost_config['marketing_efficiency'].items()
    }
    
    return revenue_config, cost_config

# Load initial configurations
default_revenue_config, default_cost_config = load_default_configs()

# Convert matplotlib figure to image for Gradio
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

# Run the model with given configurations
def run_model(revenue_config, cost_config, initial_investment=5000000):
    # Initialize models
    growth_model = SaaSGrowthModel(revenue_config)
    cost_model = AISaaSCostModel(cost_config)
    
    # Run growth model
    growth_model.run_model()
    
    # Run cost model with growth model
    cost_model.run_model(growth_model)
    
    # Initialize and run financial model
    financial_model = SaaSFinancialModel(
        revenue_model=growth_model,
        cost_model=cost_model,
        initial_investment=initial_investment
    )
    financial_model.run_model()
    
    return growth_model, cost_model, financial_model

# Generate images from model results
def generate_model_charts(growth_model, cost_model, financial_model):
    images = {}
    
    # Growth curves
    growth_curves_fig = growth_model.plot_growth_curves()
    images['growth_curves'] = fig_to_image(growth_curves_fig)
    
    # Annual metrics
    annual_metrics_fig = growth_model.plot_annual_metrics()
    images['annual_metrics'] = fig_to_image(annual_metrics_fig)
    
    # Segment shares
    segment_shares_fig = growth_model.plot_customer_segment_shares()
    images['segment_shares'] = fig_to_image(segment_shares_fig)
    
    # Cost breakdown
    expense_breakdown_fig = cost_model.plot_expense_breakdown()
    images['expense_breakdown'] = fig_to_image(expense_breakdown_fig)
    
    # Headcount growth
    headcount_growth_fig = cost_model.plot_headcount_growth()
    images['headcount_growth'] = fig_to_image(headcount_growth_fig)
    
    # Financial summary
    financial_summary_fig = financial_model.plot_financial_summary()
    images['financial_summary'] = fig_to_image(financial_summary_fig)
    
    # Break-even analysis
    break_even_fig = financial_model.plot_break_even_analysis()
    images['break_even'] = fig_to_image(break_even_fig)
    
    # Runway and capital
    runway_fig = financial_model.plot_runway_and_capital()
    images['runway'] = fig_to_image(runway_fig)
    
    # Unit economics
    unit_economics = cost_model.calculate_unit_economics(growth_model)
    unit_economics_fig = cost_model.plot_unit_economics(unit_economics)
    images['unit_economics'] = fig_to_image(unit_economics_fig)
    
    # Close all figures to avoid memory leaks
    plt.close('all')
    
    return images

# Generate summary tables from model results
def generate_summary_tables(growth_model, cost_model, financial_model):
    tables = {}
    
    # Growth summary
    growth_summary = growth_model.display_summary_metrics()
    tables['growth_summary'] = growth_summary
    
    # Cost summary
    cost_summary = cost_model.display_summary_metrics()
    tables['cost_summary'] = cost_summary
    
    # Key financial metrics
    key_metrics = financial_model.get_key_metrics_table()
    tables['key_metrics'] = key_metrics
    
    # Unit economics
    unit_economics = cost_model.calculate_unit_economics(growth_model)
    unit_economics_table = cost_model.display_unit_economics_table(unit_economics)
    tables['unit_economics'] = unit_economics_table
    
    return tables

# Generate a textual financial summary
def generate_financial_summary(financial_model, growth_model, cost_model):
    key_metrics = financial_model.get_key_metrics_table()
    
    summary = "Financial Summary:\n\n"
    
    # ARR growth
    if len(key_metrics.index) > 1:
        latest_year_idx = len(key_metrics.index) - 1
        prev_year_idx = latest_year_idx - 1
        
        prev_arr = float(key_metrics.iloc[prev_year_idx]['ARR ($M)'].replace('$', '').replace('M', ''))
        curr_arr = float(key_metrics.iloc[latest_year_idx]['ARR ($M)'].replace('$', '').replace('M', ''))
        arr_growth = ((curr_arr / prev_arr) - 1) * 100 if prev_arr > 0 else 0
        
        latest_year = key_metrics.index[latest_year_idx]
        summary += f"Year {latest_year} ARR: ${curr_arr:.1f}M (Growth: {arr_growth:.1f}%)\n"
    
    # Customer metrics
    if len(key_metrics.index) > 0:
        latest_year_idx = len(key_metrics.index) - 1
        total_customers = key_metrics.iloc[latest_year_idx]['Customers']
        summary += f"Total Customers: {total_customers}\n"
    
    # Profitability
    monthly_data = financial_model.get_monthly_data()
    if 'profitable_month' in monthly_data.columns:
        profitable_months = monthly_data[monthly_data['profitable_month'] == True]
        if len(profitable_months) > 0:
            first_profitable = profitable_months.iloc[0]
            month_number = first_profitable['month_number']
            year = first_profitable['year']
            summary += f"Break-even Point: Month {month_number} (Year {year}, Month {first_profitable['month']})\n"
        else:
            summary += "Break-even Point: Not achieved within projection period (6 years)\n"
    
    # Unit economics
    unit_economics = cost_model.calculate_unit_economics(growth_model)
    unit_econ_table = cost_model.display_unit_economics_table(unit_economics)
    
    if len(unit_econ_table.index) > 0:
        latest_year_idx = len(unit_econ_table.index) - 1
        latest_year = unit_econ_table.index[latest_year_idx]
        churn = unit_econ_table.loc[latest_year, 'Effective Churn Rate (%)']
        lifetime = unit_econ_table.loc[latest_year, 'Customer Lifetime (Years)']
        payback = unit_econ_table.loc[latest_year, 'CAC Payback (Months)']
        
        summary += f"\nUnit Economics (Year {latest_year}):\n"
        summary += f"Churn Rate: {churn} | Customer Lifetime: {lifetime} | CAC Payback: {payback}\n"
    
    return summary

# Create a visualization of S-curves with current parameters
def create_s_curve_preview(revenue_config, segment, scaling_factors=None):
    # If no scaling factors provided, use all 1.0
    if scaling_factors is None:
        scaling_factors = {year: 1.0 for year in range(1, 7)}
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get original parameters
    segment_params = revenue_config['s_curve'][segment]
    
    # For each year (1-6)
    for year in range(1, 7):
        # Get parameters
        params = segment_params[year]
        midpoint = params['midpoint'] - 1  # Convert to 0-indexed
        steepness = params['steepness']
        max_monthly = params['max_monthly']
        
        # Apply scaling factor
        max_monthly_scaled = max_monthly * scaling_factors.get(year, 1.0)
        
        # Generate x values (months 0-11 of the year)
        x = np.arange(12)
        
        # Calculate the s-curve values
        y = [max_monthly_scaled / (1 + np.exp(-steepness * (month - midpoint))) for month in x]
        
        # Plot the curve for this year
        ax.plot(x + (year-1)*12, y, label=f'Year {year}', marker='o')
        
    # Add labels and title
    ax.set_title(f'S-curves for {segment} Segment', fontsize=14)
    ax.set_xlabel('Month')
    ax.set_ylabel('New Customers per Month')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add year boundaries
    for year in range(2, 7):
        month = (year - 1) * 12
        ax.axvline(x=month, color='gray', linestyle='--', alpha=0.7)
        ax.text(month, ax.get_ylim()[1]*0.95, f'Year {year}',
                ha='center', va='top', backgroundcolor='white', alpha=0.8)
    
    # Convert to image
    img = fig_to_image(fig)
    plt.close(fig)
    
    return img

# Update the S-curve parameters in the revenue config
def update_s_curve_params(revenue_config, segment, year, midpoint, steepness, max_monthly):
    # Deep copy the config to avoid modifying the original
    new_config = copy.deepcopy(revenue_config)
    
    # Update the parameters
    new_config['s_curve'][segment][year]['midpoint'] = midpoint
    new_config['s_curve'][segment][year]['steepness'] = steepness
    new_config['s_curve'][segment][year]['max_monthly'] = max_monthly
    
    return new_config

# Apply scaling factors to S-curve max_monthly parameters
def apply_scaling_factors(revenue_config, segment, scaling_factors):
    new_config = copy.deepcopy(revenue_config)
    
    for year, factor in scaling_factors.items():
        year = int(year)
        if 1 <= year <= 6:
            # Get the original max_monthly value
            original_max = default_revenue_config['s_curve'][segment][year]['max_monthly']
            # Apply the scaling factor
            new_config['s_curve'][segment][year]['max_monthly'] = int(original_max * factor)
    
    return new_config

# Export configuration to a file
def export_config(config, config_type):
    fd, path = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as f:
        # Convert int keys to strings for JSON serialization
        config_copy = copy.deepcopy(config)
        
        if config_type == 'revenue':
            # Convert s_curve int keys to strings
            for segment in config_copy['segments']:
                config_copy['s_curve'][segment] = {
                    str(year): params for year, params in config_copy['s_curve'][segment].items()
                }
            
            # Convert seasonality int keys to strings
            config_copy['seasonality'] = {
                str(month): factor for month, factor in config_copy['seasonality'].items()
            }
        else:  # cost config
            # Convert headcount growth factors int keys to strings
            for dept in config_copy['headcount']:
                config_copy['headcount'][dept]['growth_factors'] = {
                    str(year): factor for year, factor in config_copy['headcount'][dept]['growth_factors'].items()
                }
            
            # Convert marketing efficiency int keys to strings
            config_copy['marketing_efficiency'] = {
                str(year): factor for year, factor in config_copy['marketing_efficiency'].items()
            }
        
        json.dump(config_copy, f, indent=2)
    return path

# Import configuration from a file
def import_config(file_path, config_type):
    with open(file_path.name, 'r') as f:
        config = json.load(f)
    
    # Convert string keys to integers for certain nested dictionaries
    if config_type == 'revenue':
        # S-curves
        for segment in config['segments']:
            config['s_curve'][segment] = {
                int(year): params for year, params in config['s_curve'][segment].items()
            }
        
        # Seasonality
        config['seasonality'] = {
            int(month): factor for month, factor in config['seasonality'].items()
        }
    else:  # cost config
        # Headcount growth factors
        for dept in config['headcount']:
            config['headcount'][dept]['growth_factors'] = {
                int(year): factor for year, factor in config['headcount'][dept]['growth_factors'].items()
            }
        
        # Marketing efficiency
        config['marketing_efficiency'] = {
            int(year): factor for year, factor in config['marketing_efficiency'].items()
        }
    
    return config

# Main function to process form updates and run the model
def process_form(
    # General parameters
    initial_investment,
    
    # Enterprise S-curve scaling factors
    enterprise_y1_scale, enterprise_y2_scale, enterprise_y3_scale,
    enterprise_y4_scale, enterprise_y5_scale, enterprise_y6_scale,
    
    # Mid-Market S-curve scaling factors
    midmarket_y1_scale, midmarket_y2_scale, midmarket_y3_scale,
    midmarket_y4_scale, midmarket_y5_scale, midmarket_y6_scale,
    
    # SMB S-curve scaling factors
    smb_y1_scale, smb_y2_scale, smb_y3_scale,
    smb_y4_scale, smb_y5_scale, smb_y6_scale,
    
    # Other parameters that could be added from the configs
    revenue_config, cost_config
):
    # Collect scaling factors
    enterprise_scaling = {
        1: enterprise_y1_scale, 2: enterprise_y2_scale, 3: enterprise_y3_scale, 
        4: enterprise_y4_scale, 5: enterprise_y5_scale, 6: enterprise_y6_scale
    }
    
    midmarket_scaling = {
        1: midmarket_y1_scale, 2: midmarket_y2_scale, 3: midmarket_y3_scale, 
        4: midmarket_y4_scale, 5: midmarket_y5_scale, 6: midmarket_y6_scale
    }
    
    smb_scaling = {
        1: smb_y1_scale, 2: smb_y2_scale, 3: smb_y3_scale, 
        4: smb_y4_scale, 5: smb_y5_scale, 6: smb_y6_scale
    }
    
    # Apply scaling factors to each segment
    updated_config = apply_scaling_factors(revenue_config, 'Enterprise', enterprise_scaling)
    updated_config = apply_scaling_factors(updated_config, 'Mid-Market', midmarket_scaling)
    updated_config = apply_scaling_factors(updated_config, 'SMB', smb_scaling)
    
    # Run the model with updated configuration
    growth_model, cost_model, financial_model = run_model(
        updated_config, cost_config, initial_investment
    )
    
    # Generate charts and tables
    model_charts = generate_model_charts(growth_model, cost_model, financial_model)
    summary_tables = generate_summary_tables(growth_model, cost_model, financial_model)
    
    # Generate text summary
    financial_summary = generate_financial_summary(financial_model, growth_model, cost_model)
    
    # Create S-curve previews
    enterprise_preview = create_s_curve_preview(updated_config, 'Enterprise')
    midmarket_preview = create_s_curve_preview(updated_config, 'Mid-Market')
    smb_preview = create_s_curve_preview(updated_config, 'SMB')
    
    # Return all outputs
    return (
        updated_config,  # Updated revenue config
        cost_config,     # Cost config (unchanged)
        model_charts['growth_curves'],
        model_charts['annual_metrics'],
        model_charts['segment_shares'],
        model_charts['expense_breakdown'],
        model_charts['headcount_growth'],
        model_charts['financial_summary'],
        model_charts['break_even'],
        model_charts['runway'],
        model_charts['unit_economics'],
        summary_tables['key_metrics'].to_html(classes='gradio-table'),
        summary_tables['growth_summary'].to_html(classes='gradio-table'),
        summary_tables['cost_summary'].to_html(classes='gradio-table'),
        summary_tables['unit_economics'].to_html(classes='gradio-table'),
        financial_summary,
        enterprise_preview,
        midmarket_preview,
        smb_preview
    )

# Handler for resetting to defaults
def reset_to_defaults():
    revenue_config, cost_config = load_default_configs()
    
    # Run the model with default configuration
    growth_model, cost_model, financial_model = run_model(
        revenue_config, cost_config, 5000000
    )
    
    # Generate charts and tables
    model_charts = generate_model_charts(growth_model, cost_model, financial_model)
    summary_tables = generate_summary_tables(growth_model, cost_model, financial_model)
    
    # Generate text summary
    financial_summary = generate_financial_summary(financial_model, growth_model, cost_model)
    
    # Create S-curve previews
    enterprise_preview = create_s_curve_preview(revenue_config, 'Enterprise')
    midmarket_preview = create_s_curve_preview(revenue_config, 'Mid-Market')
    smb_preview = create_s_curve_preview(revenue_config, 'SMB')
    
    # Return all outputs plus the default slider values
    return (
        revenue_config,
        cost_config,
        model_charts['growth_curves'],
        model_charts['annual_metrics'],
        model_charts['segment_shares'],
        model_charts['expense_breakdown'],
        model_charts['headcount_growth'],
        model_charts['financial_summary'],
        model_charts['break_even'],
        model_charts['runway'],
        model_charts['unit_economics'],
        summary_tables['key_metrics'].to_html(classes='gradio-table'),
        summary_tables['growth_summary'].to_html(classes='gradio-table'),
        summary_tables['cost_summary'].to_html(classes='gradio-table'),
        summary_tables['unit_economics'].to_html(classes='gradio-table'),
        financial_summary,
        enterprise_preview,
        midmarket_preview,
        smb_preview,
        5000000,  # Default initial investment
        # Default Enterprise scaling factors
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        # Default Mid-Market scaling factors
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        # Default SMB scaling factors
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )

# Handler for exporting revenue config
def export_revenue_config(revenue_config):
    return export_config(revenue_config, 'revenue')

# Handler for exporting cost config
def export_cost_config(cost_config):
    return export_config(cost_config, 'cost')

# Handler for importing revenue config
def import_revenue_config(file_path, current_revenue_config, current_cost_config):
    try:
        new_revenue_config = import_config(file_path, 'revenue')
        # Run the model with the new config
        growth_model, cost_model, financial_model = run_model(
            new_revenue_config, current_cost_config, 5000000
        )
        
        # Generate charts and tables
        model_charts = generate_model_charts(growth_model, cost_model, financial_model)
        summary_tables = generate_summary_tables(growth_model, cost_model, financial_model)
        
        # Generate text summary
        financial_summary = generate_financial_summary(financial_model, growth_model, cost_model)
        
        # Create S-curve previews
        enterprise_preview = create_s_curve_preview(new_revenue_config, 'Enterprise')
        midmarket_preview = create_s_curve_preview(new_revenue_config, 'Mid-Market')
        smb_preview = create_s_curve_preview(new_revenue_config, 'SMB')
        
        # Return all outputs
        return (
            new_revenue_config,
            current_cost_config,
            model_charts['growth_curves'],
            model_charts['annual_metrics'],
            model_charts['segment_shares'],
            model_charts['expense_breakdown'],
            model_charts['headcount_growth'],
            model_charts['financial_summary'],
            model_charts['break_even'],
            model_charts['runway'],
            model_charts['unit_economics'],
            summary_tables['key_metrics'].to_html(classes='gradio-table'),
            summary_tables['growth_summary'].to_html(classes='gradio-table'),
            summary_tables['cost_summary'].to_html(classes='gradio-table'),
            summary_tables['unit_economics'].to_html(classes='gradio-table'),
            financial_summary,
            enterprise_preview,
            midmarket_preview,
            smb_preview,
            f"Successfully imported revenue configuration from {file_path.name}"
        )
    except Exception as e:
        return (
            current_revenue_config,
            current_cost_config,
            None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            f"Error importing revenue configuration: {str(e)}"
        )

# Handler for importing cost config
def import_cost_config(file_path, current_revenue_config, current_cost_config):
    try:
        new_cost_config = import_config(file_path, 'cost')
        # Run the model with the new config
        growth_model, cost_model, financial_model = run_model(
            current_revenue_config, new_cost_config, 5000000
        )
        
        # Generate charts and tables
        model_charts = generate_model_charts(growth_model, cost_model, financial_model)
        summary_tables = generate_summary_tables(growth_model, cost_model, financial_model)
        
        # Generate text summary
        financial_summary = generate_financial_summary(financial_model, growth_model, cost_model)
        
        # Create S-curve previews
        enterprise_preview = create_s_curve_preview(current_revenue_config, 'Enterprise')
        midmarket_preview = create_s_curve_preview(current_revenue_config, 'Mid-Market')
        smb_preview = create_s_curve_preview(current_revenue_config, 'SMB')
        
        # Return all outputs
        return (
            current_revenue_config,
            new_cost_config,
            model_charts['growth_curves'],
            model_charts['annual_metrics'],
            model_charts['segment_shares'],
            model_charts['expense_breakdown'],
            model_charts['headcount_growth'],
            model_charts['financial_summary'],
            model_charts['break_even'],
            model_charts['runway'],
            model_charts['unit_economics'],
            summary_tables['key_metrics'].to_html(classes='gradio-table'),
            summary_tables['growth_summary'].to_html(classes='gradio-table'),
            summary_tables['cost_summary'].to_html(classes='gradio-table'),
            summary_tables['unit_economics'].to_html(classes='gradio-table'),
            financial_summary,
            enterprise_preview,
            midmarket_preview,
            smb_preview,
            f"Successfully imported cost configuration from {file_path.name}"
        )
    except Exception as e:
        return (
            current_revenue_config,
            current_cost_config,
            None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None,
            f"Error importing cost configuration: {str(e)}"
        )

# Initialize default configs and models
revenue_config, cost_config = load_default_configs()
growth_model, cost_model, financial_model = run_model(revenue_config, cost_config)
model_charts = generate_model_charts(growth_model, cost_model, financial_model)
summary_tables = generate_summary_tables(growth_model, cost_model, financial_model)
financial_summary = generate_financial_summary(financial_model, growth_model, cost_model)

# Create S-curve previews
enterprise_preview = create_s_curve_preview(revenue_config, 'Enterprise')
midmarket_preview = create_s_curve_preview(revenue_config, 'Mid-Market')
smb_preview = create_s_curve_preview(revenue_config, 'SMB')

# Create the Gradio interface
with gr.Blocks(title="2025 Financial Model Dashboard", theme=gr.themes.Base()) as app:
    # Store configurations in state
    revenue_config_state = gr.State(revenue_config)
    cost_config_state = gr.State(cost_config)
    
    gr.Markdown("# 2025 Financial Model Dashboard")
    gr.Markdown("Adjust parameters to see real-time impact on financial projections")
    
    with gr.Tabs():
        with gr.TabItem("Dashboard"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### Model Parameters")
                        initial_investment = gr.Slider(
                            label="Initial Investment ($)",
                            minimum=1000000, maximum=20000000, step=1000000,
                            value=5000000, info="Starting capital"
                        )
                        
                        reset_btn = gr.Button("Reset to Defaults", variant="secondary")
                        
                        with gr.Accordion("Import/Export", open=False):
                            gr.Markdown("### Export Configurations")
                            export_revenue_btn = gr.Button("Export Revenue Config")
                            export_cost_btn = gr.Button("Export Cost Config")
                            
                            gr.Markdown("### Import Configurations")
                            import_revenue_file = gr.File(label="Import Revenue Config")
                            import_revenue_btn = gr.Button("Upload Revenue Config")
                            
                            import_cost_file = gr.File(label="Import Cost Config")
                            import_cost_btn = gr.Button("Upload Cost Config")
                            
                            import_status = gr.Textbox(label="Import Status", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("### Financial Summary")
                        financial_summary_text = gr.Textbox(
                            label="Key Metrics",
                            value=financial_summary,
                            interactive=False,
                            lines=10
                        )
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Revenue & Growth"):
                            with gr.Row():
                                growth_curves_img = gr.Image(
                                    model_charts['growth_curves'],
                                    label="Growth Curves",
                                    show_download_button=True
                                )
                            
                            with gr.Row():
                                annual_metrics_img = gr.Image(
                                    model_charts['annual_metrics'],
                                    label="Annual Metrics",
                                    show_download_button=True
                                )
                                segment_shares_img = gr.Image(
                                    model_charts['segment_shares'],
                                    label="Customer Segment Shares",
                                    show_download_button=True
                                )
                        
                        with gr.TabItem("Costs"):
                            with gr.Row():
                                expense_breakdown_img = gr.Image(
                                    model_charts['expense_breakdown'],
                                    label="Expense Breakdown",
                                    show_download_button=True
                                )
                                headcount_growth_img = gr.Image(
                                    model_charts['headcount_growth'],
                                    label="Headcount Growth",
                                    show_download_button=True
                                )
                        
                        with gr.TabItem("Financial Metrics"):
                            with gr.Row():
                                financial_summary_img = gr.Image(
                                    model_charts['financial_summary'],
                                    label="Financial Summary",
                                    show_download_button=True
                                )
                            
                            with gr.Row():
                                break_even_img = gr.Image(
                                    model_charts['break_even'],
                                    label="Break Even Analysis",
                                    show_download_button=True
                                )
                                runway_img = gr.Image(
                                    model_charts['runway'],
                                    label="Runway and Capital",
                                    show_download_button=True
                                )
                        
                        with gr.TabItem("Unit Economics"):
                            with gr.Row():
                                unit_economics_img = gr.Image(
                                    model_charts['unit_economics'],
                                    label="Unit Economics",
                                    show_download_button=True
                                )
                        
                        with gr.TabItem("Detailed Tables"):
                            with gr.Accordion("Key Metrics", open=True):
                                key_metrics_table = gr.HTML(
                                    summary_tables['key_metrics'].to_html(classes='gradio-table'),
                                    label="Key Financial Metrics"
                                )
                            
                            with gr.Accordion("Growth Summary", open=False):
                                growth_summary_table = gr.HTML(
                                    summary_tables['growth_summary'].to_html(classes='gradio-table'),
                                    label="Growth Summary"
                                )
                            
                            with gr.Accordion("Cost Summary", open=False):
                                cost_summary_table = gr.HTML(
                                    summary_tables['cost_summary'].to_html(classes='gradio-table'),
                                    label="Cost Summary"
                                )
                            
                            with gr.Accordion("Unit Economics", open=False):
                                unit_economics_table = gr.HTML(
                                    summary_tables['unit_economics'].to_html(classes='gradio-table'),
                                    label="Unit Economics"
                                )
            
        with gr.TabItem("S-Curve Tuning"):
            gr.Markdown("### Adjust S-Curves for Customer Acquisition")
            gr.Markdown("Use the sliders to scale the S-curve parameters for each customer segment and year.")
            
            with gr.Tabs():
                with gr.TabItem("Enterprise Segment"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Enterprise Segment Scaling Factors")
                            enterprise_y1_scale = gr.Slider(label="Year 1 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            enterprise_y2_scale = gr.Slider(label="Year 2 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            enterprise_y3_scale = gr.Slider(label="Year 3 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            enterprise_y4_scale = gr.Slider(label="Year 4 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            enterprise_y5_scale = gr.Slider(label="Year 5 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            enterprise_y6_scale = gr.Slider(label="Year 6 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                        
                        with gr.Column():
                            enterprise_preview = gr.Image(
                                enterprise_preview,
                                label="Enterprise S-Curve Preview",
                                show_download_button=True
                            )
                
                with gr.TabItem("Mid-Market Segment"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Mid-Market Segment Scaling Factors")
                            midmarket_y1_scale = gr.Slider(label="Year 1 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            midmarket_y2_scale = gr.Slider(label="Year 2 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            midmarket_y3_scale = gr.Slider(label="Year 3 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            midmarket_y4_scale = gr.Slider(label="Year 4 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            midmarket_y5_scale = gr.Slider(label="Year 5 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            midmarket_y6_scale = gr.Slider(label="Year 6 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                        
                        with gr.Column():
                            midmarket_preview = gr.Image(
                                midmarket_preview,
                                label="Mid-Market S-Curve Preview",
                                show_download_button=True
                            )
                
                with gr.TabItem("SMB Segment"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### SMB Segment Scaling Factors")
                            smb_y1_scale = gr.Slider(label="Year 1 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            smb_y2_scale = gr.Slider(label="Year 2 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            smb_y3_scale = gr.Slider(label="Year 3 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            smb_y4_scale = gr.Slider(label="Year 4 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            smb_y5_scale = gr.Slider(label="Year 5 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                            smb_y6_scale = gr.Slider(label="Year 6 Scale", minimum=0.0, maximum=3.0, step=0.1, value=1.0)
                        
                        with gr.Column():
                            smb_preview = gr.Image(
                                smb_preview,
                                label="SMB S-Curve Preview",
                                show_download_button=True
                            )
    
    # Set up event handlers
    
    # Main form update
    form_inputs = [
        initial_investment,
        enterprise_y1_scale, enterprise_y2_scale, enterprise_y3_scale,
        enterprise_y4_scale, enterprise_y5_scale, enterprise_y6_scale,
        midmarket_y1_scale, midmarket_y2_scale, midmarket_y3_scale,
        midmarket_y4_scale, midmarket_y5_scale, midmarket_y6_scale,
        smb_y1_scale, smb_y2_scale, smb_y3_scale,
        smb_y4_scale, smb_y5_scale, smb_y6_scale,
        revenue_config_state, cost_config_state
    ]
    
    form_outputs = [
        revenue_config_state,
        cost_config_state,
        growth_curves_img,
        annual_metrics_img,
        segment_shares_img,
        expense_breakdown_img,
        headcount_growth_img,
        financial_summary_img,
        break_even_img,
        runway_img,
        unit_economics_img,
        key_metrics_table,
        growth_summary_table,
        cost_summary_table,
        unit_economics_table,
        financial_summary_text,
        enterprise_preview,
        midmarket_preview,
        smb_preview
    ]
    
    # Update all outputs when form is changed
    for input_elem in form_inputs:
        input_elem.change(process_form, inputs=form_inputs, outputs=form_outputs)
    
    # Reset to defaults button
    reset_outputs = form_outputs + [
        initial_investment,
        enterprise_y1_scale, enterprise_y2_scale, enterprise_y3_scale,
        enterprise_y4_scale, enterprise_y5_scale, enterprise_y6_scale,
        midmarket_y1_scale, midmarket_y2_scale, midmarket_y3_scale,
        midmarket_y4_scale, midmarket_y5_scale, midmarket_y6_scale,
        smb_y1_scale, smb_y2_scale, smb_y3_scale,
        smb_y4_scale, smb_y5_scale, smb_y6_scale
    ]
    
    reset_btn.click(reset_to_defaults, inputs=[], outputs=reset_outputs)
    
    # Export buttons
    export_revenue_btn.click(export_revenue_config, inputs=[revenue_config_state], outputs=[])
    export_cost_btn.click(export_cost_config, inputs=[cost_config_state], outputs=[])
    
    # Import buttons
    import_revenue_btn.click(
        import_revenue_config,
        inputs=[import_revenue_file, revenue_config_state, cost_config_state],
        outputs=form_outputs + [import_status]
    )
    
    import_cost_btn.click(
        import_cost_config,
        inputs=[import_cost_file, revenue_config_state, cost_config_state],
        outputs=form_outputs + [import_status]
    )

# Run the app
if __name__ == "__main__":
    app.launch(share=False, inbrowser=True, server_name="localhost")