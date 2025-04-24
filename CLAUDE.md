# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run the full model: `python app.py`
- Run interactive dashboard: `streamlit run streamlit_app.py`
- Install requirements: `pip install -r requirements.txt`

## Running the Streamlit Dashboard
The Streamlit dashboard allows you to interactively modify S-curves and other parameters:

1. Install requirements: `pip install -r requirements.txt`
2. Run the dashboard: `streamlit run streamlit_app.py`
3. Open your browser to the URL shown in the terminal (usually http://localhost:8501)
4. Use the sidebar to modify parameters and see real-time updates

## Code Style Guidelines
- Indentation: 4 spaces
- Line length: 100 characters max
- Imports: Group by standard library, third-party, local imports
- Naming: Classes use CamelCase, functions/methods use snake_case
- Type hints: Not used in this codebase
- Docstrings: Numpy-style docstrings with Parameters/Returns sections
- Error handling: Use specific exceptions with informative messages
- Comments: Descriptive comments for complex operations

## Project Structure
- Models are defined in the `models/` directory
- Financial outputs are saved to `reports/` directory
- Run models with default parameters via `app.py`
- Interactive analysis via `streamlit_app.py`

## Optimization Strategies
- **Baseline, Conservative, Aggressive, Hypergrowth**: Standard S-curve growth profiles
- **Breakeven**: Finds the growth multiplier needed to reach breakeven by a target month
  - `python app.py --strategy breakeven --breakeven-target 24`
- **Series B**: Optimizes for $10M ARR with 100%+ growth by a target month
  - `python app.py --strategy series_b --series-b-target 36`
- **Revenue Target**: Calculates customers needed to reach a specific monthly revenue target
  - `python app.py --strategy revenue_target --revenue-target 5000000 --revenue-target-month 36`
- **Annual Revenue Target**: Calculates customers needed to reach a specific annual revenue target
  - `python app.py --strategy annual_revenue --annual-revenue-target 360000000 --annual-revenue-year 5`
  - Uses more sophisticated segment-specific growth with year-by-year tapering
  - Ideal for precise control targeting specific annual revenue goals
- **Enterprise First**: Strategy focusing on enterprise and mid-market in early years
- **Regulatory Impact**: Strategy accounting for AI regulation impact on segment adoption