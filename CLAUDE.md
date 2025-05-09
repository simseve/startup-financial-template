# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run the full model: `python app.py`
- Run interactive Gradio dashboard: `python gradio_app.py` 
- Install requirements: `pip install -r requirements.txt`

## Running the Gradio Dashboard
The Gradio dashboard allows you to interactively modify S-curves and other parameters:

1. Install requirements: `pip install -r requirements.txt`
2. Run the dashboard: `python gradio_app.py`
3. Open your browser to the URL shown in the terminal (usually http://localhost:7860)
4. Use the S-Curve Tuning tab to adjust parameters and see real-time updates

## Running the Streamlit Dashboard (Legacy)
The Streamlit dashboard is no longer available in the current version (streamlit_app.py was removed):

1. Install requirements: `pip install -r requirements.txt`
2. Run the dashboard: `streamlit run streamlit_app.py` (if available)
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
- Interactive analysis via `gradio_app.py`

## Optimization Strategies
- **Baseline**: Standard S-curve growth profiles
