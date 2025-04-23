# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Run the full model: `python app.py`
- Run interactive dashboard: `streamlit run streamlit_app.py`
- Install requirements: `pip install -r requirements.txt`

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