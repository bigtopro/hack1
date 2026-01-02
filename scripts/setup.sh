#!/bin/bash

# Create virtual environment
python3 -m venv analysis_env

# Activate virtual environment
source analysis_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

echo "Virtual environment 'analysis_env' created and packages installed."
echo "To activate the environment, run: source analysis_env/bin/activate"
echo "To run the analysis engine, execute: python analysis_engine.py"