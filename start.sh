#!/bin/bash

# Step 1: Create a virtual environment and activate it
python3 -m venv .venv
source .venv/bin/activate

# Step 2: Upgrade pip and install the required packages
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit --upgrade

# Step 3: Run the Streamlit app
streamlit run Ana-Paula.py
