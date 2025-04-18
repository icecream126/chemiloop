#!/bin/bash

# Customize these variables
conda_env="hm_chromophore2"           # Replace with your conda environment name
property_name="logP"   # Replace with your property name
property_value=2.0 # Replace with your property value
project_name="test"

for tmux_num in {1..30}
do
    session_name="${tmux_num}_${property_name}_${property_value}"
    
    tmux kill-session -t "$session_name"
    
    echo "Started tmux session: $session_name"
done
