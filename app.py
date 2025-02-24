import streamlit as st
import pandas as pd
from ev_model import PopulationSimulator
from helper import (
    convert_time,
)  # Import the same convert_time function you use in the notebook

# Set page config for wider layout
st.set_page_config(layout="wide")

# Set page title
st.title("EV Population Charging Patterns by Archetype")


# Load archetype data
@st.cache_data
def load_archetype_data():
    archetypes_df = pd.read_csv("EV_Archetypes.csv")

    # Convert time columns to int - handle both HH:MM and H formats
    archetypes_df["Plug-in time"] = archetypes_df["Plug-in time"].apply(convert_time)
    archetypes_df["Plug-out time"] = archetypes_df["Plug-out time"].apply(convert_time)

    # Convert SoC columns to float - stripping %
    archetypes_df["Target SoC"] = (
        archetypes_df["Target SoC"].str.rstrip("%").astype(float)
    )
    archetypes_df["Plug-in SoC"] = (
        archetypes_df["Plug-in SoC"].str.rstrip("%").astype(float)
    )
    return archetypes_df


# Sidebar controls
st.sidebar.header("Simulation Parameters")
population_size = st.sidebar.slider("Population Size", 100, 10000, 1000, 100)
percentile = st.sidebar.slider("Percentile Range", 50, 95, 95, 5)

# Load data and run simulation
archetypes_df = load_archetype_data()
simulator = PopulationSimulator(archetypes_df, population_size)
results = simulator.run_simulation(hours=24)


# Display plot
st.plotly_chart(
    simulator.plot_population_results(percentile=percentile), use_container_width=True
)


# Add explanation
st.markdown(
    """
### About this visualization
This plot shows the EV archetypes and their charging patterns:
- Adjust the population size and percentile range using the sidebar controls
- The solid red line represents the mean state of charge (SoC) across all vehicles
- The dashed lines show the confidence intervals
- Use the dropdown menu above the plot to view different archetype patterns
"""
)
