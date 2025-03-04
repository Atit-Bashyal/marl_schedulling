import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
import tensorflow as tf
from scipy.interpolate import CubicSpline

def plot_stacked_bar(data, price_signal, pv_signal=None):
    """
    Plots a stacked bar chart of energy demand, overlays a price signal, 
    and optionally includes a PV signal with a third y-axis.

    Args:
        data (dict): A dictionary where keys are timesteps (int) and values are
                     dictionaries with equipment names (str) as keys and energy
                     demand (float/int) as values.
        price_signal (list): A list of price signal values corresponding to each timestep.
        pv_signal (list, optional): A list of PV signal values corresponding to each timestep. Must match timesteps.
    """
    # Extract timesteps and equipment names
    timesteps = list(data.keys())
    equip = list(next(iter(data.values())).keys())
    equipment_names = [i for i in equip if i != 'Reductionfurnace']  # Get equipment names from the first timestep

    # Validate price_signal length matches timesteps
    if len(price_signal) != len(timesteps):
        raise ValueError("Length of price_signal must match the number of timesteps in data.")
    
    # Validate PV signal if provided
    if pv_signal is not None and len(pv_signal) != len(timesteps):
        raise ValueError("Length of pv_signal must match the number of timesteps in data.")

    # Prepare data for stacking
    energy_values = {equip: [] for equip in equipment_names}
    for timestep in timesteps:
        for equip in equipment_names:
            energy_values[equip].append(data[timestep].get(equip, 0))
    
    # Plot stacked bar chart
    x = np.array(timesteps)
    bottom = np.zeros(len(x))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    bars = []
    for equip in equipment_names:
        bar = ax1.bar(x, energy_values[equip], bottom=bottom, label=equip)
        bars.append(bar)
        bottom += np.array(energy_values[equip])
    
    # Customize bar chart (primary y-axis)
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Energy demand (kWh)")
    ax1.set_title("Energy demand by equipment")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot price signal on secondary y-axis
    ax2 = ax1.twinx()  # Create a twin y-axis
    price_line, = ax2.plot(x, price_signal, color="red", marker="o", label="Electricity price(€/kWH)", linestyle="--", linewidth=2)
    ax2.set_ylabel("Price Signal")
    ax2.tick_params(axis="y")
    
    # Add a third y-axis for PV signal, if provided
    if pv_signal is not None:
        ax3 = ax1.twinx()  # Create another twin y-axis
        ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.set_ticks_position("right")
        pv_line, = ax3.plot(x, pv_signal, color="blue", marker="x", label="PV Signal", linestyle=":", linewidth=2)
        ax3.set_ylabel("PV Signal (kWh)", color="blue")
        ax3.tick_params(axis="y", labelcolor="blue")
    else:
        pv_line = None

    # Combine legends and place them at the bottom
    handles1, labels1 = ax1.get_legend_handles_labels()  # Bars
    handles2, labels2 = [price_line], ["Price Signal"]   # Line for price signal
    if pv_signal is not None:
        handles2.append(pv_line)
        labels2.append("PV Signal")  # Line for PV signal

    handles = handles1 + handles2
    labels = labels1 + labels2
    fig.legend(handles, labels, loc="lower center", ncol=6)
    
    # Adjust layout to fit the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Leave space for the legend
    plt.show()


def plot_stacked_bar_comparison(data1, data2, data3, price_signal, pv_signal=None):
    """
    Plots stacked bar charts for three datasets in the same plot, overlays a price signal,
    and optionally includes a PV signal with a third y-axis.

    Args:
        data1, data2, data3 (dict): Dictionaries where keys are timesteps (int) and values are
                                     dictionaries with equipment names (str) as keys and energy
                                     demand (float/int) as values.
        price_signal (list): A list of price signal values corresponding to each timestep.
        pv_signal (list, optional): A list of PV signal values corresponding to each timestep. Must match timesteps.
    """
    datasets = [data1, data2, data3]
    labels = ["Scenario 1", "Scenario 2", "Scenario 3"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
    
    # Extract timesteps and equipment names
    timesteps = list(data1.keys())
    equip = list(next(iter(data1.values())).keys())
    equipment_names = [i for i in equip if i != 'Reductionfurnace']  # Get equipment names from the first timestep

    # Validate price_signal length matches timesteps
    if len(price_signal) != len(timesteps):
        raise ValueError("Length of price_signal must match the number of timesteps in data.")
    
    # Validate PV signal if provided
    if pv_signal is not None and len(pv_signal) != len(timesteps):
        raise ValueError("Length of pv_signal must match the number of timesteps in data.")

    # Prepare data for stacking
    energy_values = []
    for data in datasets:
        temp_values = {equip: [] for equip in equipment_names}
        for timestep in timesteps:
            for equip in equipment_names:
                temp_values[equip].append(data[timestep].get(equip, 0))
        energy_values.append(temp_values)
    
    x = np.array(timesteps)
    width = 0.25  # Bar width for side-by-side comparison
    offsets = [-width, 0, width]  # Offsets for each dataset
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    for idx, (data_values, label, color, offset) in enumerate(zip(energy_values, labels, colors, offsets)):
        bottom = np.zeros(len(x))
        for equip in equipment_names:
            ax1.bar(x + offset, data_values[equip], width=width, bottom=bottom, label=f"{label} - {equip}", alpha=0.7, edgecolor='black')
            bottom += np.array(data_values[equip])
    
    # Customize bar chart (primary y-axis)
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Energy Demand (kWh)")
    ax1.set_title("Energy Demand Comparison Across Scenarios")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Plot price signal on secondary y-axis
    ax2 = ax1.twinx()
    price_line, = ax2.plot(x, price_signal, color="red", marker="o", label="Price Signal", linestyle="--", linewidth=2)
    ax2.set_ylabel("Price Signal")
    ax2.tick_params(axis="y", labelcolor="red")
    
    # Add a third y-axis for PV signal, if provided
    if pv_signal is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.set_ticks_position("right")
        pv_line, = ax3.plot(x, pv_signal, color="blue", marker="x", label="PV Signal", linestyle=":", linewidth=2)
        ax3.set_ylabel("PV Signal (kWh)", color="blue")
        ax3.tick_params(axis="y", labelcolor="blue")
    else:
        pv_line = None
    
    # Combine legends and place them at the bottom
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = [price_line], ["Price Signal"]
    if pv_signal is not None:
        handles2.append(pv_line)
        labels2.append("PV Signal")
    
    handles = handles1 + handles2
    labels = labels1 + labels2
    fig.legend(handles, labels, loc="lower center", ncol=6)
    
    # Adjust layout to fit the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()

def plot_storage_charging_discharging(energy_storage, price_signal, pv_signal=None):
    """
    Plots charging and discharging activities of storage devices alongside price and PV signals.

    Args:
        energy_storage (dict): A dictionary where keys are timesteps (int) and values are
                               dictionaries with storage device names (str) as keys, each containing
                               {'energy_consumed': float, 'energy_discharged': float, 'energy_stored': float, 'status': str}.
        price_signal (list): A list of price signal values corresponding to each timestep.
        pv_signal (list, optional): A list of PV signal values corresponding to each timestep. Must match timesteps.
    """
    # Extract timesteps and storage device names
    timesteps = list(energy_storage.keys())
    storage_devices = list(next(iter(energy_storage.values())).keys())
    
    # Validate price_signal length matches timesteps
    if len(price_signal) != len(timesteps):
        raise ValueError("Length of price_signal must match the number of timesteps in energy_storage.")
    
    # Validate PV signal if provided
    if pv_signal is not None and len(pv_signal) != len(timesteps):
        if len(pv_signal) != len(timesteps):
            raise ValueError("Length of pv_signal must match the number of timesteps in energy_storage.")
    
    # Extract charging and discharging data for each storage device
    charging_data = {device: [] for device in storage_devices}
    discharging_data = {device: [] for device in storage_devices}
    
    for timestep in timesteps:
        for device in storage_devices:
            charging_data[device].append(energy_storage[timestep][device].get('energy_consumed', 0))
            discharging_data[device].append(-energy_storage[timestep][device].get('energy_discharged', 0))  # Negative for discharge
    
    # Plot charging and discharging data
    x = np.array(timesteps)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    for device in storage_devices:
        # Plot charging (positive)
        ax1.bar(x, charging_data[device], label=f"{device} (Charging)", alpha=0.7)
        # Plot discharging (negative)
        ax1.bar(x, discharging_data[device], label=f"{device} (Discharging)", alpha=0.7)
    
    # Customize bar chart (primary y-axis)
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Energy (kWh)")
    ax1.set_title("Storage device charging and discharging")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot price signal on secondary y-axis
    ax2 = ax1.twinx()  # Create a twin y-axis
    price_line, = ax2.plot(x, price_signal, color="red", marker="o", label="Electricity price(€/kWH)", linestyle="--", linewidth=2)
    ax2.set_ylabel("Price signal")
    ax2.tick_params(axis="y", labelcolor="red")
    
    # Add a third y-axis for PV signal, if provided
    if pv_signal is not None:
        ax3 = ax1.twinx()  # Create another twin y-axis
        ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.set_ticks_position("right")
        pv_line, = ax3.plot(x, pv_signal, color="blue", marker="x", label="PV Signal", linestyle=":", linewidth=2)
        ax3.set_ylabel("PV Signal (kWh)", color="blue")
        ax3.tick_params(axis="y", labelcolor="blue")
    else:
        pv_line = None

    # Combine legends and place them at the bottom
    handles1, labels1 = ax1.get_legend_handles_labels()  # Bars
    handles2, labels2 = [price_line], ["Price Signal"]   # Line for price signal
    if pv_signal is not None:
        handles2.append(pv_line)
        labels2.append("PV Signal")  # Line for PV signal

    handles = handles1 + handles2
    labels = labels1 + labels2
    fig.legend(handles, labels, loc="lower center", ncol=6)
    
    # Adjust layout to fit the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Leave space for the legend
    plt.show()


def plot_material_data(data, steelpowder_key='Steelpowder', water_key='Coolwater', nitrogen_key='Nitrogen'):
    """
    Plots material values over time along with their min and max capacity.
    
    Parameters:
        data (dict): Dictionary containing material data over time.
                     Format: {timestep: {material: {value, min_cap, max_cap}, ...}, ...}
        steelpowder_key (str): Key for steel powder in the dataset. Defaults to 'Steelpowder'.
        water_key (str): Key for water in the dataset. Defaults to 'Coolwater'.
        nitrogen_key (str): Key for nitrogen in the dataset. Defaults to 'Nitrogen'.
    """
    # Prepare data for Steelpowder
    timesteps = list(data.keys())
    steel_values = [data[t][steelpowder_key]['value'] for t in timesteps]
    steel_min = [data[t][steelpowder_key]['min_cap'] for t in timesteps]
    steel_max = [data[t][steelpowder_key]['max_cap'] for t in timesteps]

    # Plot Steelpowder
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, steel_values, label='Steelpowder Value', marker='o')
    plt.axhline(y=steel_min[0], color='red', linestyle='--', label='Steelpowder Min Cap')
    plt.axhline(y=steel_max[0], color='green', linestyle='--', label='Steelpowder Max Cap')
    plt.title("Steelpowder Values Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Steelpowder")
    plt.legend()
    plt.grid()
    plt.show()

    # Prepare data for Coolwater and Nitrogen
    water_values = [data[t][water_key]['value'] for t in timesteps]
    water_min = [data[t][water_key]['min_cap'] for t in timesteps]
    water_max = [data[t][water_key]['max_cap'] for t in timesteps]

    nitrogen_values = [data[t][nitrogen_key]['value'] for t in timesteps]
    nitrogen_min = [data[t][nitrogen_key]['min_cap'] for t in timesteps]
    nitrogen_max = [data[t][nitrogen_key]['max_cap'] for t in timesteps]

    # Plot Coolwater and Nitrogen
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, water_values, label='Coolwater Value', marker='o', color='blue')
    plt.axhline(y=water_min[0], color='cyan', linestyle='--', label='Coolwater Min Cap')
    plt.axhline(y=water_max[0], color='navy', linestyle='--', label='Coolwater Max Cap')

    plt.plot(timesteps, nitrogen_values, label='Nitrogen Value', marker='s', color='orange')
    plt.axhline(y=nitrogen_min[0], color='yellow', linestyle='--', label='Nitrogen Min Cap')
    plt.axhline(y=nitrogen_max[0], color='brown', linestyle='--', label='Nitrogen Max Cap')

    plt.title("Coolwater and Nitrogen Values Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()


def plot_reward_distribution_individual(individual_rewards):
    """
    Plot the reward distribution for each equipment as subplots.
    
    Parameters:
    - individual_rewards: A dictionary where each key is a timestep and 
      the value is a list of tuples (equipment_name, reward).
    """
    # Extract unique equipment names
    equipment_names = sorted({equipment for rewards in individual_rewards.values() for equipment, _ in rewards})
    
    # Organize rewards for each equipment
    equipment_rewards = {equipment: [] for equipment in equipment_names}
    for rewards in individual_rewards.values():
        for equipment, reward in rewards:
            equipment_rewards[equipment].append(reward)
    
    # Create subplots
    num_equipment = len(equipment_names)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_equipment + cols - 1) // cols  # Calculate rows based on the number of equipment

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot the reward distribution for each equipment
    for i, equipment in enumerate(equipment_names):
        sns.kdeplot(equipment_rewards[equipment], fill=True, alpha=0.5, ax=axes[i])
        axes[i].set_title(f"Reward Distribution: {equipment}", fontsize=12)
        axes[i].set_xlabel("Reward", fontsize=10)
        axes[i].set_ylabel("Density", fontsize=10)
        axes[i].grid(alpha=0.3)

    # Hide unused subplots if there are extra axes
    for j in range(len(equipment_names), len(axes)):
        axes[j].axis('off')

    # Adjust spacing between subplots
    plt.tight_layout()
    plt.show()

def plot_reward_distribution(individual_rewards, agents_):
    """
    Plot the distribution of rewards for each equipment across the episode.
    
    Parameters:
    - individual_rewards: A dictionary with timestep as the key and list of (agent, reward) as values.
    - agents_: List of agent names.
    """
    # Collect rewards for each agent
    rewards_by_agent = {agent: [] for agent in agents_}
    
    for timestep, rewards in individual_rewards.items():
        for agent, reward in rewards:
            rewards_by_agent[agent].append(reward)

    # Plot the distribution for each agent
    plt.figure(figsize=(12, 8))
    for agent, rewards in rewards_by_agent.items():
        sns.kdeplot(rewards, label=agent, fill=True, alpha=0.5)
    
    plt.title("Reward Distribution for Each Equipment", fontsize=16)
    plt.xlabel("Reward", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(title="Agent", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

def plot_cumulative_rewards(cumulative_rewards):

    """
    Plot the cumulative rewards as a distribution using a KDE plot.
    
    Parameters:
    - cumulative_rewards: A dictionary with timestep as the key and cumulative reward as the value.
    """
    # Extract the cumulative rewards into a list
    cumulative_rewards_values = list(cumulative_rewards.values())
    
    # Create the KDE plot for cumulative rewards
    plt.figure(figsize=(10, 6))
    sns.kdeplot(cumulative_rewards_values, fill=True, alpha=0.5, color='blue')
    
    # Add labels and title
    plt.title("Distribution of Cumulative Rewards", fontsize=16)
    plt.xlabel("Cumulative Reward", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(alpha=0.3)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_container_value_over_time(data, container_name):
    time_steps = sorted(data.keys())
    values = [data[time][container_name]['value'] for time in time_steps]

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, values, label="Value", marker="o")
    plt.title(f"{container_name} Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


def plot_container_comparison_over_time(container_data, container_1, container_2):
    """
    Plots a comparison of the states of two containers over all time steps with dual y-axes.

    Args:
    container_data: Dict mapping container names to lists of states (list of size 24).
    container_1: Name of the first container to compare.
    container_2: Name of the second container to compare.
    """

    
    time_steps = sorted(container_data.keys())
    data_1 = [container_data[time][container_1]['value'] for time in time_steps]
    data_2 = [container_data[time][container_2]['value'] for time in time_steps]
    fig, ax1 = plt.subplots()

    ax1.plot(time_steps, data_1, 'b-', label=container_1)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel(f'{container_1} Values', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(time_steps, data_2, 'r-', label=container_2)
    ax2.set_ylabel(f'{container_2} Values', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    fig.tight_layout()
    plt.title(f'Comparison of {container_1} and {container_2} Over Time')
    plt.show()


def plot_smooth_heatmap_over_time(data):
    """
    Plots a smoother heatmap of normalized container values over time with a red color gradient.

    Args:
    data: Dict mapping time steps to container data.
          Each time step contains a dictionary with container names as keys, and each container
          has a 'value' and a 'max_cap'.
          Example: {time_step: {container_name: {'value': 100, 'max_cap': 200}}}
    """
    time_steps = sorted(data.keys())
    containers = [c for c in data[time_steps[0]].keys() if c != 'Battery']
    
    # Normalize the data by maximum capacity
    heatmap_data = {
        container: [
            data[time][container]['value'] / data[time][container]['max_cap'] 
            for time in time_steps
        ]
        for container in containers
    }
    
    # Create a DataFrame for the original data
    df = pd.DataFrame(heatmap_data, index=time_steps)
    
    # Smooth the data by interpolating
    zoom_factor = 1  # Adjust for smoothness
    smoothed_data = zoom(df.values, zoom=(zoom_factor, 1), order=3)  # Interpolate time steps only
    
    # Create a DataFrame for the smoothed data
    smoothed_df = pd.DataFrame(
        smoothed_data,
        index=np.linspace(min(time_steps), max(time_steps), smoothed_data.shape[0]),
        columns=containers
    )
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        smoothed_df,
        cmap="viridis_r",
        annot=False,
        cbar_kws={'label': 'Normalized Capacity Value'}
    )
    plt.title("Smoothed Heatmap of Container Values Over Time")
    plt.xticks(rotation=60,fontsize=10)
    plt.yticks(fontsize=10) 
    plt.show()


def plot_3d_heatmap_over_time(data):
    """
    Plots a 3D heatmap of container values over time using normalized values (value/max_cap).
    
    Args:
        data: Dictionary containing container data with normalized values.
    """
    time_steps = sorted(data.keys())
    containers = [c for c in data[time_steps[0]].keys() if c not in ['Battery','Coolwater','Nitrogen']]
    heatmap_data = {
        container: [data[time][container]['value'] / data[time][container]['max_cap'] for time in time_steps]
        for container in containers
    }
    
    # Create DataFrame
    df = pd.DataFrame(heatmap_data, index=time_steps)
    
    # Get the x, y, and z coordinates for 3D plotting
    x = range(len(time_steps))  # Time steps
    y = range(len(containers))  # Containers
    X, Y = np.meshgrid(x, y)  # Create grid
    Z = np.array([df.iloc[:, i].values for i in range(len(containers))])  # Values for height
    
    # Create the 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Adjust the bar width (dx, dy)
    dx = np.ones_like(X) * 0.8  # Wider bars along the x-axis
    dy = np.ones_like(Y) * 0.8  # Wider bars along the y-axis
    
    # Add bars to represent the data in 3D
    for i in range(len(y)):  # Iterate through containers
        ax.bar3d(X[i], Y[i], np.zeros_like(Z[i]), dx=dx[i], dy=dy[i], dz=Z[i], alpha=0.8, color=plt.cm.Reds(Z[i] / Z.max()))
    
    # Set labels and title
    ax.set_title("3D heatmap of container storage over time", fontsize=16)
    ax.set_xlabel(" ", fontsize=12)
    ax.set_ylabel(" ", fontsize=12)
    ax.set_zlabel("Normalized capacity value", fontsize=12)
    ax.set_yticks(range(len(containers)))
    ax.set_yticklabels(containers, rotation=90, fontsize=10)  # Rotate y-axis container labels
    ax.set_xticks(range(len(time_steps)))
    ax.set_xticklabels(time_steps, rotation=90, fontsize=10)  # Rotate x-axis time step labels

     # Adjust axis limits to make the space bigger
    ax.set_xlim([min(x) - 1, max(x) + 1])  # Increase the x-axis range
    ax.set_ylim([min(y) - 1, max(y) + 1])  # Increase the y-axis range
    ax.set_zlim([0, Z.max() + 0.1])  # Increase the z-axis range for better visibility


    plt.show()

def plot_2d_heatmap_over_time(data):
    """
    Plots a 2D heatmap of container values over time using normalized values (value/max_cap).
    
    Args:
        data: Dictionary containing container data with normalized values.
    """
    time_steps = sorted(data.keys())
    containers = [c for c in data[time_steps[0]].keys() if c not in ['Battery', 'Coolwater', 'Nitrogen']]
    
    # Create a heatmap data dictionary
    heatmap_data = {
        container: [data[time][container]['value'] / data[time][container]['max_cap'] for time in time_steps]
        for container in containers
    }
    
    # Convert dictionary to a DataFrame
    df = pd.DataFrame(heatmap_data, index=time_steps)

    # Plot heatmap
    plt.figure(figsize=(14, 8))  # Adjust figure size for larger "pixels"
    ax = sns.heatmap(df.T, cmap="Reds", annot=False, fmt=".2f", linewidths=1, linecolor="black", square=True, cbar_kws={'label': 'Normalized Capacity'})

    # Set labels and title
    plt.xlabel("Time steps", fontsize=14)
    plt.ylabel("Containers", fontsize=14)
    plt.xticks(rotation=90, fontsize=12)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=30,fontsize=12)

    plt.show()

def compute_and_plot_energy_cost(consumption_dict, price_list):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(consumption_dict, orient='index')
    
    # Ensure the price_list length matches the number of timesteps
    if len(price_list) != len(df):
        raise ValueError("The length of the price list must match the number of timesteps in the consumption dictionary.")
    
    # Calculate the total consumption per hour (sum of all equipment values)
    df['Total_Consumption'] = df.sum(axis=1)
    
    # Multiply the total consumption by the price signal for each hour [price/1000 for price in price_list]
    df['Energy_Cost'] = df['Total_Consumption'] * [price/1000 for price in price_list]
    
    # Create a figure and axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot energy cost as a bar plot
    ax1.bar(df.index, df['Energy_Cost'], color='b', alpha=0.7, label='Energy Cost')
    ax1.set_xlabel("Time (Hours)")
    ax1.set_ylabel("Energy Cost", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title("Hourly Energy Cost and Price Signal")
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add a secondary y-axis for price signal
    ax2 = ax1.twinx()
    ax2.plot(df.index, price_list, color='r', marker='o', linestyle='-', label='Price Signal')
    ax2.set_ylabel("Price Signal", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legends
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def compare_total_energy_cost(consumption_dict1, consumption_dict2, price_signal):
    # Convert dictionaries to DataFrames
    df1 = pd.DataFrame.from_dict(consumption_dict1, orient='index')
    df2 = pd.DataFrame.from_dict(consumption_dict2, orient='index')
    
    # Ensure the price_signal length matches the number of timesteps
    if len(price_signal) != len(df1) or len(price_signal) != len(df2):
        raise ValueError("The length of the price signal must match the number of timesteps in the consumption dictionaries.")
    
    # Compute total consumption and energy cost for each facility
    df1['Total_Consumption'] = df1.sum(axis=1)
    df1['Energy_Cost'] = df1['Total_Consumption'] * [price/1000 for price in price_list]
    
    df2['Total_Consumption'] = df2.sum(axis=1)
    df2['Energy_Cost'] = df2['Total_Consumption'] * [price/1000 for price in price_list]
    
    # Compute total energy cost over 24 hours for each facility
    total_cost1 = df1['Energy_Cost'].sum()
    total_cost2 = df2['Energy_Cost'].sum()
    
    # Print results
    print(f"Total Energy Cost for Facility 1: ${total_cost1:.2f}")
    print(f"Total Energy Cost for Facility 2: ${total_cost2:.2f}")
    
    # Compare and plot
    facilities = ['Facility 1', 'Facility 2']
    total_costs = [total_cost1, total_cost2]
    
    plt.figure(figsize=(8, 5))
    plt.bar(facilities, total_costs, color=['blue', 'green'], alpha=0.7)
    plt.ylabel("Total Energy Cost ($)")
    plt.title("Comparison of Total Energy Costs")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.show()
    
    return total_cost1, total_cost2

def compare_hourly_energy_costs(consumption_dict1, consumption_dict2, price_signal):
    # Convert dictionaries to DataFrames
    df1 = pd.DataFrame.from_dict(consumption_dict1, orient='index')
    df2 = pd.DataFrame.from_dict(consumption_dict2, orient='index')
    
    # Ensure the price_signal length matches the number of timesteps
    if len(price_signal) != len(df1) or len(price_signal) != len(df2):
        raise ValueError("The length of the price signal must match the number of timesteps in the consumption dictionaries.")
    
    # Compute total consumption and energy cost for each facility at each hour
    df1['Total_Consumption'] = df1.sum(axis=1)
    df1['Energy_Cost'] = df1['Total_Consumption'] * [price/1000 for price in price_list]
    
    df2['Total_Consumption'] = df2.sum(axis=1)
    df2['Energy_Cost'] = df2['Total_Consumption'] * [price/1000 for price in price_list]
    
    # Combine the energy costs into a single DataFrame
    comparison_df = pd.DataFrame({
        'Hour': df1.index,
        'Facility_1_Cost': df1['Energy_Cost'],
        'Facility_2_Cost': df2['Energy_Cost']
    })
    
    # Plot the hourly energy costs
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plots for Facility 1 and Facility 2
    ax1.bar(comparison_df['Hour'] - 0.2, comparison_df['Facility_1_Cost'], width=0.4, label='Facility 1', alpha=0.7, color='blue')
    ax1.bar(comparison_df['Hour'] + 0.2, comparison_df['Facility_2_Cost'], width=0.4, label='Facility 2', alpha=0.7, color='green')
    
    # Formatting
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Energy Cost ($)')
    ax1.set_title('Hourly Energy Cost Comparison')
    ax1.set_xticks(comparison_df['Hour'])
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # Show the plot
    plt.show()
    
    return comparison_df

def extract_histogram_from_tensorboard(log_dir, tag):
    """
    Extracts histogram data for a specific tag from TensorBoard logs.
    
    Args:
        log_dir (str): Path to the TensorBoard log directory.
        tag (str): The tag to extract the histogram data (e.g., "ray/tune/env_runners/hist_stats/episode_reward").
    
    Returns:
        (list of bins, list of frequencies): The histogram data (bin edges and frequencies).
    """
    # Create a summary iterator to read through the event files in the log directory
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events.out.tfevents' in f]
    
    histograms = []

    for event_file in event_files:
        # Read events from the file
        for summary in tf.compat.v1.train.summary_iterator(event_file):
            for value in summary.summary.value:
                if value.tag == tag:
                    # Extract histogram data (bin edges and values)
                    print(value) 
    #                 histograms.append(value.histo)
    
    # if not histograms:
    #     raise ValueError(f"No histogram found for tag: {tag}")
    
    # # Assuming that all histograms are the same, we just take the first one
    # histogram = histograms[0]

    # # Extract bin edges and frequencies from the histogram
    # bin_edges = histogram.min_edge
    # bin_values = histogram.bucket

    # return bin_edges, bin_values

def plot_histogram(bin_edges, bin_values):
    """
    Plots a histogram from the given bin edges and frequencies.
    
    Args:
        bin_edges (list): The edges of the histogram bins.
        bin_values (list): The frequencies of the histogram bins.
    """
    # Convert bin edges to a format that matplotlib understands for plotting
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, bin_values, width=(bin_edges[1] - bin_edges[0]), alpha=0.7, color='blue')
    plt.title("Histogram of Episode Rewards")
    plt.xlabel("Episode Reward")
    plt.ylabel("Frequency")
    plt.show()

def plot_total_energy_cost(consumption_dict, price_signal):
    """
    Computes and plots the total energy cost for a 24-hour period for a single facility.

    Parameters:
    - consumption_dict: dict, energy consumption values for each equipment per timestep.
    - price_signal: list, hourly price signal for energy.

    Returns:
    - total_cost: float, total energy cost over 24 hours.
    """
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(consumption_dict, orient='index')
    
    # Ensure the price_signal length matches the number of timesteps
    if len(price_signal) != len(df):
        raise ValueError("The length of the price signal must match the number of timesteps in the consumption dictionary.")
    
    # Compute total consumption and energy cost for each hour
    df['Total_Consumption'] = df.sum(axis=1)
    df['Energy_Cost'] = df['Total_Consumption'] * [price/1000 for price in price_signal]
    
    # Compute the total energy cost for the 24-hour period
    total_cost = df['Energy_Cost'].sum()
    
    # Plot the hourly energy cost
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for energy costs
    ax1.bar([1], [total_cost], color='blue', alpha=0.7, label='Total Energy Cost')
    
    # Formatting the plot
    ax1.set_xlabel('')
    ax1.set_ylabel('Energy Cost ($)')
    ax1.set_title('Total Energy Cost')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Show the plot
    plt.show()


def plot_value_vs_step(csv_file1, csv_file2, csv_file3):
    """
    Plots 'Value' against 'Step' from given CSV files with vertical dotted lines at the end of each dataset.
    
    Parameters:
        csv_file1 (str): Path to the first CSV file containing the data.
        csv_file2 (str): Path to the second CSV file containing the data.
        csv_file3 (str): Path to the third CSV file containing the data.
    """
    # Read the CSV files into pandas DataFrames
    data1 = pd.read_csv(csv_file1)
    data2 = pd.read_csv(csv_file2)
    data3 = pd.read_csv(csv_file3)
    
    # Extract the 'Step' and 'Value' columns
    steps1 = [x/240 for x in data1['Step']]
    steps2 = [x/240 for x in data2['Step']]
    steps3 = [x/240 for x in data3['Step']]
    values1 = data1['Value'] * 10
    values2 = data2['Value']
    values3 = data3['Value']

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps1, values1, linestyle='-', color='r', label='Episode Mean PPO')
    plt.plot(steps2, values2, linestyle='-', color='b', label='Episode Mean MAPPO')
    plt.plot(steps3, values3, linestyle='-', color='g', label='Episode Mean Proposed algorithm')

    # Add vertical dotted lines at the end of each line
    plt.axvline(x=steps1[-1], linestyle=':', color='r', linewidth=1.5)  # Vertical line for dataset 1 (PPO)
    plt.axvline(x=steps2[-1], linestyle=':', color='b', linewidth=1.5)  # Vertical line for dataset 2 (MAPPO)
    plt.axvline(x=steps3[-1], linestyle=':', color='g', linewidth=1.5)  # Vertical line for dataset 3 (Proposed Algorithm)

    # Add labels and title
    plt.xlabel('Training episode')
    plt.ylabel('Episode reward (mean)')
    plt.title('Comparison of convergence of different training algorithms')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def plot_value_vs_step_smoothed(csv_file1, csv_file2, csv_file3, smoothing_window=5):
    """
    Plots 'Value' against 'Step' from given CSV files with optional smoothing.
    
    Parameters:
        csv_file1 (str): Path to the first CSV file containing the data.
        csv_file2 (str): Path to the second CSV file containing the data.
        csv_file3 (str): Path to the third CSV file containing the data.
        smoothing_window (int): Size of the moving average window for smoothing (default is 5).
    """
    # Read the CSV files into pandas DataFrames
    data1 = pd.read_csv(csv_file1)
    data2 = pd.read_csv(csv_file2)
    data3 = pd.read_csv(csv_file3)
    
    # Extract the 'Step' and 'Value' columns
    steps1 = [x/240 for x in data1['Step']]
    steps2 = [x/240 for x in data2['Step']]
    steps3 = [x/240 for x in data3['Step']]
    values1 = data1['Value'] * 2
    values2 = data2['Value']
    values3 = data3['Value']

    # Apply moving average for smoothing (optional)
    values1_smoothed = pd.Series(values1).rolling(window=smoothing_window).max()
    values2_smoothed = pd.Series(values2).rolling(window=smoothing_window).max()
    values3_smoothed = pd.Series(values3).rolling(window=smoothing_window).max()

    # Alternatively, you can use spline interpolation (comment out the moving average lines above)
    # Interpolate using cubic spline for smoother curves
    cs1 = CubicSpline(steps1, values1)
    cs2 = CubicSpline(steps2, values2)
    cs3 = CubicSpline(steps3, values3)
    
    # Create a finer set of steps for interpolation
    finer_steps = np.linspace(min(steps1+steps2+steps3), max(steps1+steps2+steps3), 1000)
    smoother_values1 = cs1(finer_steps)
    smoother_values2 = cs2(finer_steps)
    smoother_values3 = cs3(finer_steps)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(finer_steps, smoother_values1, linestyle='-', color='r', label='Episode Mean PPO')
    plt.plot(finer_steps, smoother_values2, linestyle='-', color='b', label='Episode Mean MAPPO')
    plt.plot(finer_steps, smoother_values3, linestyle='-', color='g', label='Episode Mean Proposed algorithm')

    # Add labels and title
    plt.xlabel('Training episode')
    plt.ylabel('Episode reward (mean)')
    plt.title('Comparison of Training Algorithms')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()


def plot_value_vs_step_mean(mean_file, max_file, min_file):
    """
    Plots 'Value' against 'Step' from a given mean CSV file,
    with shaded regions representing max and min values from
    corresponding files.
    
    Parameters:
        mean_file (str): Path to the CSV file containing the mean values.
        max_file (str): Path to the CSV file containing the max values.
        min_file (str): Path to the CSV file containing the min values.
    """
    # Read the CSV files into pandas DataFrames
    mean_data = pd.read_csv(mean_file)
    max_data = pd.read_csv(max_file)
    min_data = pd.read_csv(min_file)
    
    # Extract the 'Step' and 'Value' columns
    steps = [x / 10 for x in mean_data['Step']]  # Scale steps as specified
    mean_values = mean_data['Value']
    max_values = max_data['Value']
    min_values = min_data['Value']
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot mean values
    plt.plot(steps, mean_values, linestyle='-', color='b', label='Mean Value')
    
    # Add shaded regions for max and min values
    plt.fill_between(steps, min_values, max_values, color='b', alpha=0.2, label='Min-Max Range')
    
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Episode Reward with Min-Max Range')
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.show()

# def plot_comparative_cost(csv_paths, labels):
    """
    Plots the comparative total cost at each timestep for three CSV files as a bar chart sharing the same axis.
    
    Args:
        csv_paths (list of str): List of paths to three CSV files containing cost data.
        labels (list of str): List of labels for each CSV file.
    """
    if len(csv_paths) != 3 or len(labels) != 3:
        raise ValueError("Three CSV files and three labels are required.")
    
    timestep_sets = []
    cost_data = []
    
    for csv_path in csv_paths:
        timesteps = []
        total_cost = []
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                timesteps.append(int(row[0]))
                total_cost.append(float(row[3]))
        timestep_sets.append(set(timesteps))
        cost_data.append(dict(zip(timesteps, total_cost)))
    
    # Find common timesteps across all files
    common_timesteps = sorted(set.intersection(*timestep_sets))
    
    # Extract cost data for common timesteps
    aligned_cost_data = [[costs[t] for t in common_timesteps] for costs in cost_data]
    
    x = np.arange(len(common_timesteps))
    width = 0.25
    
    plt.figure(figsize=(10, 5))
    for i, (cost, label) in enumerate(zip(aligned_cost_data, labels)):
        plt.bar(x + i * width, cost, width=width, label=label)
    
    plt.xlabel("Timestep")
    plt.ylabel("Total Cost ($)")
    plt.title("Comparative Total Cost at Each Timestep")
    plt.xticks(ticks=x + width, labels=common_timesteps)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_comparative_cost(csv_paths, price, labels):
    """
    Plots:
    1. The comparative total cost at each timestep for three CSV files as a bar chart.
    2. The total summed cost over 24 hours for each method as a separate bar chart.

    Args:
        csv_paths (list of str): List of paths to three CSV files containing cost data.
        labels (list of str): List of labels for each CSV file.
    """
    if len(csv_paths) != 3 or len(labels) != 3:
        raise ValueError("Three CSV files and three labels are required.")
    
    timestep_sets = []
    cost_data = []
    total_sums = []
    energy_data = []
    
    for csv_path in csv_paths:
        timesteps = []
        total_cost = []
        energy_consumption = []
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                timesteps.append(int(row[0]))
                total_cost.append(float(row[3]))
                energy_consumption.append(float(row[1]))
        
        timestep_sets.append(set(timesteps))
        cost_dict = dict(zip(timesteps, total_cost))
        energy_dict = dict(zip(timesteps, energy_consumption))
        cost_data.append(cost_dict)
        energy_data.append(energy_dict)
        total_sums.append(sum(total_cost))  # Compute total sum for each method
    
    # Find common timesteps across all files
    common_timesteps = sorted(set.intersection(*timestep_sets))
    

    aligned_cost_data = [[costs[t] for t in common_timesteps] for costs in cost_data]

    aligned_energy_data = [[e[t] for t in common_timesteps] for e in energy_data]
    
    x = np.arange(len(common_timesteps))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # ---- First Plot: Comparative Cost at Each Timestep ----
    plt.figure(figsize=(10, 5))
    for i, (cost, label) in enumerate(zip(aligned_cost_data, labels)):
        plt.bar(x + i * width, cost, width=width, label=label)
    
    plt.xlabel("Timestep")
    plt.ylabel("Total Cost ($)")
    plt.title("Comparative Total Cost at Each Timestep")
    plt.xticks(ticks=x + width, labels=common_timesteps)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

     # ---- Second Plot: Comparative Cost at Each Timestep ----
    fig, ax1 = plt.subplots(figsize=(14, 7))
    for i, (cost, label) in enumerate(zip(aligned_energy_data, labels)):
        ax1.bar(x + i * width, cost, width=width, label=label)
    
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Total energy demand(kWh)")
    ax1.set_title("Comparative energy demand at each timestep")
    ax1.set_xticks(ticks=x + width, labels=common_timesteps)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

     # Plot price signal on secondary y-axis
    ax2 = ax1.twinx()
    price_line, = ax2.plot(x, price, color="red", marker="o", label="Price signal", linestyle="--", linewidth=2)
    ax2.set_ylabel("Electricity price(€/kWh)")
    ax2.tick_params(axis="y")

    # Combine legends and place them at the bottom
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = [price_line], ["Price signal"]
    handles = handles1 + handles2
    labels_ = labels1 + labels2
    fig.legend(handles, labels_, loc="lower center", ncol=6)
    
    # Adjust layout to fit the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.show()
    
    # ----Third Plot: Total Cost Over 24 Hours ----
    plt.figure(figsize=(7, 5))
    plt.bar(labels, total_sums, color=["blue", "orange", "green"])
    
    plt.xlabel("Method")
    plt.ylabel("Total cost over 24 Hours (€)")
    plt.title("Total cost comparison over 24 hours")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def convert_csv_to_list(file_path):
    adjusted_operating_points = []
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            adjusted_operating_points.append([int(x) for x in row[1:-1]]) 
    
    return adjusted_operating_points

def convert_csv_to_list2(file_path):
    adjusted_operating_points = []
    
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            adjusted_operating_points.append([int(x) for x in row[1:]]) 
    
    return adjusted_operating_points



def compute_and_save_total_cost(data, price_signal, output_filename="total_cost.csv"):
    """
    Computes the total cost at each timestep by multiplying total energy consumption by price signal
    and saves the results as a CSV file.
    
    Args:
        data (dict): A dictionary where keys are timesteps (int) and values are
                     dictionaries with equipment names (str) as keys and energy
                     demand (float/int) as values.
        price_signal (list): A list of price signal values corresponding to each timestep.
        output_filename (str, optional): Name of the output CSV file. Defaults to "total_cost.csv".
    """
    # Extract timesteps
    timesteps = list(data.keys())
    
    # Validate price_signal length matches timesteps
    if len(price_signal) != len(timesteps):
        raise ValueError("Length of price_signal must match the number of timesteps in data.")
    
    # Compute total energy consumption at each timestep
    total_energy = []
    for timestep in timesteps:
        total_energy.append(sum(data[timestep].values()))
    
    # Compute total cost at each timestep
    total_cost = np.array(total_energy) * np.array(price_signal)
    
    # Save results to CSV file
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestep", "Total Energy (kWh)", "Price Signal", "Total Cost ($)"])
        for i, timestep in enumerate(timesteps):
            writer.writerow([timestep, total_energy[i], price_signal[i], total_cost[i]])
    
    print(f"Total cost data saved to {output_filename}")