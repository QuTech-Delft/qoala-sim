import json
import matplotlib.pyplot as plt
import numpy as np

def plot_success_probability_and_formula(json_file):
    # Load the JSON data from the provided file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Filter and organize the data based on param_value
    param_values = [0.95, 0.96, 0.97]
    prog_sizes = [2, 4, 6, 8, 10]
    
    # Create a plot
    plt.figure()
    
    # Plot data from JSON file (naive_succ_prob vs prog_size)
    for param_value in param_values:
        # Filter data for the current param_value
        filtered_data = [
            item for item in data['data_points']
            if item['param_value'] == param_value
        ]
        
        # Extract prog_size and naive_succ_prob for plotting
        prog_size_values = []
        naive_succ_prob_values = []
        
        for item in filtered_data:
            prog_size_values.append(item['prog_size'])
            naive_succ_prob_values.append(item['naive_succ_prob'])
        
        # Ensure we only plot for the required program sizes (2, 4, 6, 8, 10)
        filtered_prog_sizes = [p for p in prog_size_values if p in prog_sizes]
        filtered_succ_probs = [s for p, s in zip(prog_size_values, naive_succ_prob_values) if p in prog_sizes]

        # Plot the data for the current param_value
        plt.plot(filtered_prog_sizes, filtered_succ_probs, label=f'Subopt Data (F = {param_value})', marker='o')
    
    # Plot the formula (2F - 1)^n / 2 for F = 0.95, 0.96, 0.97
    n_values = np.array([2, 4, 6, 8, 10])
    
    for F in param_values:
        y_values = ((2 * F - 1) ** (n_values+10/6)+1) / 2
        plt.plot(n_values, y_values, label=f'Formula (F = {F})', linestyle='--', color='black')

    # Labels and title
    plt.ylim(0.5, 1.01)
    plt.xlabel("Program size (n)", fontsize=12)
    plt.ylabel("Success Probability", fontsize=12)
    # plt.title("Naive Success Probability vs Program size (n) for different param_values and Formula")

    # Add legend
    plt.legend(fontsize=10)

    # Display the plot
    # plt.show()
    plt.savefig("./plots/20250210_170032_single_gate_fid_rotation_TI_seed.png", transparent=True, dpi=1000)

# Example usage:
# plot_success_probability_and_formula('data.json')

# Example usage:
plot_success_probability_and_formula('./data/20250210_170032_single_gate_fid_rotation_config2/20250210_170032_single_gate_fid_rotation_TI_seed7.json')
