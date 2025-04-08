import pandas as pd
import matplotlib.pyplot as plt

# Define metrics and models
metrics = ["Loss", "Accuracy"]
models = ["log_reg", "xgb", "mlp", "dqn"]
colors = ['red', 'blue', 'green', 'black']
labels = ['Logistic Regression', 'XGBoost', 'MLP', 'DQN']

# Load central results
central_results = pd.read_csv("central_res.csv")

# Loop through models and metrics
for metric in metrics:
    plt.figure()
    for model, color, label in zip(models, colors, labels):
        # Load the per-round data for the model
        data = pd.read_csv(f"{model}_res.csv")
        data['round'] = data.index + 1  # Add round numbers

        # Get the central value for the metric from central_res.csv
        central_value = central_results.loc[central_results["Model Name"] == model, metric].values[0]

        # Plot the per-round metric
        plt.plot(data['round'], data[metric],color=color, label=f"{label} {metric}", linestyle='-')
        
        # Add the central result as a dashed line
        plt.axhline(y=central_value, color=color, linestyle='--', label=f"Central {label}")

        # Customize the plot
        plt.xlabel('Rounds')
        plt.ylabel(metric)
        plt.title(f"{metric} vs Rounds")
        
        # Adjust the legend position
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

        plt.tight_layout()  # Adjust layout to prevent clipping
    
        # Save the figure
        plt.savefig(f"{metric}.png")
    plt.clf()  # Clear the figure for the next plot



# for model in models:
#     for m in metrics:
#         data = pd.read_csv(f"{model}_res.csv")
#         # The round is the row index + 1
#         data['round'] = data.index + 1
#         plt.figure()  # Create a new figure for each plot
#         plt.plot(data['round'], data[m])
#         plt.xlabel('round')
#         plt.ylabel(m)
#         plt.title(f'{m} vs round')
#         plt.savefig(f"{model}_{m}.png")
#         plt.clf()  # Clear the figure after saving it    

