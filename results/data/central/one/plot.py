import pandas as pd
import matplotlib.pyplot as plt

# Define metrics and models
metrics = ["Loss", "Accuracy"]
models = ["log_reg", "xgb", "mlp", "dqn"]
colors = ['red', 'blue', 'green', 'black']
labels = ['Logistic Regression', 'XGBoost', 'MLP', 'DQN']

# Load central results
central_results = pd.read_csv("central_res.csv")

metrics = ["Loss", "Accuracy", "Precision", "Recall", "Training Time"]

# bar plot metrics versus models for each metric
for m in metrics:
    plt.figure()
    #increase the precision of the y-ax
    plt.bar(models, central_results[m], color=colors)
    plt.xlabel('Model')
    plt.ylabel(m)
    plt.title(f'{m} vs Model')
    plt.savefig(f"central_{m}.png")
    plt.clf()


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

