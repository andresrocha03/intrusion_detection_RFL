import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_file = 'log_reg_results.csv'
# Read existing results or create a new DataFrame if it doesn't exist
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
else:
    results_df = pd.DataFrame(columns=['Model Name', 'Loss', 'Accuracy'])


# Save the DataFrame to a CSV file
results_df.to_csv(results_file, index=False)