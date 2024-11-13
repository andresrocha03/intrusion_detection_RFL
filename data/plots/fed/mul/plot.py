import pandas as pd
import matplotlib.pyplot as plt

plt.plot()
data = pd.read_csv("xgb_results.csv")
#the round is the row index+1
data['round'] = data.index + 1
plt.plot(data['round'], data['Accuracy'])
plt.xlabel('Round')
plt.ylabel('accuracies')
plt.title('acc vs Round')
plt.savefig("xgb_acc.png")
plt.show()
