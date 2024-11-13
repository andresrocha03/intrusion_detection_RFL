import pandas as pd
import matplotlib.pyplot as plt

plt.plot()
data = pd.read_csv("mlp_results.csv")
#the round is the row index+1
data['round'] = data.index + 1
plt.plot(data['round'], data['Loss'])
plt.xlabel('Round')
plt.ylabel('loss')
plt.title('loss vs Round')
plt.savefig("mlp_loss.png")
plt.show()
