import pandas as pd
import matplotlib.pyplot as plt

pd.read_csv("xgboost_res.csv")

plt.plot()
data = pd.read_csv("xgboost_res.csv")
#the round is the row index+1
data['round'] = data.index + 1
plt.plot(data['round'], data['Loss'])
plt.xlabel('Round')
plt.ylabel('accuracy')
plt.title('loss vs Round')
plt.savefig("loss_xgboost.png")
plt.show()
