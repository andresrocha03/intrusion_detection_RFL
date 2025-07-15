import os
from typing import Dict
import argparse
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import utils
from sklearn.metrics import log_loss
from typing import List, Tuple
from flwr.common import Context, Metrics


# Load your dataset
df = pd.read_csv("x_one_train.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

model = LogisticRegression()

model.fit(X,y.ravel())

df = pd.read_csv("x_one_test.csv")
X = df.drop('label', axis=1).values
y = df['label'].values

predic = model.predict(X)

print(predic)