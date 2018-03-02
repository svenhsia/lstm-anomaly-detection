import numpy as np
import pandas as pd
from collections import defaultdict

data = pd.read_csv('cleaned_data/feature_table.csv')
cols = data.columns
ts = data['timestamp']
distribution = defaultdict(int)
for t in ts:
    distribution[t // 1000] += 1
print(distribution)
