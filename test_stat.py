import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('test_ishikawa_wave.csv')

print(df)
df2 = df.groupby(['line', 'number']).mean()
print(df2)

prob = df2['score']
prob.plot.bar()
plt.show()