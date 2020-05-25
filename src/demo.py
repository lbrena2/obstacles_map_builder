from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Index = np.linspace(1, -1, 20)
Cols = np.linspace(1, -1, 20)
df = DataFrame(abs(np.random.randn(20, 20)), index=Index, columns=Cols)

sns.heatmap(df, annot=True)

plt.show()


