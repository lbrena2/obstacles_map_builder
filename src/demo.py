import numpy as np
from scipy import stats

a = [value for value in range(20) if value%2==0]
print(a)

np.array(a).reshape((5,5))