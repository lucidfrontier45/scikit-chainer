__author__ = 'du'

import numpy as np
from skchainer import linear

x = np.random.randn(1000)[:, np.newaxis].astype(np.float32)
y = 2 * x + 1
model = linear.LinearRegression(ndim=1).fit(x, y)
print(model.score(x, y))