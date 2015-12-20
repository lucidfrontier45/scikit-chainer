__author__ = 'du'

import numpy as np
from skchainer import linear

x = np.linspace(-10, 10, 100).astype(np.float32)
y = 2 * x + 1 + np.random.randn(len(x)).astype(x.dtype) * 5
x = x.reshape(len(x), 1)
y = y.reshape(len(y), 1)
model = linear.LinearRegression(n_dim=1, report=1).fit(x, y)
print(model.score(x, y))

for param in model.network.params():
    print(param.name, param.data)