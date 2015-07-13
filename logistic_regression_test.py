__author__ = 'du'

import numpy as np
from skchainer import linear
from scipy import special

x = np.linspace(-10, 10, 10000).astype(np.float32)
p = special.expit(x)
y = np.random.binomial(1, p).astype(np.int32)
x = x.reshape(len(x), 1)
model = linear.LogisticRegression(n_dim=1, n_classes=2, report=100).fit(x, y)
print(model.score(x, y))
