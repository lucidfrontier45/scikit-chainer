__author__ = 'du'

import numpy as np
from chainer import optimizers
from sklearn import datasets, cross_validation

from skchainer import linear

iris = datasets.load_iris()
x = iris.data.astype(np.float32)
y = iris.target.astype(np.int32)

n_dim = 4
n_classes = 3

model = linear.LogisticRegression(optimizer=optimizers.AdaDelta(),
                                  network_params=dict(n_dim=n_dim, n_classes=n_classes),
                                  n_iter=500, report=0)

score = cross_validation.cross_val_score(model, x, y, cv=5, n_jobs=-1)

print(score)
