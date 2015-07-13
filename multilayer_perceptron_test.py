__author__ = 'du'

import numpy as np
from sklearn import datasets
from chainer import optimizers

from skchainer.mlp import MultiLayerPerceptron

iris = datasets.load_iris()
x = iris.data.astype(np.float32)
y = iris.target.astype(np.int32)

input_dim = x.shape[1]
n_classes = len(set(y))

model = MultiLayerPerceptron(optimizer=optimizers.AdaDelta(rho=0.5),
                             input_dim=input_dim, hidden_dim=2, n_classes=n_classes, report=100).fit(x, y)
print(model.score(x, y))
print(model.network.l1.parameters)
