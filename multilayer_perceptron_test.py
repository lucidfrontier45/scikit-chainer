__author__ = 'du'

import numpy as np
from chainer import optimizers
from sklearn import datasets, cross_validation

from skchainer.mlp import DropoutMultiLayerPerceptron

iris = datasets.load_iris()
x = iris.data.astype(np.float32)
y = iris.target.astype(np.int32)

input_dim = x.shape[1]
n_classes = len(set(y))

network_params = {"units": [input_dim, 100, 50, 10, n_classes]}
model = DropoutMultiLayerPerceptron(optimizer=optimizers.AdaDelta(), report=0, n_iter=200,
                                    network_params=network_params)

print(model.get_params())

# score = model.fit(x, y).score(x, y)
score = cross_validation.cross_val_score(model, x, y, cv=5, n_jobs=-1)

print(score)
