__author__ = 'du'

import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from chainer import optimizers

from skchainer.linear import LogisticRegression
from skchainer.autoencoder import AutoEncoder

x, y = make_classification(1000, 20, 5, 2, 0, 4, random_state=0)
x = x.astype(np.float32)
y = y.astype(np.int32)

x_train = x[:900]
y_train = y[:900]
x_test = x[900:]
y_test = y[900:]

input_dim = x.shape[1]
hidden_dim = 40
n_classes = len(set(y))

model = LogisticRegression(optimizer=optimizers.AdaDelta(), eps=1e-5,
                           n_dim=input_dim, n_classes=n_classes).fit(x_train, y_train)
print(model.score(x_test, y_test))

model = Pipeline([
    ("AutoEncoder", AutoEncoder(optimizer=optimizers.AdaDelta(), eps=1e-5,
                                input_dim=input_dim, hidden_dim=hidden_dim, report=100)),
    ("LogisticRegression", LogisticRegression(optimizer=optimizers.AdaDelta(), eps=1e-5,
                                              n_dim=hidden_dim, n_classes=n_classes))
]).fit(x_train, y_train)

print(model.score(x_train, y_train))
