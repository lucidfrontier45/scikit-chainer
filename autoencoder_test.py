__author__ = 'du'

import numpy as np
from chainer import optimizers
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline

from skchainer.autoencoder import DenoisingAutoEncoder, DropoutAutoEncoder
from skchainer.linear import LogisticRegression

X = np.random.randn(1000, 2).astype(np.float32)
y = np.array((X ** 2).sum(1) > 1.0, dtype=np.int32)

input_dim = X.shape[1]
output_dim = len(set(y))
hidden_dim = 10

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = LogisticRegression(optimizer=optimizers.Adam(), report=10, n_iter=100,
                           network_params=dict(n_dim=input_dim, n_classes=output_dim)
                           ).fit(x_train, y_train)

print("LogisticRegressoin Score")
print(model.score(x_test, y_test))

model = Pipeline([
    ("AutoEncoder", DenoisingAutoEncoder(optimizer=optimizers.Adam(), n_iter=100,
                                         network_params=dict(input_dim=input_dim, hidden_dim=hidden_dim, report=10))),
    ("LogisticRegression", LogisticRegression(optimizer=optimizers.Adam(), n_iter=100, report=10,
                                              network_params=dict(n_dim=hidden_dim, n_classes=output_dim)))
]).fit(x_train, y_train)

print("LogisticRegressoin with DenoisingAutoEncoder Score")
print(model.score(x_test, y_test))

model = Pipeline([
    ("AutoEncoder", DropoutAutoEncoder(optimizer=optimizers.Adam(), n_iter=100,
                                       network_params=dict(input_dim=input_dim, hidden_dim=hidden_dim, report=10))),
    ("LogisticRegression", LogisticRegression(optimizer=optimizers.Adam(), n_iter=100, report=10,
                                              network_params=dict(n_dim=hidden_dim, n_classes=output_dim)))
]).fit(x_train, y_train)

print("LogisticRegressoin with DropoutAutoEncoder Score")
print(model.score(x_test, y_test))
