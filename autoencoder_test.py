__author__ = 'du'

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from chainer import optimizers

from skchainer.linear import LogisticRegression
from skchainer.autoencoder import DenoisingAutoEncoder

X = np.random.randn(1000, 2).astype(np.float32)
y = np.array((X ** 2).sum(1) > 1.0, dtype=np.int32)

input_dim = X.shape[1]
output_dim = len(set(y))
hidden_dim = 10

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

model = LogisticRegression(optimizer=optimizers.SGD(), eps=1e-4,
                           n_dim=input_dim, n_classes=output_dim).fit(x_train, y_train)

print("LogisticRegressoin Score")
print(model.score(x_test, y_test))

model = Pipeline([
    ("AutoEncoder", DenoisingAutoEncoder(optimizer=optimizers.SGD(), eps=1e-4,
                                input_dim=input_dim, hidden_dim=hidden_dim, report=100)),
    ("LogisticRegression", LogisticRegression(optimizer=optimizers.SGD(), eps=1e-4,
                                              n_dim=hidden_dim, n_classes=output_dim))
]).fit(x_train, y_train)

print("LogisticRegressoin with AutoEncoder Score")
print(model.score(x_test, y_test))
