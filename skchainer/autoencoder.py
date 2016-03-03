__author__ = 'du'

import numpy as np
from chainer import Chain, functions as F

from . import BaseChainerEstimator, ChainerTransformer


class AutoEncoder(ChainerTransformer):
    def __init__(self, activation=F.relu, **params):
        super(ChainerTransformer, self).__init__(**params)
        self.activation = activation

    def _setup_network(self, **params):
        return Chain(
            encoder=F.Linear(params["input_dim"], params["hidden_dim"]),
            decoder=F.Linear(params["hidden_dim"], params["input_dim"])
        )

    def _forward(self, x, train=False):
        z = self._transform(x, train=train)
        y = self.network.decoder(z)
        return y

    def _loss_func(self, y, t):
        return F.mean_squared_error(y, t)

    def _transform(self, x, train=False):
        return self.activation(self.network.encoder(x))


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, noise_ratio=0.2, **params):
        super(DenoisingAutoEncoder, self).__init__(**params)
        self.noise_ratio = float(noise_ratio)

    def _setup_network(self, **params):
        return Chain(
            encoder=F.Linear(params["input_dim"], params["hidden_dim"]),
            decoder=F.Linear(params["hidden_dim"], params["input_dim"])
        )

    def _transform(self, x, train=False):
        return self.activation(self.network.encoder(x))

    def fit(self, x_data, y_data=None):
        s = np.std(x_data, 0) * self.noise_ratio
        noisy_x = x_data + np.random.randn(*x_data.shape) * s
        return BaseChainerEstimator.fit(self, x_data, noisy_x.astype(np.float32))


class DropoutAutoEncoder(AutoEncoder):
    def _transform(self, x, train=False):
        return F.dropout(self.activation(self.network.encoder(x)), train=train)
