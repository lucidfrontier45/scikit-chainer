__author__ = 'du'

from chainer import FunctionSet, functions as F
from . import ChainerTransformer


class AutoEncoder(ChainerTransformer):
    def _setup_network(self, **params):
        return FunctionSet(
            encoder=F.Linear(params["input_dim"], params["hidden_dim"]),
            decoder=F.Linear(params["hidden_dim"], params["input_dim"])
        )

    def _forward(self, x, train=False):
        z = self._transform(x, train)
        y = self.network.decoder(z)
        return y

    def _loss_func(self, y, t):
        return F.mean_squared_error(y, t)

    def _transform(self, x, train=False):
        return F.sigmoid(self.network.encoder(x))
