__author__ = 'du'

from chainer import Chain, functions as F
from . import ChainerRegresser, ChainerClassifier


class LinearRegression(ChainerRegresser):
    def _setup_network(self, **params):
        return Chain(l1=F.Linear(params["n_dim"], 1))

    def _forward(self, x, train=False):
        y = self.network.l1(x)
        return y

    def _loss_func(self, y, t):
        return F.mean_squared_error(y, t)


class LogisticRegression(ChainerClassifier):
    def _setup_network(self, **params):
        return Chain(l1=F.Linear(params["n_dim"], params["n_classes"]))

    def _forward(self, x, train=False):
        y = self.network.l1(x)
        return y

    def _loss_func(self, y, t):
        return F.softmax_cross_entropy(y, t)
