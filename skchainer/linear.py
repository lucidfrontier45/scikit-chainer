__author__ = 'du'

from chainer import FunctionSet, functions as F
from . import ChainerRegresser, ChainerClassifier


class LinearRegression(ChainerRegresser):
    def _setup_network(self, **params):
        return FunctionSet(l1=F.Linear(params["n_dim"], 1))

    def forward(self, x):
        y = self.network.l1(x)
        return y

    def loss_func(self, y, t):
        return F.mean_squared_error(y, t)

    def output_func(self, h):
        return F.identity(h)


class LogisticRegression(ChainerClassifier):
    def _setup_network(self, **params):
        return FunctionSet(l1=F.Linear(params["n_dim"], params["n_class"]))

    def forward(self, x):
        y = self.network.l1(x)
        return y

    def loss_func(self, y, t):
        return F.softmax_cross_entropy(y, t)

    def output_func(self, h):
        return F.softmax(h)
