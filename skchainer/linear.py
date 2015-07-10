__author__ = 'du'

from sklearn import base
from chainer import FunctionSet, functions as F
from . import BaseChainerEstimator


class LinearRegression(BaseChainerEstimator, base.RegressorMixin):
    def _setup_network(self, **params):
        return FunctionSet(l1=F.Linear(params["ndim"], 1))

    def forward(self, x):
        y = self.network.l1(x)
        return y

    def loss_func(self, y, t):
        return F.mean_squared_error(y, t)

    def output_func(self, x):
        return F.identity(x)
