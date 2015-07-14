__author__ = 'du'

from abc import ABCMeta, abstractmethod
from chainer import FunctionSet, Variable, optimizers
from chainer import functions as F
from sklearn import base
import numpy as np


class BaseChainerEstimator(base.BaseEstimator, metaclass=ABCMeta):
    def __init__(self, optimizer=optimizers.SGD(), batch_size=100, n_iter=10000, eps=1e-5, report=100,
                 **params):
        self.network = self._setup_network(**params)
        self.optimizer = optimizer
        self.optimizer.setup(self.network.collect_parameters())
        self.n_iter = n_iter
        self.eps = eps
        self.report = report
        self.batch_size = batch_size

    @abstractmethod
    def _setup_network(self, **params):
        return FunctionSet(l1=F.Linear(1, 1))

    @abstractmethod
    def _forward(self, x, train=False):
        y = self.network.l1(x)
        return y

    @abstractmethod
    def _loss_func(self, y, t):
        return F.mean_squared_error(y, t)

    def fit(self, x_data, y_data=None):
        score = 1e100
        if y_data is None:
            y_data = x_data

        for i in range(self.n_iter):
            if 0 < self.batch_size < len(x_data):
                idx = np.random.permutation(self.batch_size)
                xx = x_data[idx]
                yy = y_data[idx]
            else:
                xx = x_data
                yy = y_data
            x = Variable(xx)
            t = Variable(yy)
            self.optimizer.zero_grads()
            loss = self._loss_func(self._forward(x, train=True), t)
            loss.backward()
            self.optimizer.update()
            d_score = score - loss.data
            score = loss.data
            if d_score < self.eps:
                print(i, loss.data, d_score)
                break
            if self.report > 0 and i % self.report == 0:
                print(i, loss.data, d_score)

        return self


class ChainerRegresser(BaseChainerEstimator, base.RegressorMixin):
    def predict(self, x_data):
        x = Variable(x_data)
        y = self._forward(x, train=False)
        return y.data


class ChainerClassifier(BaseChainerEstimator, base.ClassifierMixin):
    def predict(self, x_data):
        x = Variable(x_data)
        y = self._forward(x, train=False)
        return F.softmax(y).data.argmax(1)


class ChainerTransformer(BaseChainerEstimator, base.TransformerMixin):
    @abstractmethod
    def _transform(self, x, train=False):
        raise NotImplementedError

    def transform(self, x_data):
        x = Variable(x_data)
        z = self._transform(x)
        return z.data

    def fit(self, x_data, y_data=None):
        return BaseChainerEstimator.fit(self, x_data, None)
