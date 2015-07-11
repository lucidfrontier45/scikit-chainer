__author__ = 'du'

from abc import ABCMeta, abstractmethod
from chainer import FunctionSet, Variable, optimizers
from chainer import functions as F
from sklearn import base


class BaseChainerEstimator(base.BaseEstimator, metaclass=ABCMeta):
    def __init__(self, optimizer=optimizers.SGD(), n_iter=10000, eps=1e-5, report=100,
                 **params):
        self.network = self._setup_network(**params)
        self.optimizer = optimizer
        self.optimizer.setup(self.network.collect_parameters())
        self.n_iter = n_iter
        self.eps = eps
        self.report = report

    @abstractmethod
    def _setup_network(self, **params):
        return FunctionSet(l1=F.Linear(1, 1))

    @abstractmethod
    def forward(self, x):
        y = self.network.l1(x)
        return y

    @abstractmethod
    def loss_func(self, y, t):
        return F.mean_squared_error(y, t)

    @abstractmethod
    def output_func(self, h):
        return F.identity(h)

    def fit(self, x_data, y_data):
        score = 1e100
        x = Variable(x_data)
        t = Variable(y_data)
        for i in range(self.n_iter):
            self.optimizer.zero_grads()
            loss = self.loss_func(self.forward(x), t)
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

    def predict(self, x_data):
        x = Variable(x_data)
        y = self.forward(x)
        return self.output_func(y).data


class ChainerRegresser(BaseChainerEstimator, base.RegressorMixin):
    pass


class ChainerClassifier(BaseChainerEstimator, base.ClassifierMixin):
    def predict(self, x_data):
        return BaseChainerEstimator.predict(self, x_data).argmax(1)
