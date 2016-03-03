__author__ = 'du'

from abc import ABCMeta, abstractmethod

import numpy as np
from chainer import Chain, Variable, optimizers
from chainer import functions as F
from sklearn import base


class BaseChainerEstimator(base.BaseEstimator, metaclass=ABCMeta):
    def __init__(self, optimizer=optimizers.SGD(), batch_size=10, n_iter=100, report=10,
                 network_params=None):
        if network_params is None:
            network_params = dict()
        self.network_params = network_params
        self.network = self._setup_network(**network_params)
        self.optimizer = optimizer
        self.optimizer.setup(self.network)
        self.n_iter = n_iter
        self.report = report
        self.batch_size = batch_size

    @abstractmethod
    def _setup_network(self, **params):
        return Chain(l1=F.Linear(1, 1))

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

        all_x = Variable(x_data)
        all_y = Variable(y_data)

        data_size = len(x_data)

        for epoch in range(self.n_iter):
            indexes = np.random.permutation(data_size)
            for i in range(0, data_size, self.batch_size):
                xx = Variable(x_data[indexes[i: i + self.batch_size]])
                yy = Variable(y_data[indexes[i: i + self.batch_size]])
                self.optimizer.zero_grads()
                loss = self._loss_func(self._forward(xx, train=True), yy)
                loss.backward()
                self.optimizer.update()

            if self.report > 0 and epoch % self.report == 0:
                loss = self._loss_func(self._forward(all_x), all_y)
                d_score = score - loss.data
                score = loss.data
                print(epoch, loss.data, d_score)

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
