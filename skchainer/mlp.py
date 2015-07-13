__author__ = 'du'

from chainer import FunctionSet, functions as F
from . import ChainerClassifier


class MultiLayerPerceptron(ChainerClassifier):
    def _setup_network(self, **params):
        network = FunctionSet(
            l1=F.Linear(params["input_dim"], params["hidden_dim"]),
            l2=F.Linear(params["hidden_dim"], params["n_classes"])
        )

        return network

    def forward(self, x):
        h = F.relu(self.network.l1(x))
        y = self.network.l2(h)
        return y

    def loss_func(self, y, t):
        return F.softmax_cross_entropy(y, t)

    def output_func(self, h):
        return F.softmax(h)
