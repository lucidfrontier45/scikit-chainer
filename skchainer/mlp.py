__author__ = 'du'

from chainer import functions as F, ChainList
from . import ChainerClassifier


class MultiLayerPerceptron(ChainerClassifier):
    def _setup_network(self, **params):
        units = params["units"]
        # network = Chain(
        #     l1=F.Linear(params["input_dim"], params["hidden_dim"]),
        #     l2=F.Linear(params["hidden_dim"], params["n_classes"])
        # )

        network = ChainList(*[F.Linear(units[i], units[i + 1]) for i in range(len(units) - 1)])

        return network

    def _forward(self, x, train=True):
        h = x
        for f in self.network[:-1]:
            h = F.relu(f(h))
        y = self.network[-1](h)
        return y

    def _loss_func(self, y, t):
        return F.softmax_cross_entropy(y, t)


class DropoutMultiLayerPerceptron(MultiLayerPerceptron):
    def _forward(self, x, train=True):
        h = x
        for f in self.network[:-1]:
            h = F.dropout(F.relu(f(h)), train=train)
        y = self.network[-1](h)
        return y
