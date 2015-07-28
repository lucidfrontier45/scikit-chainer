# scikit-chainer
scikit-learn like interface to chainer

## How to install

```bash
$ pip install scikit-chainer
```

## what's this?
This is a scikit-learn like interface to the chainer deeplearning framework.
You can use it to build your network model and use the model with scikit-learn APIs (e.g. `fit`, `predict`)
There are `ChainerRegresser` for regression, `ChainerClassifer` for classification base classes and `ChainerTransformer` for transformation.
You need to inherit them and implement the following functions,

1. `_setup_network` : network definition (`FunctionSet` of chainer)
2. `forward` : emit the result `z` input `x` (note this is not the final predicted value)
3. `loss_func`: the loss function to minimize (e.g. `mean_squared_error`, `softmax_cross_entropy` etc)
4. `output_func` : emit the final result `y` from forwarded values `z` (e.g. `identity` for regression and `softmax` for classification. 

## Example

### Linear Regression

```python
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
```

### LogisticRegression
```python
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
```

### AutoEncoder

```python
class AutoEncoder(ChainerTransformer):
    def __init__(self, activation=F.relu, **params):
        super(ChainerTransformer, self).__init__(**params)
        self.activation = activation

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
        return self.activation(self.network.encoder(x))
```
