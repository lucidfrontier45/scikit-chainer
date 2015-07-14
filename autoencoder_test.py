__author__ = 'du'

import numpy as np
from sklearn.datasets import make_classification, fetch_lfw_people
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from chainer import optimizers

from skchainer.linear import LogisticRegression
from skchainer.autoencoder import AutoEncoder, DenoisingAutoEncoder

lfw_people = fetch_lfw_people(min_faces_per_person=30, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
x = lfw_people.data.astype(np.float32) / 255
input_dim = x.shape[1]
hidden_dim = 200

# the label to predict is the id of the person
y = lfw_people.target.astype(np.int32)
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

model = LogisticRegression(optimizer=optimizers.SGD(), eps=1e-4,
                           n_dim=input_dim, n_classes=n_classes).fit(x_train, y_train)
print(model.score(x_test, y_test))

model = Pipeline([
    ("AutoEncoder", AutoEncoder(optimizer=optimizers.SGD(), eps=1e-4,
                                input_dim=input_dim, hidden_dim=hidden_dim, report=100)),
    ("LogisticRegression", LogisticRegression(optimizer=optimizers.SGD(), eps=1e-4,
                                              n_dim=hidden_dim, n_classes=n_classes))
]).fit(x_train, y_train)

print(model.score(x_test, y_test))
