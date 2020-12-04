import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)

print(mnist.keys())
print(mnist['url'])

X, y = mnist['data'], mnist['target']

print(X[0])

print(type(mnist))
print(len(X[0]))

print(len(mnist['data']))
print(len(mnist['target']))

for pair in mnist.items():
    print(pair)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[1]
some_digit_image = some_digit.reshape(28,28)
#
# plt.imshow(some_digit_image, cmap= mpl.cm.binary, interpolation= "nearest")
# plt.axis("off")
# plt.show()

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test ==5)

print(y_train_5)
print(y_train[0])
print(y_train[1])
print(type(y_train))

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print(sgd_clf.predict([some_digit]))
