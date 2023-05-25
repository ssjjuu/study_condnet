import pickle
import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# make index for shuffling
idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)

# shuffle train data
x_train = x_train[idx]
y_train = y_train[idx]

# make xtrain to 50000,3,32,32 and xtest to 10000,3,32,32
x_train = x_train.transpose(0, 3, 1, 2)
x_test = x_test.transpose(0, 3, 1, 2)

# make float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

with open('cifar_10_shuffled.pkl', 'wb') as f:
    pickle.dump((x_train, y_train, x_test, y_test), f)