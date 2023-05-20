import pickle
import numpy as np
from tensorflow.keras.datasets import cifar10

# cifar10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# pkl 파일로 저장
with open('cifar_10_shuffled.pkl', 'wb') as f:
    pickle.dump((x_train, y_train, x_test, y_test), f)
