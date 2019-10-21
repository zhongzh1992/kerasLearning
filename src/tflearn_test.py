import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences


def test_pad():
    trainX = 'w18476 w4454 w1674 w6 w25 w474 w1333 w1467 w863 w6 w4430 w11 w813 w4463 w863 w6 w4430 w111'
    trainX = trainX.split(" ")
    trainX = pad_sequences([trainX], maxlen=100, value=0.)
    print("trainX:", trainX)


test_pad()
