import tensorflow as tf
from keras import Sequential,layers,Model
import typing

def res(inp,stride,reshape = False,padding = "same") -> tf.Tensor:
    idd = inp
    x = layers.Conv2D(1, 3,strides=stride, padding = padding)(inp)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.Conv2D(1, 3, padding = padding)(x)

    x = layers.BatchNormalization()(x)
    if(stride > 1):
        idd = layers.Conv2D(1,1,padding = padding, strides = stride)(idd)
    x = layers.Add()([idd, x])
    x = layers.ReLU()(x)

    return x

# input 1024x1024 image
# output 4
def create_model():
    inputs = layers.Input(shape=(1024, 1024,1))
    i2 = layers.Conv2D(1,3)(inputs)
    r1 = res(i2,1)
    r2 = res(r1,2)
    out1 = layers.Dense(1)(r2)
    model = Model(inputs,out1)
    return model
