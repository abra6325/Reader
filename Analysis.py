import tensorflow as tf
from keras import Sequential, layers, Model


import typing


def res(inp, stride, reshape=False, padding="same",filter_num = 4) -> tf.Tensor:
    idd = inp
    x = layers.Conv2D(filter_num, 3, strides=stride, padding=padding)(inp)
    x = layers.BatchNormalization()(x)
    print(x.shape)
    x = layers.Conv2D(filter_num, 3, padding=padding)(x)

    x = layers.BatchNormalization()(x)
    if (stride > 1):
        idd = layers.Conv2D(filter_num, 1, padding=padding, strides=stride)(idd)
    x = layers.Add()([idd, x])
    x = layers.ReLU()(x)

    return x


# input 1024x1024 image
# output 4
def create_model() -> Model:
    inputs = layers.Input(shape=(1024, 1024, 3))
    i2 = layers.Conv2D(1, 3)(inputs)
    r1 = res(i2, 1,filter_num = 3)
    r2 = res(r1, 2,filter_num = 3)
    r3 = res(r2,2)
    r4 = res(r3,2)
    r5 = res(r4,2)
    r6 = res(r5,2)
    r7 = res(r6,2,filter_num=2)
    r8 = res(r7,8,filter_num =1)
    r9 = res(r8,4,filter_num=1)
    shape = layers.Reshape((1,1))(r9)
    out1 = layers.Dense(1)(shape)
    model = Model(inputs, out1)
    return model
create_model().summary()