import tensorflow as tf
from keras import Sequential,layers,Model


def res(inp,stride,reshape = False):
    x = layers.Conv2D(1, 3,strides=stride)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(1, 3,strides = stride)(x)
    x = layers.BatchNormalization()(x)
    if reshape: idd = layers.Reshape((inp.shape[-3]-4,inp.shape[-2]-4,1))(x)
    else: idd=inp
    x = layers.Add()([idd, x])
    x = layers.ReLU()(x)
    return x


# input 1024x1024 image
# output 4
inputs = layers.Input(shape=(1024, 1024,1))
i2 = layers.Conv2D(1,3)(inputs)
r1 = res(i2,1,reshape = True)
r2 = res(r1,2)
out1 = layers.Dense(1)(r2)
model = Model(inputs,out1)
model.summary()
