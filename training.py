import Analysis
from PIL import Image
import math
import pandas as pd
from numpy import asarray
import tensorflow as tf
import keras
import random
path = r"datas/"
NL = 12
X_inp1 = []
Y_out = []

for i in range(1, NL + 1):
    fname = path + str(i) + ".jpg"
    valname = path + str(i) + ".txt"
    tmp = Image.open(fname)
    b = asarray(tmp)[:1024,:1024,:]
    tensor_inp = tf.convert_to_tensor(b)
    X_inp1.append(tensor_inp)
    txt = tuple(i/1024 for i in map(int, open(valname, "r").read().split()))[0]

    Y_out.append(txt)
X_inp = asarray(X_inp1)
Y_out = asarray(Y_out)
train = tf.keras.preprocessing.image.ImageDataGenerator()
model = Analysis.create_model()
model.compile(optimizer="adam", loss=keras.losses.mse,metrics = ["accuracy"])
model.fit(X_inp,Y_out, validation_data= (X_inp,Y_out),epochs=12,batch_size=1)
print(model.predict(X_inp))
model.save()

