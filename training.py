import Analysis
from PIL import Image
import math
import pandas as pd
from numpy import asarray
import tensorflow as tf
import keras
import random
import numpy as np

path = r"datas/"
path2 = r"datas_processed/"
NL = 12
Y_out = []


def normalize(img: Image.Image, name: str) -> Image:
    w, h = img.size
    if w > h:
        process = img.crop((0, 0, h, h))
    elif h > w:
        process = img.crop((0, h - w, w, h))
    else:
        process = img

    process.save(path2 + name + ".jpg")
    return process


for i in range(1, NL + 1):
    fname = path + str(i) + ".jpg"
    valname = path + str(i) + ".txt"
    tmp = Image.open(fname)
    tmp = normalize(tmp, str(i))
    txt = tuple(i / tmp.size[0] for i in map(int, open(valname, "r").read().split()))

    Y_out.append(np.asarray(txt).astype(np.float32))
img_paths_lst = []
for i in range(1, NL + 1):
    img_paths_lst.append(path2 + str(i) + ".jpg")
img_paths = pd.Series(img_paths_lst, name="FilePath").astype(str)
values = pd.Series(Y_out, name="Values")
images = pd.concat([img_paths, values], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
print(images)
train_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./32,
    validation_split=.2
)
train_imgs = train_gen.flow_from_dataframe(
    dataframe=images,
    x_col="FilePath",
    y_col="Values",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="raw",
    batch_size=2,
    shuffle=True,
    seed=42,
    subset="training"
)
val_imgs = train_gen.flow_from_dataframe(
    dataframe=images,
    x_col="FilePath",
    y_col="Values",
    target_size=(224,224),
    color_mode="rgb",
    class_mode="raw",
    batch_size=2,
    shuffle=True,
    seed=42,
    subset="validation"
)

model = Analysis.create_model_2()
model.compile(optimizer="adam", loss=keras.losses.mse, metrics=["accuracy"])
model.fit(train_imgs, validation_data=val_imgs, epochs=12,
          callbacks=[
              keras.callbacks.EarlyStopping(
                  monitor="val_loss",
                  patience=5,
                  restore_best_weights=True
              )
          ])
print(model.predict(train_imgs))
model.save()
