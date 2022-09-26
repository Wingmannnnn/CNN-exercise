import pandas as pd
import tensorflow
import keras
from keras.datasets import mnist
from keras.utils import np_utils

#---input data---
data = mnist.load_data()
(x_Train , y_Train),(x_Test , y_Test) = mnist.load_data()

#---feature scaling---
x_Train2D = x_Train.reshape(x_Train.shape[0],28,28,1).astype("float32")     #將x_train資料轉成三維(1為此資料為黑白圖片，若為彩色圖片必須為3(RGB))
x_Test2D = x_Test.reshape(x_Test.shape[0],28,28,1).astype("float32")

x_Train2D_normalized = x_Train2D / 255  #將所有值變成0~1範圍
x_Test2D_normalized = x_Test2D / 255

y_Train_OneHot = np_utils.to_categorical(y_Train)   #做one-hot-encoding的動作(原本會直接顯示數字)
y_Test_OneHot = np_utils.to_categorical(y_Test)

#---build model---
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=16,    #add convolution
                 kernel_size=(5,5),
                 padding="same",
                 input_shape=(28,28,1),
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))    #add maxpooling
model.add(Conv2D(filters=16,      #add convolution
                 kernel_size=(5,5),
                 padding="same",
                 activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))    #add maxpooling
model.add(Dropout(0.25))    #add dropout(避免ovetfitting)
model.add(Flatten())    #add flatten
model.add(Dense(128,activation="relu"))     #add dense
model.add(Dropout(0.5))     #add dropout
model.add(Dense(10,activation="softmax"))   #add dense
print(model.summary())

#---train model---
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
train_model = model.fit(x = x_Train2D_normalized,
                        y = y_Train_OneHot,
                        validation_split=0.2,
                        epochs=20,
                        batch_size=300,
                        verbose=2)
print(train_model)
