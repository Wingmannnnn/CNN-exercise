from keras.preprocessing import image
from keras.models import Sequential
from keras.utils import image_utils
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import numpy as np

#---preprocessing---
train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    "dataset/training_set",
    target_size=(64,64),
    batch_size=32,
    class_mode="binary"
)
test_datagen = image.ImageDataGenerator(rescale=1./255)
testing_set = test_datagen.flow_from_directory(
    "dataset/test_set",
    target_size=(64,64),
    batch_size=32,
    class_mode="binary"
)

#---build CNN---
cnn = Sequential()
cnn.add(Conv2D(
    filters=32,
    kernel_size=3,
    input_shape=[64,64,3],
    activation="relu"
))
cnn.add(MaxPooling2D(pool_size= 2,strides= 2))
cnn.add(Conv2D(
    filters=32,
    kernel_size=3,
    activation="relu"
))
cnn.add(MaxPooling2D(pool_size= 2,strides= 2))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
#full connection layer
cnn.add(Dense(units= 128,activation= "relu"))
cnn.add(Dropout(0.5))
#output layer
cnn.add(Dense(units= 1,activation= "sigmoid"))

#---train CNN---
cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
cnn.fit(x=training_set, validation_data=testing_set, epochs=30)

#---prediction---
# test_img = image_utils.load_img("dataset/single_prediction/cat_or_dog_1.jpg",target_size = (64,64))
# test_img = image_utils.img_to_array(test_img)
# test_img = np.expand_dims(test_img, axis= 0)
# result = cnn.predict(test_img)
print(training_set.class_indices)
# if result[0][0] == 1:
#     prediction = "trans"
# else:
#     prediction = "person"
# print(prediction)
cnn.save("fuck.h5")