import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
glioma_tumor = glob('/Training/glioma_tumor/*.jpg')
meningioma_tumor = glob('/Training/meningioma_tumor/*.jpg')
no_tumor = glob('/Training/no_tumor/*.jpg')
pituitary_tumor = glob('/Training/pituitary_tumor/*.jpg')


img_width = 48
img_height = 48

datagen = ImageDataGenerator(rescale=1/255.0,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True
)
test_datagen = ImageDataGenerator(rescale = 1./255)


train_data_gen = datagen.flow_from_directory(directory='C:\\Users\\Acer\PycharmProjects\\Brain_Tumor_Prediction\\Training\\',
                                             target_size=(img_width, img_height),
                                             class_mode='categorical')
vali_data_gen = test_datagen.flow_from_directory(directory='C:\\Users\\Acer\PycharmProjects\\Brain_Tumor_Prediction\\Testing\\',
                                            target_size=(img_width, img_height),
                                            class_mode='categorical')

print('Labels')
print(np.unique(train_data_gen.labels))
print(np.unique(vali_data_gen.labels))

print('\nTotal Labels')
print(len(np.unique(train_data_gen.labels)))
print(len(np.unique(vali_data_gen.labels)))
model = Sequential()

# convolution
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(192, (3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(228, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

r = model.fit_generator(generator=train_data_gen,
                              steps_per_epoch=len(train_data_gen),
                              epochs=30,
                              validation_data= vali_data_gen,
                              validation_steps = len(vali_data_gen))
plt.title('Loss')
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig('Loss')
plt.plot('Accuracy')
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig('Accuracy')
model.save('bt.h5')

