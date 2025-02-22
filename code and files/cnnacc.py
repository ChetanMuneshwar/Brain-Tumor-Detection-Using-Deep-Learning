
#import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, Dropout
from sklearn.metrics import precision_score, recall_score, f1_score

# load weights into new model
model = tf.keras.models.load_model("bt.h5")
print("Loaded model from disk")

# Initialize image data generator with rescaling
test_data_gen = ImageDataGenerator(rescale=1./255,
shear_range = 0.2,
              zoom_range = 0.2,
                           horizontal_flip = True
)

# Preprocess all test images
# Preprocessing the Test set
datagen_validation = ImageDataGenerator(horizontal_flip=True)
validation_generator = test_data_gen.flow_from_directory("Training/",
                                                         target_size=(48, 48),
                                                         batch_size=32,
                                                         class_mode='categorical',
                                                         shuffle=False)
print(validation_generator)
# do prediction on test data
predictions = model.predict_generator(validation_generator)

# see predictions
# for result in predictions:
#     max_index = int(np.argmax(result))
#     print(emotion_dict[max_index])

print("-----------------------------------------------------------------")
# confusion matrix
c_matrix = confusion_matrix(validation_generator.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()
print(accuracy_score(validation_generator.classes, predictions.argmax(axis=1)))
cc = accuracy_score(validation_generator.classes, predictions.argmax(axis=1))
cc = round(cc * 100,2)
# Classification report
#print("-----------------------------------------------------------------")
print(classification_report(validation_generator.classes, predictions.argmax(axis=1)))




from tkinter import *
root = Tk()
root.configure(background="white")
root.title("CNN Accuracy")

root.configure(background="white")
root.title("CNN Accuracy")
Label(root, text="CNN Accuracy:{}".format(cc
                                          ), font=("times new roman", 15), fg="white",
          bg="#000000",
          height=2).grid(row=0, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
root.mainloop()