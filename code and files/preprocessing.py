import numpy as np
import pandas as pd
import os
import sys
sys.executable
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf

X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
paths = []
label = []

for i in labels:
    folderPath = os.path.join('/Users/Acer/PycharmProjects/Brain_Tumor_Prediction/Training', i)
    paths.append(folderPath)
    label.append(i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

for i in labels:
    folderPath = os.path.join('/Users/Acer/PycharmProjects/Brain_Tumor_Prediction/Testing', i)
    for j in os.listdir(folderPath):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        Y_train.append(i)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

data = {
    'Path': paths,
    'label': label
}
df = pd.DataFrame(data)

df.to_csv('dataset.csv', index=False)
from tkinter import *
root = Tk()
root.configure(background="white")
root.title("Done")
Label(root, text="Data Preprocessing Done!", font=("times new roman", 15), fg="white",
          bg="#000000",
          height=2).grid(row=0, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
root.mainloop()