#from livelossplot import PlotLossesKeras
# import module from tkinter for UI
from matplotlib.pyplot import *
    # from array import *
from numpy import *
from tkinter import *
import os
root = Tk()
img_size = 48
batch_size = 64
def datapreprocessing():
    os.system("py preprocessing.py")
def nbtraining():
    os.system("py cnn.py")
def plotacc():
    os.system("py cnnacc.py")
def function6():
    root.destroy()
def appopen():
    os.system("py app.py")
root.configure(background="white")
root.title("Brain Tumor Detection Using Deep Learning")
# creating a text label
Label(root, text="Brain Tumor Detection Using Deep Learning", font=("times new roman", 20), fg="white", bg="#1A3C40",
      height=2).grid(row=0, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Data Preprocessing", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=datapreprocessing).grid(
    row=1, columnspan=2, sticky=N + E + W + S,padx=75, pady=15)
Button(root, text="Model Training", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=nbtraining).grid(
    row=2, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Model Accuracy", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=plotacc).grid(
    row=3, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
# creating second button
Button(root, text="Brain Tumor Detection Web App", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=appopen).grid(
    row=4, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
Button(root, text="Exit", font=('times new roman', 20), bg="#EDE6DB", fg="#3e2723", command=function6).grid(
    row=5, columnspan=2, sticky=N + E + W + S, padx=75, pady=15)
root.mainloop()
