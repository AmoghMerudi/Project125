import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image 
import PIL.ImageOps 

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 10, test_size = 2500, train_size = 7500)

X_train_scale = X_train/255
X_test_scale = X_test/255

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scale, y_train)

def getPrediction(image):
    imgpil = Image.open
    imgbw = imgpil.convert("L")
    imgbwresize = imgbw.resize((28,28), Image.ANTIALIAS)

    pixelfactor = 20
    minpixel = np.percentile(imgbwresize, pixelfactor)

    imgbwresize_scale = np.clip(imgbwresize - minpixel, 0, 255)
        
    maxpixel = np.max(imgbwresize)

    imgbwresize_scale = np.asarray(imgbwresize_scale)/maxpixel

    test_sample = np.array(imgbwresize_scale).reshape(1,784)
    test_prediction = clf.predict(test_sample)

    return test_prediction[0]