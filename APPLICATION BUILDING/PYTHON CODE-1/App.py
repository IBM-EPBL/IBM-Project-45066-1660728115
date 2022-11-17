#!/usr/bin/env python
# coding: utf-8

# Analyzing Data

# In[3]:


#importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import to_categorical
import cv2


# In[4]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[5]:


#Analyzing the Data
plt.imshow(X_train[0], cmap="binary")
plt.show()
print (y_train[0])


# In[6]:


#Reshaping the Data
print("---Before reshaping the Data---")
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print("Shape of X_test: {}".format(X_test.shape))
print("Shape of y_test: {}".format(y_test.shape))
print("-----------------------------------------")
print()
print("---After reshaping the Data---")
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of y_train: {}".format(y_train.shape))
print("Shape of X_test: {}".format(X_test.shape))
print("Shape of y_test: {}".format(y_test.shape))
print("-----------------------------------------")

