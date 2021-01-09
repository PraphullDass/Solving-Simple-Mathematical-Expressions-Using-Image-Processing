# importing necessary modules
import numpy as np
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, ReLU

# defining random seed
np.random.seed(28)

# function to read the training dataset
# we obtain 800 images for each category
def read_data(filename):
    pwd = os.getcwd()
    training = os.path.join(pwd, filename)
    categories = os.listdir(training)
    x = []
    y = []
    kernel = np.ones(shape=(3,3), dtype = np.uint8)
    for category in categories:
        path = os.path.join(training, category)
        files = random.sample(os.listdir(path), 800)
        for file in files:
            temp = 255 - cv2.imread(os.path.join(path, file), 0)
            temp1 = np.zeros(shape = (temp.shape[0] + 10, temp.shape[1] + 10), dtype = np.uint8)
            temp1[5:5+temp.shape[0], 5:5+temp.shape[1]] = temp.copy()
            temp2 = cv2.dilate(temp1.copy(), kernel.copy(), iterations = 1)
            temp2[temp2 >= 10] = 255
            temp2[temp2 < 10] = 0
            x.append(cv2.resize(temp2.copy(), (60, 60)).reshape(60, 60, 1))
            y.append(category)
    x = np.array(x)
    y = np.array(y)
    y[y == "("] = 10
    y[y == ")"] = 11
    y[y == "+"] = 12
    y[y == "-"] = 13
    y[y == "times"] = 14
    y[y == "div"] = 15
    y[y == "0"] = 0
    y[y == "1"] = 1
    y[y == "2"] = 2
    y[y == "3"] = 3
    y[y == "4"] = 4
    y[y == "5"] = 5
    y[y == "6"] = 6
    y[y == "7"] = 7
    y[y == "8"] = 8
    y[y == "9"] = 9
    y = y.astype(int)
    Y = keras.utils.to_categorical(y, 16)
    X = x.astype(float) / 255
    return X, Y

# function to build our CNN model and compile it
def get_model():
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 5, padding = "same", input_shape=  (60, 60, 1)))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters = 64, kernel_size = 5, padding = "same"))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters = 96, kernel_size = 5, padding = "same"))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size = 2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = "softmax"))
    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    return model

# function to train our CNN model
def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test))
    return model

# function to save out CNN model
def save_model(model):
    model.save("PRAPHULL_DASS_2018071_CNN_MODEL", save_format = "h5")
    return

# function to split our data into train and test to evaluate our CNN model
def split_data(X, Y):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, shuffle = True)
    return x_train, x_test, y_train, y_test

# main function
def run(filename):
    print("Starting process")
    X, Y = read_data(filename)
    print("Data read successfully")
    x_train, x_test, y_train, y_test = split_data(X.copy(), Y.copy())
    model = get_model()
    print("CNN model created successfully")
    model = train_model(model, x_train, y_train, x_test, y_test, 100, 20)
    print("CNN model trained successfully")
    save_model(model)
    print("CNN model saved successfully")
    print("End of process")
    return

# calling main function
run("Train_Data")
