Dataset_path = '/home/ec2-user/SageMaker/ECG_analysis/scalogram_plots/'

import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
import random

images = sorted(os.listdir(Dataset_path))

images = [image for image in images if '.png' in image]

### GLobal Config
s_x = 91;
e_x = 647;
s_y = 44;
e_y = 314;

size_x = e_x - s_x ;
size_y = e_y - s_y;

n = len(images)
n_cat = 3;

print(size_y,size_x)

def show_i(x):
    img = cv.imread(Dataset_path + images[x], 0)
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.imshow(img[s_y:e_y,s_x:e_x])
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.show()
    
def show_img(x):
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.show()
    
X = np.zeros(shape=(n,270,556),dtype=np.uint8)
Y = np.zeros(shape=(n))
Y = Y.astype('int8') 

def load_data(train_percentage):
    for i in range(0,n):
        img = cv.imread(Dataset_path + images[i], 0)
        X[i]= img[s_y:e_y,s_x:e_x]

        if images[i][-5]=='N':
            Y[i] = 2
        elif images[i][-5]=='~':
            Y[i] = 0
        else:
            Y[i] = 1

        if(i%1000==0):
            print(i)

    print(str(n)+ " Images loaded across " + '3' + " Categories")  


    xtrain = []
    ytrain = []

    xtest = []
    ytest = []

    X_Lists = [0,0,0]
    Y_Lists = [0,0,0]

    #
    img_names = np.array(images)
    names = [img_names[Y==2] , img_names[Y==1] , img_names[Y==0]] 
    
    X_Lists[0] = X[Y==2] #normal
    X_Lists[1] = X[Y==1] #abnormal
    X_Lists[2] = X[Y==0] #other

    Y_Lists[0] = Y[Y==2] #normal
    Y_Lists[1] = Y[Y==1] #abnormal
    Y_Lists[2] = Y[Y==0] #other

    for l in range(0,3):
        count = len(Y_Lists[l])

        for i in range(0,int(train_percentage*count)):
            xtrain.append(list(X_Lists[l][i]))
            ytrain.append(Y_Lists[l][i])

        for i in range(int(train_percentage*count),count):
            xtest.append(list(X_Lists[l][i]))
            ytest.append(Y_Lists[l][i])
    
    x_train = np.array(xtrain, dtype=np.uint8)
    y_train = np.array(ytrain)

    x_test = np.array(xtest,dtype=np.uint8)
    y_test = np.array(ytest)

    return n,(x_train, y_train), (x_test, y_test)

Xx = np.zeros(shape=(n,270,556),dtype=np.uint8)
Yy = np.zeros(shape=(n))
Yy = Y.astype('int8') 

def log(train_percentage):
    for i in range(0,n):

        if images[i][-5]=='N':
            Yy[i] = 2
        elif images[i][-5]=='~':
            Yy[i] = 0
        else:
            Yy[i] = 1

    print(str(n)+ " Images loaded across " + '3' + " Categories")  


    ytrain = []

    ytest = []

    Y_Lists = [0,0,0]

    img_names = np.array(images)
    names = [img_names[Y==2] , img_names[Y==1] , img_names[Y==0]] 
    

    Y_Lists[0] = Y[Y==2] #normal
    Y_Lists[1] = Y[Y==1] #abnormal
    Y_Lists[2] = Y[Y==0] #other

    train_names = []
    test_names  = []
    for l in range(0,3):
        count = len(Y_Lists[l])

        for i in range(0,int(train_percentage*count)):
            train_names.append(names[l][i])
            ytrain.append(Y_Lists[l][i])

        for i in range(int(train_percentage*count),count):
            test_names.append(names[l][i])
            ytest.append(Y_Lists[l][i])
    

    return n,train_names,test_names


