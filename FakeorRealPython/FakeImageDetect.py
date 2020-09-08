#{'Fake': 0, 'Real': 1}
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from tkinter import messagebox

import numpy as np
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import keras
import os
import cv2

from keras import applications
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
import socket
from PIL import Image

main = tkinter.Tk()
main.title("Fake or Real Photo Detection with Deep Learning") #designing main screen
main.geometry("600x500")

global filename
global loaded_model

def upload(): 
    global filename
    filename = filedialog.askopenfilename(initialdir="test_dataset")
    messagebox.showinfo("File Information", "image file loaded")
    

def generateModel():
    global loaded_model
    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
           loaded_model_json = json_file.read()
           loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("Model/model_weights.h5")
        loaded_model._make_predict_function()   
        print(loaded_model.summary())
        messagebox.showinfo("Deep Learning CNN Model Generated", "Deep Learning CNN Model Generated on Train & Test Data. See black console for details")
    else:
        files = []
        label = []
        filename = 'train_dataset/fake_train_data'
        for root, dirs, directory in os.walk(filename):
            for i in range(len(directory)):
                if directory[i] != 'Thumbs.db':
                    files.append(filename+"/"+directory[i]);
                    label.append(0)

        filename = 'train_dataset/real_train_data'
        for root, dirs, directory in os.walk(filename):
            for i in range(len(directory)):
                if directory[i] != 'Thumbs.db':
                    files.append(filename+"/"+directory[i]);
                    label.append(1)
        X = []
        Y = []
        for i in range(len(files)):
            img = cv2.imread(files[i])
            print(files[i])
            img = cv2.resize(img, (64,64))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(64,64,3)
            X.append(im2arr)
            Y.append(label[i])
        X = np.asarray(X)
        Y = np.asarray(Y)
        print(X.shape)
        Y = to_categorical(Y)
        print(Y.shape)
        img = X[20].reshape(64,64,3)
        cv2.imshow('ff',cv2.resize(img,(250,250)))
        cv2.waitKey(0)
        print("shape == "+str(X.shape))
        print("shape == "+str(Y.shape))
        print(Y)
        X = X.astype('float32')
        X = X/255
        classifier = Sequential() 
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print(classifier.summary())
        classifier.fit(X, Y, batch_size=16, epochs=18, validation_split=0.1, shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        loaded_model = classifier
        messagebox.showinfo("Deep Learning CNN Model Generated", "Deep Learning CNN Model Generated on Train & Test Data. See black console for details")


def classify():
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    X = np.asarray(im2arr)
    X = X.astype('float32')
    X = X/255
    preds = loaded_model.predict(X)
    print(str(preds)+" "+str(np.argmax(preds)))
    predict = np.argmax(preds)
    print(predict)
    msg = ''
    if predict == 0:
        msg = 'Photo detected as FAKE'
    if predict == 1:
        msg = 'Photo detected as REAL'
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,600))
    cv2.putText(img, 'Prediction Result : '+msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow('Prediction Result : '+msg,img) 
    cv2.waitKey(0)
    
def recvall(sock, size):
    msg = ''
    while len(msg) < size:
        part = sock.recv(size-len(msg)).decode()
        if 'hello' in part:
            msg += part
            print("break")
            break 
        msg += part
    return msg.strip()

def predictImg():
    img = cv2.imread('test.png')
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    X = np.asarray(im2arr)
    X = X.astype('float32')
    X = X/255
    preds = loaded_model.predict(X)
    print(str(preds)+" "+str(np.argmax(preds)))
    predict = np.argmax(preds)
    print(predict)
    msg = ''
    if predict == 0:
        msg = 'Photo detected as FAKE'
    if predict == 1:
        msg = 'Photo detected as REAL'
    return msg

def runServer():
    host = "192.168.0.7"
    print(host)
    port = 5000
    server_socket = socket.socket()
    server_socket.bind((host, port))
    while True:   
        server_socket.listen(2)
        conn, address = server_socket.accept()
        data = recvall(conn,100000)
        print(data)
        arr = data.split("#")
        width = int(arr[0])
        height = int(arr[1])
        im = Image.new("RGB", (width, height))
        print(width)
        print(height)
        pix = im.load()
        for i in range(2,64):
            pixels = arr[i].split(",")
            for y in range(len(pixels)):
                temp = pixels[y].split("?")
                #print(temp)
                pix[i,y] = (int(temp[0]),int(temp[1]),int(temp[2]))

        im.save("test.png", "PNG")
        data = predictImg()+'\r\n'
        print(data)
        conn.send(data.encode())
        
    
font = ('times', 16, 'bold')
title = Label(main, text='Fake or Real Photo Detection with Deep Learning', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 14, 'bold')
model = Button(main, text="Generate Deep Learning Model", command=generateModel)
model.place(x=200,y=100)
model.config(font=font1)  

uploadimage = Button(main, text="Upload Test Image", command=upload)
uploadimage.place(x=200,y=150)
uploadimage.config(font=font1) 

classifyimage = Button(main, text="Classify Photo", command=classify)
classifyimage.place(x=200,y=200)
classifyimage.config(font=font1) 

exitapp = Button(main, text="Run Server to Receive Photo from Android", command=runServer)
exitapp.place(x=200,y=250)
exitapp.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
