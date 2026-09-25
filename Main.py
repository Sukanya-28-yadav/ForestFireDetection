#importing require python packages and classes
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
import pickle
import pandas as pd
from keras.applications import VGG16 #loading VGG16 as based classifier
from keras.applications import VGG19 #loading VGG19 as extension classifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("A Deep Learning-Based Experiment on Forest Wildfire Detection in Machine Vision Course")
main.geometry("1300x1200")

#define global variables to calculate and store accuracy and other metrics
precision = []
recall = []
fscore = []
accuracy = []
global filename, dataset,predict
global X, Y,vggnet_extension
global X_train, X_test, y_train, y_test
global X_train1,X_test1,y_train1,y_test1

def UploadDataset():
    global X, Y
    global filename, dataset
    #load dataset images
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'): #if dataset already processed then load it
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else: #if not processed then process, save, and load for future execution
        X = []
        Y = []
        path = 'Dataset'
        for root, dirs, directory in os.walk(path):#loop all images in dataset
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j]) #read image
                    img = cv2.resize(img, (32,32)) #resize image
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr) #add images to array
                    if name == "Fire": #if fire then set label as 1 else 0 for non fire
                        Y.append(1)
                    else:
                        Y.append(0)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)

    text.insert(END,"Dataset Loading Completed\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Classes found in dataset are Fire & No-Fire")
    unqiue, count = np.unique(Y, return_counts = True)
    height = count
    bars = ['No Fire', 'Fire']
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Fire Or No-Fire Graph")
    plt.ylabel("Count")
    plt.title("Dataset Class Label Graph")
    plt.tight_layout()
    plt.show()
def VNFSI():
    #display no fire image
    text.delete('1.0', END)
    text.insert(END,"No Fired Sample Image Detected\n")
    no_fire = cv2.imread("Dataset/No_Fire/lake_resized_lake_frame22.jpg")
    plt.imshow(no_fire)
    plt.title('Processed No Fire Sample Image')
    plt.axis('off')
    plt.show()
    text.insert(END,"No Fire Image Sample Image Detected\n")

def VFSI():
    #display fire image
    text.delete('1.0', END)
    text.insert(END,"Fired Sample Image Detected\n")
    fire = cv2.imread("Dataset/Fire/resized_frame1.jpg")
    plt.imshow(cv2.cvtColor(fire, cv2.COLOR_RGB2BGR))
    plt.title('Processed Fire Sample Image')
    plt.axis('off')
    plt.show()

def DataPreprocessing():
    global filename, dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train1,X_test1,y_train1,y_test1
    #dataset preprocessing such as shuffling and normalization
    text.delete('1.0', END)
    X = X.astype('float32')
    X = X/255 #normalizing images
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffling images
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    text.insert(END,"Dataset Normalization & Shuffling Process completed \n")
    #now splitting dataset into train & test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset train & test split as 80% dataset for training and 20% for testing\n")
    text.insert(END,"Training Size (80%): "+str(X_train.shape[0])+"\n") #print training and test size
    text.insert(END,"Testing Size (20%): "+str(X_test.shape[0]))

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    text.delete('1.0', END)
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100     
    print()
    text.insert(END,algorithm+' Accuracy  : '+str(a)+"\n")
    text.insert(END,algorithm+' Precision   : '+str(p)+"\n")
    text.insert(END,algorithm+' Recall      : '+str(r)+"\n")
    text.insert(END,algorithm+' FMeasure    : '+str(f)+"\n")    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    labels = ["No Fire", "Fire"]
    conf_matrix = confusion_matrix(testY, predict) 
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

def RunSVM():
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train1,X_test1,y_train1,y_test1
    #now train SVM algorithm on color features
    X_train1 = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))
    X_test1 = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))
    y_train1 = np.argmax(y_train, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #now train SVM and then calculate accuracy on trained model
    if os.path.exists("model/svm_model"):
        f = open("model/svm_model", 'rb')
        svm_cls = pickle.load(f)
        f.close()
    else:
        svm_cls = svm.SVC(kernel="rbf")
        svm_cls.fit(X_train1, y_train1)
        f = open("model/svm_model", 'wb')
        pickle.dump(svm_cls, f)
        f.close()
    predict = svm_cls.predict(X_test1)
    calculateMetrics("SVM", predict, y_test1)
def RunVGG16():
    global filename, dataset,vggnet_extension
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train1,X_test1,y_train1,y_test1
    #now train propose VGG-Reducenet algorithm by applying VGG16 as transfer learning and in propose VGG-ReduceNet we are reducing
    #VGG16 layers by setting to false
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in vgg.layers:
        layer.trainable = False #reducing VGG16 layers
    vggnet = Sequential()#creating own CNN object
    vggnet.add(vgg)#adding VGG16 reduce layer object to new model as VGG ReduceNet 
    #define new layers for VGG reduceNet
    vggnet.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vggnet.add(MaxPooling2D(pool_size = (1, 1)))
    vggnet.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vggnet.add(MaxPooling2D(pool_size = (1, 1)))
    vggnet.add(Flatten())
    vggnet.add(Dense(units = 256, activation = 'relu'))
    vggnet.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    #compile and train model
    vggnet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/vggnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vggnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vggnet.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vggnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vggnet.load_weights("model/vggnet_weights.hdf5")
    #perform prediction using VGGReduceNet and then calculate metrics    
    predict = vggnet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_label = np.argmax(y_test, axis=1)
    calculateMetrics("Propose VGG-ReduceNet", predict, y_label)

def RunVGG19():
    global filename, dataset,vggnet_extension,predict
    global X, Y
    global X_train, X_test, y_train, y_test
    global X_train1,X_test1,y_train1,y_test1
    #now as extension we are using VGG19 as transfer learning and then comparing its performnace with Propose VGG-ReduceNet
    #creating VGG19 object
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in vgg.layers:
        layer.trainable = False #setting VGG19 layers to reduce
    vggnet_extension = Sequential()
    #adding VGG19 as transer learning to make extension VGG
    vggnet_extension.add(vgg)
    #defining new layers for VGG extension
    vggnet_extension.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    vggnet_extension.add(MaxPooling2D(pool_size = (1, 1)))
    vggnet_extension.add(Convolution2D(32, (1, 1), activation = 'relu'))
    vggnet_extension.add(MaxPooling2D(pool_size = (1, 1)))
    vggnet_extension.add(Flatten())
    vggnet_extension.add(Dense(units = 256, activation = 'relu'))
    vggnet_extension.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    vggnet_extension.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #compiling and training model
    if os.path.exists("model/extension_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vggnet_extension.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vggnet_extension.load_weights("model/extension_weights.hdf5")
    #perform prediction using VGG extension
    predict = vggnet_extension.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_label = np.argmax(y_test, axis=1)
    calculateMetrics("Extension VGG19-ReduceNet", predict, y_label)

def RunGraph():
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    df = pd.DataFrame([['Existing SVM','Precision',precision[0]],['Existing SVM','Recall',recall[0]],['Existing SVM','F1 Score',fscore[0]],['Existing SVM','Accuracy',accuracy[0]],
                       ['Propose VGG16-ReduceNet','Precision',precision[1]],['Propose VGG16-ReduceNet','Recall',recall[1]],['Propose VGG16-ReduceNet','F1 Score',fscore[1]],['Propose VGG16-ReduceNet','Accuracy',accuracy[1]],
                       ['Extension VGG19-ReduceNet','Precision',precision[2]],['Extension VGG19-ReduceNet','Recall',recall[2]],['Extension VGG19-ReduceNet','F1 Score',fscore[2]],['Extension VGG19-ReduceNet','Accuracy',accuracy[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("Existing SVM, Propose VGG16-ReduceNet & Extension VGG19-ReduceNet Performance Graph")
    plt.show()


def vibeAnnotate(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 10, 120])
    upper_red = np.array([15, 255, 255])
    mask = cv2.inRange (hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for c in contours:
            #red_area = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w >= 5 and h >= 10:
                cv2.rectangle(img,(x, y),(x+w, y+h),(0, 0, 255), 2)
    return img

#function to predict fire or no fire from images or videos

def predict():
    global filename, dataset, vggnet_extension
    global X, Y
    global X_train, X_test, y_train, y_test
    labels = ['No Fire', 'Fire']
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = vggnet_extension.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 10, 120])
    upper_red = np.array([15, 255, 255])

    mask = cv2.inRange (hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for c in contours:
            #red_area = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if w >= 5 and h >= 10:
                cv2.rectangle(img,(x, y),(x+w, y+h),(0, 0, 255), 2)
          
    cv2.putText(img, 'Prediction Output : '+labels[predict]+" Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('Image Classified as : '+labels[predict], img)
    cv2.waitKey(0)

def VideoPredict():
    global filename, dataset,vggnet_extension
    global X, Y
    global X_train, X_test, y_train, y_test
    #now detect fire from video
    labels = ['No Fire', 'Fire']
    filename = "testVideos/videoplayback.mp4" #loading video
    video = cv2.VideoCapture(filename)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while(True):
        ret, frame = video.read()
        img = cv2.resize(frame, (32,32))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255 #normalizing test image
        predict = vggnet_extension.predict(img)#now using vgg19 extension to predict as fire or no fire
        predict = np.argmax(predict)
        if predict == 1:
            frame = vibeAnnotate(frame)
        cv2.putText(frame, 'Prediction Output : '+labels[predict]+" Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)    
        cv2.imshow("Frame", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
def Exit():
    main.destroy()
    

font = ('times', 14, 'bold')
title = Label(main, text='A Deep Learning-Based Experiment on Forest Wildfire Detection in Machine Vision Course')
title.config(bg='yellow3', fg='Black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)



font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=UploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)

font1 = ('times', 13, 'bold')
VNFSIButton = Button(main, text="View No Fire Sample Image", command=VNFSI)
VNFSIButton.place(x=180,y=100)
VNFSIButton.config(font=font1)

font1 = ('times', 13, 'bold')
VFSIButton = Button(main, text="View Fire Sample Image", command=VFSI)
VFSIButton.place(x=406,y=100)
VFSIButton.config(font=font1)

font1 = ('times', 13, 'bold')
VFSIButton = Button(main, text="Features Processing & Normalizaton", command=DataPreprocessing)
VFSIButton.place(x=608,y=100)
VFSIButton.config(font=font1)

font1 = ('times', 13, 'bold')
SVMButton = Button(main, text="Run SVM Algorithm", command=RunSVM)
SVMButton.place(x=905,y=100)
SVMButton.config(font=font1)

font1 = ('times', 13, 'bold')
VGG16Button = Button(main, text="VGG16-ReduceNet", command=RunVGG16)
VGG16Button.place(x=50,y=150)
VGG16Button.config(font=font1)

font1 = ('times', 13, 'bold')
VGG19Button = Button(main, text="Extension VGG-ReduceNet", command=RunVGG19)
VGG19Button.place(x=212,y=150)
VGG19Button.config(font=font1)

font1 = ('times', 13, 'bold')
GraphButton = Button(main, text="Performance Graph", command=RunGraph)
GraphButton.place(x=434,y=150)
GraphButton.config(font=font1)

font1 = ('times', 13, 'bold')
PredictButton = Button(main, text="Predict From Test Images", command=predict)
PredictButton.place(x=600,y=150)
PredictButton.config(font=font1)

font1 = ('times', 13, 'bold')
VButton = Button(main, text="Predict From Test Videoes", command=VideoPredict)
VButton.place(x=810,y=150)
VButton.config(font=font1)

font1 = ('times', 13, 'bold')
ExitButton = Button(main, text="Exit", command=Exit)
ExitButton.place(x=1030,y=150)
ExitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

main.config(bg='yellow')
main.mainloop()
