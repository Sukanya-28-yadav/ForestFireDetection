import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential
import pickle
from keras.applications import VGG16
from keras.applications import VGG19
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn import svm
'''
path = 'Dataset'
X = []
Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j])
            img = cv2.resize(img, (32,32))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(32,32,3)
            X.append(im2arr)
            if name == "Fire":
                Y.append(1)
            else:
                Y.append(0)
            print(name)

X = np.asarray(X)
Y = np.asarray(Y)
print(Y)
print(Y.shape)

np.save('model/X.txt',X)
np.save('model/Y.txt',Y)
'''
X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')

X = X.astype('float32')
X = X/255
    
test = X[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

vgg = VGG16(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
for layer in vgg.layers:
    layer.trainable = False
vggnet = Sequential()
vggnet.add(vgg)
vggnet.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
vggnet.add(MaxPooling2D(pool_size = (1, 1)))
vggnet.add(Convolution2D(32, (1, 1), activation = 'relu'))
vggnet.add(MaxPooling2D(pool_size = (1, 1)))
vggnet.add(Flatten())
vggnet.add(Dense(units = 256, activation = 'relu'))
vggnet.add(Dense(units = y_train.shape[1], activation = 'softmax'))
vggnet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/vggnet_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/vggnet_weights.hdf5', verbose = 1, save_best_only = True)
    hist = vggnet.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/vggnet_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    vggnet.load_weights("model/vggnet_weights.hdf5")
predict = vggnet.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)       


vgg = VGG19(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
for layer in vgg.layers:
    layer.trainable = False
vggnet_extension = Sequential()
vggnet_extension.add(vgg)
vggnet_extension.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
vggnet_extension.add(MaxPooling2D(pool_size = (1, 1)))
vggnet_extension.add(Convolution2D(32, (1, 1), activation = 'relu'))
vggnet_extension.add(MaxPooling2D(pool_size = (1, 1)))
vggnet_extension.add(Flatten())
vggnet_extension.add(Dense(units = 256, activation = 'relu'))
vggnet_extension.add(Dense(units = y_train.shape[1], activation = 'softmax'))
vggnet_extension.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/extension_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
    hist = vggnet_extension.fit(X_train, y_train, batch_size = 64, epochs = 20, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/extension_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    vggnet_extension.load_weights("model/extension_weights.hdf5")

predict = vggnet_extension.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test, predict)
print(acc)



X_train1 = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))
X_test1 = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))
y_train = np.argmax(y_train, axis=1)

X_train1 = X_train1[0:5000]
y_train = y_train[0:5000]
X_test1 = X_test1[0:1000]
y_test = y_test[0:1000]

if os.path.exists("model/svm_model"):
    f = open("model/svm_model", 'rb')
    svm_cls = pickle.load(f)
    f.close()
else:
    svm_cls = svm.SVC(kernel="rbf")
    svm_cls.fit(X_train1, y_train)
    f = open("model/svm_model", 'wb')
    pickle.dump(svm_cls, f)
    f.close()
print("done")    
predict = svm_cls.predict(X_test1)
acc = accuracy_score(y_test, predict)
print(acc)







                      
