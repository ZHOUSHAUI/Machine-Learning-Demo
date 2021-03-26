# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:22:04 2020

@author: Administrator
"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)

num_classes=10

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#归一化
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#一次训练所选取的样本数
batch_size = 128
#迭代次数
epochs = 2
input_shape =(28,28,1)

model = Sequential()
#第一个卷积层
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
#池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
#第二个卷积层
model.add(Conv2D(64, (3, 3), activation='relu'))
#池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
#展平所有像素 0.25的概率 （随机抽取25%展平）防止过拟合
model.add(Flatten())
model.add(Dropout(0.5))
#对所有展平的像素（保留下的）使用全连接层，输出128
model.add(Dense(num_classes, activation='softmax'))
#loss：损失函数
#optimizer:优化方法
#metrics：度量方法
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
#
#hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

model.save('mnist.h5')
print("Saving the model as mnist.h5")