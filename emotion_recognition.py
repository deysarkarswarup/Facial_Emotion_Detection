import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
#data will be availabe in follwing link
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/overview
df=pd.read_csv('fer2013.csv')

X_train,train_y,X_test,test_y=[],[],[],[]

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")

num_features = 64
num_labels = 7
batch_size = 64
epochs = 32
width, height = 48, 48

X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

#normalizing data between oand 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

##designing the cnn in keras
#1st convolution layer
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))#picks up max value 
model.add(Dropout(0.5))#to stop overfitting

#2nd convolution layer
model.add(Conv2D(2*64, (3, 3), activation='relu'))
model.add(Conv2D(2*64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))#picks up max value 
model.add(Dropout(0.5))#to stop overfitting

#3rd convolution layer
model.add(Conv2D(2*2*64, (3, 3), activation='relu'))
model.add(Conv2D(2*2*64, (3, 3), activation='relu'))
# model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())#transforming the pool into a single column

#fully connected neural networks
model.add(Dense(2*2*64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2*64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))
#model.summary()

#Compliling the model
model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

#Training the model
model.fit(X_train, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, test_y), shuffle=True)

#Saving the  model
fer_json = model.to_json()
with open("facialModels.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("facialWeights.h5")
