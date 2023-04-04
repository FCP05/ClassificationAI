#Project: Image Classification AI
#Class: CS4398 Section252-Group #1
#Member Names:
#   Carmona, Juan J.
#   PalaciosTinoco, Flavio C.
#   Mbwavi, Nickson L.
#   Polanco, Samuel S.
#File: AI MODEL STRUCTURE
#      ->Builds model structure.


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#Define the photo input shape(224pi,224pi,RGB).
input_shape = (224, 224, 3)

#Initialize the model.
model = Sequential()

#CONVOLUTION LAYERS are built with the following parameters:
#(image filters amount in use, kernel dimensions, padding to be used, activation function in use).
#POOLING LAYERS are built with the following parmeters:
#(image filter dimensions, stride of the filter).

#Add the first set of convolutional layers.
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#Add the second set of convolutional layers.
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#Add the third set of convolutional layers.
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#Add the fourth set of convolutional layers.
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#Add the fifth set of convolutional layers.
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

#FULLY CONNECTERD LAYERS process a "1-D" feature map. These layers are 
#populated with a number of desired nodes. The final layer designates
#the number of possible outputs available. 

#Add the fully connected layers.
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(8, activation='softmax'))

#Compile the model.
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])