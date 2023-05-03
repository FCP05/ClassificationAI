import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

###PalaciosTinoco, Flavio###
#Define Model Structure Function:
#----------------------------------------------------
#Defines the VGG-16 Convolutional Neural Network(CNN). Takes two parameters
#dealing with image shape and number of image labels. The image shape is 
#expressed as "pixel value x pixel value x color channels" and the labels are
#expressed as "4" classes.

#Uses the softmax activation function and the categorical_crossentropy loss
#function to enforce dealing with 4 different image labels.
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    #Convolution Blocks
    conv1of1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2of1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1of1)
    pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2of1)

    conv1of2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2of2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv1of2)
    pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2of2)

    conv1of3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv2of3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv1of3)
    conv3of3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv2of3)
    pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3of3)

    conv1of4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv2of4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv1of4)
    conv3of4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv2of4)
    pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3of4)

    conv1of5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv2of5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv1of5)
    conv3of5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv2of5)
    pool5 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3of5)

    # Classification block
    flatten = layers.Flatten()(pool5)
    dense1 = layers.Dense(4096, activation='relu')(flatten)
    dense2 = layers.Dense(4096, activation='relu')(dense1)
    outputs = layers.Dense(num_classes, activation='softmax')(dense2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model