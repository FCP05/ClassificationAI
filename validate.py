import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import imageprep.py
#This file tests to see how accurate our model is.
#It takes in an image, processes it, then loads training data, and prints out the score.
def validate(img_name) :
    image = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
) #gets validation image

    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Create batch axis
    #Load training data from "somefilename.h5"
    predictions = model.predict(image_array) #makes prediction and prints it based on loaded training data
    score = float(predictions[0])
    print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")