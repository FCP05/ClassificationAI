import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import sys

###Carmona, Juan J.###
#Prediction Function:
#-----------------------------------------------------
#Loads a previously trained model. JPG file is a parameter.
#Image is applied to the model and the model tells which of
#4 labels it is and a certainty score is shown. 
def make_prediction(givenImg):
    print("Load Previously Trained Model...")
    model = load_model('vgg16_saveState.h5')
    print("Model Load Successful")
    
    print("Load Desired Image...")
    imgPath = givenImg
    img = image.load_img(imgPath, target_size=(56, 56))
    print("Image Load Sucessful")
    
    print("Preprocess Image...")
    imgDataArray = image.img_to_array(img)
    imgDataArray = tf.keras.preprocessing.image.smart_resize(imgDataArray, size=(56, 56))
    imgDataArray = preprocess_input(imgDataArray)
    imgDataArray = np.expand_dims(imgDataArray, axis=0)
    imgDataArray = tf.keras.applications.resnet50.preprocess_input(imgDataArray)
    print("Preprocess Successful")
    
    # Make a prediction
    print("Make Prediction...")
    prediction = model.predict(imgDataArray)
    predictedClass = np.argmax(prediction)
    
    #List of subfolder names
    subfolders = os.listdir('Images')
    
    # Iterate through subfolder names
    accumulator = 0
    for folder in subfolders:
        #Accumulator finds the class name
        if accumulator == predictedClass:
            className = folder
            break
        else:
            #Accumulator does not find the class name
            accumulator = accumulator + 1 
        
    certainty = prediction[0][predictedClass]
        
    print(f"Prediction Made: {className} ({certainty*100:.2f}% certainty)")
    print(prediction)
 
make_prediction(sys.argv[1])