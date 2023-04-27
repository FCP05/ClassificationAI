from imgPrep import prepGenerateData
from model import make_model
from tensorflow import keras
import matplotlib.pyplot as plt

###Mbwavi, Nickson L.###
#Train the Model Function:
#------------------------------------------------------------
#Compiles a model instance, generates a picture of the model
#to a file, trains/validates the model, saves model weights
#to a file, and plots model performance in training. 
def trainModel():    
    train_ds, val_ds = prepGenerateData()
    
    print("Call Model Function...")
    image_size=(56,56,3)
    model = make_model(input_shape=image_size, num_classes=4)
    print("Model Call Successful")
    
    keras.utils.plot_model(model, show_shapes=True)
    print("Model Illustration Successful")
    
    epochs = 10
    
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_epoch{epoch}_progress.keras"),
    ]
    
    print("Model Compilation...")
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("Model Compilation Successful")
    
    print("Model Training/Validation...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )
    print("Model Training/Validation Successful")
    
    model.save('vgg16_saveState.h5')
    print("Model Save Successful")
    
    # Plot the training and validation loss over epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Plot the training and validation accuracy over epochs
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
trainModel()