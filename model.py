#---------------------------------------------------------------------
#Project: Image Classification AI
#Class: CS4398 Section252-Group #1
#Member Names:
#   Carmona, Juan J.
#   PalaciosTinoco, Flavio C.
#   Mbwavi, Nickson L.
#   Polanco, Samuel S.
#About File: AI MODEL STRUCTURE DEFINITION
#      ->Defines model structure. 
#      ->"Squeeze Net()" is called and built for training.
#      ->"Squeeze Net()" is called and built for validation.
#      ->Model always blank unless training progress is fetched
#        from "modelSaveState.h5". 
#---------------------------------------------------------------------
from keras.models import Model
from keras.layers import Add, Activation, Concatenate, Conv2D 
from keras.layers import Flatten, Input, GlobalAveragePooling2D, MaxPooling2D
import keras.backend as K

    
def SqueezeNet(inputShape, numOfLabels):
    """
    Basic SqueezeNet Structure
    (No Bypassing, Compression of "1.0", and No Dropout rate)
    
    Arguments:
        inputShape: shape of the input images e.g. (224,224,3).
        numOfLabels: number of classes.
    Returns:
        Model: Keras Model instance stored as "Model".
    """
    input_img = Input(shape=inputShape)

    layer_x = Conv2D(96, (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(input_img)

    layer_x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxPool1')(layer_x)
    
    layer_x = createFireModule(layer_x, 16, name='fire2')
    layer_x = createFireModule(layer_x, 16, name='fire3')
    layer_x = createFireModule(layer_x, 32, name='fire4')
    
    layer_x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxPool4')(layer_x)
    
    layer_x = createFireModule(layer_x, 32, name='fire5')
    layer_x = createFireModule(layer_x, 48, name='fire6')
    layer_x = createFireModule(layer_x, 48, name='fire7')
    layer_x = createFireModule(layer_x, 64, name='fire8')
    
    layer_x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name='maxPool8')(layer_x)
    
    layer_x = createFireModule(layer_x, 64, name='fire9')
        
    layer_x = output(layer_x, numOfLabels)
    return Model(inputs=input_img, outputs=layer_x)

def createFireModule(layer_x, numOfSqueezeFilter, name):
    """
    Creates a Fire Module
    
    Arguments:
        layer_x: Input
        numSqueezeFilter: Number of filters for squeeze. The filtersize for expand is 4 times of squeeze
        name: Name of current layer like "fire3","maxPool1", or "fire9_squeeze"
    Returns:
        layer_x: returns a fire module
    """
    
    numOfExpandFilter = 4 * numOfSqueezeFilter
    squeeze    = Conv2D(numOfSqueezeFilter,(1,1), activation='relu', padding='same', name='%s_squeeze'%name)(layer_x)
    expand_1x1 = Conv2D(numOfExpandFilter, (1,1), activation='relu', padding='same', name='%s_expand_1x1'%name)(squeeze)
    expand_3x3 = Conv2D(numOfExpandFilter, (3,3), activation='relu', padding='same', name='%s_expand_3x3'%name)(squeeze)
    
    alayer_x = getALayer_X()
    layer_x_ret = Concatenate(axis=alayer_x, name='%s_Concatenate'%name)([expand_1x1, expand_3x3])
    return layer_x_ret

def getALayer_X():
    alayer_x = -1 if K.image_data_format() == 'channels_last' else 1
    return alayer_x
    
   
def output(layer_x, numOfLabels):
    layer_x = Conv2D(numOfLabels, (1,1), strides=(1,1), padding='valid', name='conv10')(layer_x)
    layer_x = GlobalAveragePooling2D(name='avgPool10')(layer_x)
    layer_x = Activation("softmax", name='softmax')(layer_x)
    return layer_x