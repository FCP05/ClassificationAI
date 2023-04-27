# Classification AI using a VGG-16 Convolutional Neural Network
In short, this is a program that is shown a group of images of declared tags, then it can draw a prediction for what tag(of the ones previously introduced) a brand new image belongs in.

## Current Status: 
- Preprocessing, training, and prediction functions run without crashing.
- Upon running the prediction function for the model, the model favors one image tag/class over the other three available tags.
- Symptoms point to an overfitted model:
  1. Diverging pattern begins to take shape upon finishing the 10th epoch for training loss and validation loss.
  2. Diverging pattern begins to take shape upon finishing the 10th epoch for training accuracy and validation accuracy.
(Anticipated Solution) Apply more data augmentation techniques to the data set, tweak the number of epochs, experiment with different batch sizes, and make sure any corrupt image data is completley taken care of.    

## Features
  The VGG-16 Convolutional Neural Network(VGG-16 CNN) architecture looks at image features as it moves through the layers of the network. Using square filter/kernel structures, it selects edges and other simple shapes on the image. Moving further into the model,  more abstract features are dealt with such as a mouth, an ear, a furry tail, a wheel, or a beak. Finally, the final layers are the ones that look at the gathered image scraps to accurately identify the appropriate image clas.ls.
