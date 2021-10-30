# Intel-Image-Classification



Intel Image Classification
Project Workflow:
1. Examine and understand data
2. Build an input pipeline
3. Build the model
4. Train the model
5. Test the model
6. Improve the model and repeat the process



Data Summary
The data contains around 25k images of Natural Scenes around the world.

All training and testing images with a size of 150x150 are classified into 6 categories:

.buildings = 0
.forest = 1
.glacier = 2
.mountains = 3
.sea = 4
.street = 5
.The data consists of 3 separated datasets:

Train with 14034 images
Test with 3000 images
Prediction with 7301 images
This data was originally published on https://datahack.analyticsvidhya.com by Intel for the



Import TensorFlow and Other Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Rescaling, RandomFlip, RandomRotation, RandomZoom,
                                    Dense, Flatten, Dropout, Conv2D, MaxPooling2D)
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.applications import VGG16


Load and Explore the Datasets



# Define some basic parameters

batch_size = 32
img_height = 150
img_width = 150



# Define the path to the datasets directories

train_path = '/kaggle/input/intel-image-classification/seg_train/seg_train/'
test_path = '/kaggle/input/intel-image-classification/seg_test/seg_test/'
pred_path = '/kaggle/input/intel-image-classification/seg_pred/seg_pred/'





# Define data loading function

def load_data(path, labels):
    dataset = image_dataset_from_directory(
        directory=path,
        labels=labels,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    return dataset
    
    
    
    
    
    # Load the datasets

train_ds = load_data(train_path, labels='inferred')
test_ds = load_data(test_path, labels='inferred')
pred_ds = load_data(pred_path, labels=None)


# Explore the image labels

class_names = train_ds.class_names
class_names

['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']



Visualize the Data



# Show the first nine images from the training dataset

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        
        
        
#Configure the Datasets for Performance¶
Two important methods should be used when loading data:

Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training the model. If the dataset is too large to fit into memory, this method also can be used to create a performant on-disk cache.
Dataset.prefetch overlaps data preprocessing and model execution while training.




AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
pred_ds = pred_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)



#Standardize the Data¶
The RGB channel values are in the [0, 255] range. This is not ideal for a neural network.

The values should be standardized to be in the [0, 1] range.



scaling = Rescaling(1. / 255)

train_ds = train_ds.map(lambda x, y: (scaling(x), y))
test_ds = test_ds.map(lambda x, y: (scaling(x), y))
pred_ds = pred_ds.map(lambda x: scaling(x))



Create Simple CNN Model (sCNN)¶
The Sequential model consists of three convolution blocks with a max pooling layer in each of them.

There's a fully-connected layer with 128 units on top of it that is activated by a ReLU activation function ('relu').

This model is a simple model and has not been tuned for high accuracy.



sCNN = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

#Compile the Model


The Compile method configures the model for training and validation using the optimizer, loss function, and evaluation metrics.

This workflow will use the Adam optimizer, the Sparse Categorical Crossentropy loss function, and the Accuracy evaluation metric.

sCNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
              
              
Model summary

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 150, 150, 16)      448       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 75, 75, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 75, 75, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 37, 37, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 37, 37, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 18, 18, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 20736)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               2654336   
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 774       
=================================================================
Total params: 2,678,694
Trainable params: 2,678,694
Non-trainable params: 0




#Train the Model
This workflow will train the model using 10 epochs, and test_ds as the validation data.



epochs = 10

history = sCNN.fit(train_ds, validation_data=test_ds, epochs=epochs)



2021-10-28 13:48:49.087809: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:228] Shuffle buffer filled.
439/439 [==============================] - 65s 36ms/step - loss: 0.9153 - accuracy: 0.6479 - val_loss: 0.6846 - val_accuracy: 0.7443
Epoch 2/10
439/439 [==============================] - 5s 12ms/step - loss: 0.5843 - accuracy: 0.7887 - val_loss: 0.6206 - val_accuracy: 0.7783
Epoch 3/10
439/439 [==============================] - 6s 13ms/step - loss: 0.4413 - accuracy: 0.8413 - val_loss: 0.5505 - val_accuracy: 0.8137
Epoch 4/10
439/439 [==============================] - 6s 13ms/step - loss: 0.3154 - accuracy: 0.8851 - val_loss: 0.5304 - val_accuracy: 0.8250
Epoch 5/10
439/439 [==============================] - 6s 13ms/step - loss: 0.2173 - accuracy: 0.9229 - val_loss: 0.6609 - val_accuracy: 0.7980
Epoch 6/10
439/439 [==============================] - 6s 13ms/step - loss: 0.1455 - accuracy: 0.9514 - val_loss: 0.7277 - val_accuracy: 0.8093
Epoch 7/10
439/439 [==============================] - 6s 13ms/step - loss: 0.1028 - accuracy: 0.9661 - val_loss: 0.8204 - val_accuracy: 0.8060
Epoch 8/10
439/439 [==============================] - 6s 13ms/step - loss: 0.0695 - accuracy: 0.9781 - val_loss: 0.8451 - val_accuracy: 0.8103
Epoch 9/10
439/439 [==============================] - 6s 13ms/step - loss: 0.0457 - accuracy: 0.9871 - val_loss: 1.0854 - val_accuracy: 0.7870
Epoch 10/10
439/439 [==============================] - 6s 13ms/step - loss: 0.0525 - accuracy: 0.9839 - val_loss: 1.0761 - val_accuracy: 0.7907





#Visualize Training Results
Create plots of loss and accuracy on the training and validation sets




def visualize_results(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower center')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper center')
    plt.title('Training and Validation Loss')
    plt.show();
    
    
    
    visualize_results(history, epochs)
    
    
    
#Evaluate the Model
Explore the loss and accuracy of validation data after training with 10 epochs.


sCNN_loss, sCNN_acc = sCNN.evaluate(test_ds, verbose=2)


94/94 - 1s - loss: 1.0761 - accuracy: 0.7907



Conclusion
In the plots above, the training accuracy is increasing linearly over time, whereas validation accuracy stalls around 80% in the training process. Also, the difference in accuracy between training and validation accuracy is noticeable — a sign of overfitting.
