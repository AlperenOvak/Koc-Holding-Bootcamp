#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install librosa')


# In[ ]:


get_ipython().system('dir')


# In[ ]:


import IPython.display as ipd
import pandas as pd
import numpy as np
import os
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image
import cv2


# In[ ]:


get_ipython().system('pip3 install opencv-python')


# In[ ]:


extracted_features_df = np.load('file_name.npy', allow_pickle=True)


# In[ ]:


extracted_features_df=pd.DataFrame(extracted_features_df,columns=['feature','classID'])
extracted_features_df.head()


# In[ ]:


### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['classID'].tolist())


# In[ ]:


print(X[50].shape)


# In[ ]:


### attempt_2: distrubte the data later
X_train=X[:6987]
y_train=y[:6987]
X_val=X[6987:7860]
y_val=y[6987:7860]
X_test=X[7860:]
y_test=y[7860:]


# In[ ]:


X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)


# In[ ]:


print(len(X_train),len(X_val),len(X_test))


# In[ ]:


X_train=X_train/255
X_val=X_val/255
X_test=X_test/255


# In[ ]:


get_ipython().system('pip install tensorflow')


# In[ ]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


##


# In[ ]:


# Create a model object
model= tf.keras.Sequential()


# In[ ]:


# Add a convolution and max pooling layer
model.add(tf.keras.layers.Conv2D(32,
                                 kernel_size=(5),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(224,224,1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))


# In[ ]:


# Add more convolution and max pooling layers
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(5),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(5),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu"))


# In[ ]:


# Flatten the convolution layer
model.add(tf.keras.layers.Flatten())


# In[ ]:


# Add the dense layer and dropout layer
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

# Add the dense layer and dropout layer
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))


# In[ ]:


# Add the output layer
model.add(tf.keras.layers.Dense(10,activation="softmax"))


# In[ ]:


# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[ ]:


# Train the model for 50 epochs with batch size of 128r
results=model.fit(X_train,y_train,
                  batch_size=128,
                  epochs=50,
                  validation_data=(X_val,y_val))


# In[ ]:


# Plot the the training loss
plt.plot(results.history["loss"],label="loss")

# Plot the the validation loss

plt.plot(results.history["val_loss"],label="val_loss")
# Name the x and y axises
plt.xlabel("Epoch")
plt.ylabel("Loss")


# Put legend table
plt.legend()

# Show the plot
plt.show()


# In[ ]:


# Plot the the training loss
plt.plot(results.history["accuracy"],label="accuracy")

# Plot the the validation loss

plt.plot(results.history["val_accuracy"],label="val_accuracy")
# Name the x and y axises
plt.xlabel("Epoch")
plt.ylabel("Accuracy")


# Put legend table
plt.legend()

# Show the plot
plt.show()


# In[ ]:


model.evaluate(X_test,y_test)

