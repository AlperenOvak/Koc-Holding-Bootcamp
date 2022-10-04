#!/usr/bin/env python
# coding: utf-8

# In[130]:


get_ipython().system('pip install librosa')


# In[131]:


get_ipython().system('dir')


# In[132]:


import IPython.display as ipd
import pandas as pd
import numpy as np
import os
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[153]:


filename='UrbanSound8K/audio/fold3/13230-0-0-1.wav'
plt.figure(figsize=(14,5))
scale, sr=librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)
ipd.Audio(filename)


# In[154]:


def create_spectrogram(scale, sr):
    spec=librosa.feature.melspectrogram(scale, sr=sr)
    spec.shape
    spec_conv=librosa.amplitude_to_db(spec,ref=np.max)
    return spec_conv


# In[157]:


a=create_spectrogram(scale, sr)
plt.imshow(a)
a


# In[162]:


import matplotlib.image

matplotlib.image.imsave('UrbanSound8K/name.png', a)


# In[136]:


get_ipython().system('pip3 install opencv-python')


# In[137]:


image = cv2.imread('UrbanSound8K/spectrograms/0/13230-0-0-1.png')
type(image)
image


# In[138]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
image


# In[139]:


img_arr=cv2.resize(image,(224,224))
img_arr


# In[140]:


folder_path="UrbanSound8K/audio/"


# In[141]:


metadata=pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.head()


# In[142]:


from tqdm import tqdm


# In[163]:


extracted_features=[]

for index_num,row in tqdm(metadata.iterrows()):
    audio_path = os.path.join(folder_path,'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    
    scale, sr=librosa.load(audio_path)
    image_spec=create_spectrogram(scale, sr)
    image_path = os.path.join(folder_path,'fold'+str(row["fold"])+'/',(str(row["slice_file_name"])).replace("wav","png"))
    matplotlib.image.imsave(image_path, image_spec)
    
    
    img_arr=cv2.imread(image_path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_arr=cv2.resize(img_arr,(224,224))

    
    extracted_features.append([img_arr,final_class_labels])


# In[164]:


### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[46]:


"""x_train=[]
for i in range (0,8):
    train_path=folder_path+str(i)
    for img in os.listdir(train_path):

        image_path=train_path+"/"+img

        img_arr=cv2.imread(image_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr=cv2.resize(img_arr,(224,224))

        x_train.append(img_arr)
        
        
x_train[10]"""
"""x_test=[]

for i in range (8,9):
    test_path=folder_path+str(i)
    for img in os.listdir(test_path):

        image_path=test_path+"/"+img

        img_arr=cv2.imread(image_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr=cv2.resize(img_arr,(224,224))

        x_test.append(img_arr)"""
"""x_val=[]

for i in range (9,10):
    val_path=folder_path+str(i)
    for img in os.listdir(val_path):

        image_path=val_path+"/"+img

        img_arr=cv2.imread(image_path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr=cv2.resize(img_arr,(224,224))

        x_val.append(img_arr)"""
"""train_x=np.array(x_train)
test_x=np.array(x_test)
val_x=np.array(x_val)"""
"""train_x=train_x/255.0
test_x=test_x/255.0
val_x=val_x/255.0"""


# In[235]:


### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[236]:


print(X[50].shape)


# In[237]:


X_train=X[:6987]
y_train=y[:6987]
X_val=X[6987:7860]
y_val=y[6987:7860]
X_test=X[7860:]
y_test=y[7860:]


# In[238]:


print(len(X_train),len(X_val),len(X_test))


# In[239]:


X_train=X_train/255
X_val=X_val/255
X_test=X_test/255


# In[240]:


# SECOND NOTEBOOK # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# In[241]:


get_ipython().system('pip install tensorflow')


# In[242]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[243]:


# Create a model object
model= tf.keras.Sequential()


# In[244]:


# Add a convolution and max pooling layer
model.add(tf.keras.layers.Conv2D(32,
                                 kernel_size=(5),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(224,224,1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))


# In[246]:


# Add more convolution and max pooling layers
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(5),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(224,224,1)))
model.add(tf.keras.layers.MaxPooling2D((2,2)))
model.add(tf.keras.layers.Conv2D(64,
                                 kernel_size=(5),
                                 strides=(1,1),
                                 padding="same",
                                 activation="relu",
                                 input_shape=(224,224,1)))


# In[247]:


# Flatten the convolution layer
model.add(tf.keras.layers.Flatten())


# In[252]:


# Add the dense layer and dropout layer
model.add(tf.keras.layers.Dense(244,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

# Add the dense layer and dropout layer
model.add(tf.keras.layers.Dense(224,activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))


# In[253]:


# Add the output layer
model.add(tf.keras.layers.Dense(10,activation="softmax"))


# In[254]:


# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[255]:


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




