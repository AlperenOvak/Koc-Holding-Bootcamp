#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa')


# In[2]:


get_ipython().system('dir')


# In[3]:


import IPython.display as ipd
import pandas as pd
import numpy as np
import os
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


def create_spectrogram(y):
    spec=librosa.feature.melspectrogram(y=y)
    spec.shape
    spec_conv=librosa.amplitude_to_db(spec,ref=np.max)
    return spec_conv


# In[ ]:


a=create_spectrogram(scale)
plt.imshow(a)
a


# In[5]:


import matplotlib.image


# In[6]:


get_ipython().system('pip3 install opencv-python')


# In[7]:


image = cv2.imread('UrbanSound8K/spectrograms/0/13230-0-0-1.png')
type(image)
image


# In[8]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
image


# In[9]:


img_arr=cv2.resize(image,(224,224))
img_arr


# In[10]:


folder_path="UrbanSound8K/audio/"
new_image_path="Lite/Images/"


# In[11]:


metadata=pd.read_csv('Lite/UrbanSound8K.csv')
metadata.head()


# In[12]:


from tqdm import tqdm


# In[51]:


extracted_features=[]
X_array=[]
Y_array=[]
for index_num,row in tqdm(metadata.iterrows()):
    #audio_path = os.path.join(folder_path,'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["classID"]
    
    
    ## We have already saved those images, thus we don't have to do it this part again
    #scale, sr=librosa.load(audio_path)
    #image_spec=create_spectrogram(scale)
    #(DRIVE)image_path = os.path.join(drive_path+"Images/",'Fold'+str(row["fold"])+'/',(str(row["slice_file_name"])).replace("wav","png"))
    image_path = os.path.join(new_image_path,'fold'+str(row["fold"])+'/',(str(row["slice_file_name"])).replace("wav","png"))
    #matplotlib.image.imsave(image_path, image_spec)
    img_arr=cv2.imread(image_path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    img_arr=cv2.resize(img_arr,(224,224))

    
    extracted_features.append([img_arr,final_class_labels])
    X_array.append([img_arr])
    Y_array.append([final_class_labels])


# In[53]:


Y_array


# In[54]:


### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','classID'])
extracted_features_df.head()


# In[55]:


### attempt_1: distrubte the data first

"""
x_train=[]
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


# In[61]:


np.save('file_name', extracted_features)

