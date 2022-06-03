#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle,h5py,os,keras,json, os, argparse, imp
import tensorflow as tf
import numpy as np


# In[3]:


get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
import utils as ut


# In[4]:


# intializations 
path_to_data = 'D:/RPI_big/zbt/Reseach/Siamese learning/activity_recog'# Data path of the created dictionary
filename = 'All_activity_instances_data.h5'


# In[50]:


# Loading the osurce and target classes
total_Activities = 33
pickle_out = open('source_activities.pkl',"rb")
src_act_array = pickle.load(pickle_out)
trg_act_array = np.arange(total_Activities)[~np.isin(np.arange(total_Activities),src_act_array)]


# Loading the target supp and test indices

# In[6]:


pickle_out = open('Indices_trg_Test_supp.pkl',"rb")
temp = pickle.load(pickle_out) # indices of validation and test set
pickle_out.close()
indices_array_test,indices_array_supp,new_labels_test,new_labels_supp = temp[0],temp[1],temp[2],temp[3]


# Loading the source indices

# In[16]:


pickle_out = open('Indices_Train_Val.pkl',"rb")
temp = pickle.load(pickle_out) # indices of validation and test set
pickle_out.close()
# indices_array_train = temp[0]
indices_array_Val = temp[1]
# new_labels_train = temp[2]
new_labels_Val = temp[3]


# Making the validation data for source class for sample-wise and class-wise stuff later

# In[42]:


data_file = h5py.File(os.path.join(path_to_data,filename), 'r') 
X_data = data_file['X']
X_Val = np.zeros((indices_array_Val.shape[0],X_data.shape[1],X_data.shape[2]))
for i in range(indices_array_Val.shape[0]):
    X_Val[i,:,:] = X_data[indices_array_Val[i],:,:]
y_Val = ut.convert_one_hot(new_labels_Val)


# Getting the target and support set

# In[43]:


X_supp = np.zeros((indices_array_supp.shape[0],X_data.shape[1],X_data.shape[2]))

for i in range(indices_array_supp.shape[0]):
    X_supp[i,:,:] = X_data[indices_array_supp[i],:,:]
y_supp = ut.convert_one_hot(new_labels_supp)


#  Model loading 

# In[7]:


from keras.models import model_from_json
from keras.backend.tensorflow_backend import set_session


# In[18]:



config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)



# In[65]:


modelPath = "./models/"
modelName = "Stacked_LSTM_classifier"
with open(os.path.join(modelPath, modelName+'.json'), 'r') as f:
    source_model = keras.models.model_from_json(f.read())


# In[66]:


source_model.load_weights(os.path.join(modelPath, modelName+'2.wgt'))
ad =keras.optimizers.Adam(lr = 0.01)
source_model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=ad, metrics=['acc'])


# In[67]:


# Checking if the laoded weights actually work on the validation set 
source_model.evaluate(X_Val,y_Val)


# Sample-wise and Classwise Relavence

# In[68]:


num_features = source_model.layers[-3].predict(X_Val[0:1,:,:]).shape[1]
samples_trg = indices_array_supp.shape[0]
samples_src = X_Val.shape[0]


# In[69]:


features_src = source_model.layers[-3].predict(X_Val)
features_trg = source_model.layers[-3].predict(X_supp)
f_src_tilda = features_src/(np.linalg.norm(features_src,2,axis = 1))[:,None]
f_trg_tilda = features_trg/(np.linalg.norm(features_trg,2,axis = 1))[:,None]


# In[70]:


# sample-wise
A = np.zeros((samples_src,samples_trg))
for i in range(samples_src):
    for j in range(samples_trg):
        A[i,j] = np.exp(np.dot(f_src_tilda[i,:],f_trg_tilda[j,:]))


# In[71]:


# Classwise
Classwise_weights = ut.classwise_weights(src_act_array,trg_act_array,new_labels_Val,new_labels_supp,A)


# In[72]:


## Normalization
# W = ut.soft_normalization(Classwise_weights)
W = ut.hard_normalization(Classwise_weights)


# In[74]:


source_model.layers[-2].get_weights()[0].shape


# Setting up the knowledge transfer model

# In[78]:


pretrained = keras.Model(
    source_model.inputs, source_model.layers[-2].output, name="pretrained_model"
)


# In[84]:


pretrained.summary()


# In[90]:


# from keras.models import Sequential
# model = Sequential()
# model.add()
merged = keras.layers.Concatenate([merged,keras.layers.Activation("softmax")])
W_trg = keras.layers.Dense(W.shape[1],use_bias= False)(source_model.layers[-2].output)
final_out = keras.layers.Activation('softmax')(W_trg)
Final_model = keras.Model(inputs = source_model.inputs,outputs = final_out)


# In[91]:


Final_model.layers[-2].set_weights([W])


# In[92]:


Final_model.summary()


# In[99]:


ad =keras.optimizers.Adam(lr = 0.01)
Final_model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=ad, metrics=['acc'])


# In[130]:


# Setting up the number of shots
number_shots = 5
# if(shots == 1):
unqiue_labels_supp = np.unique(new_labels_supp)
final_indices = np.zeros(unqiue_labels_supp.shape[0]*number_shots)
count = 0
for item in unqiue_labels_supp:
    indices = np.where(np.array(new_labels_supp) == item)
    index = np.random.choice(indices[0],size = number_shots,replace = False)
    final_indices[count:count+number_shots] = index[0:]
    count = count+number_shots
#     print(indices, index)


# In[100]:


Final_model.fit(X_supp[final_indices,:,:],y_supp[final_indices,:],batch_size = 64,epochs=5, verbose=1,
                    shuffle = "batch",validation_data = (X_supp[final_indices,:,:],y_supp[final_indices,:]))


# In[101]:


X_test = np.zeros((indices_array_test.shape[0],X_data.shape[1],X_data.shape[2]))

for i in range(indices_array_test.shape[0]):
    X_test[i,:,:] = X_data[indices_array_test[i],:,:]
y_test = ut.convert_one_hot(new_labels_test)


# In[102]:


Final_model.evaluate(X_test,y_test)


# In[ ]:




