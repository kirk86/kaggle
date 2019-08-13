
# coding: utf-8

# In[1]:

import numpy as np 
import pandas as pd 
import skimage, os
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os
import zarr

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th') 

from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, merge, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import SpatialDropout3D
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')


        
def print_mask(lung_m, nodule_m):
    fig, ax = plt.subplots(1,2, figsize=(20,16))
    ax[0].imshow(lung_m, cmap = plt.cm.bone)
    ax[1].imshow(nodule_m, cmap = plt.cm.bone)
    return
    
def get_max_slices(start, end):
    mask_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/lung_mask/'
    nodules_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/nodule_mask/'
    patients = os.listdir(mask_path)[start:end]
    max_slices = 0
    full_slices = 0
    for i in range(len(patients)):
        num_slices = np.load(nodules_path + patients[i]).astype('float16').shape[0]
        full_slices += num_slices
        if num_slices > max_slices:
            max_slices = num_slices
    print('Number of max slices in CT image: {}'.format(max_slices))
    print('Number of 2D slices in CT image: {}'.format(full_slices))
    return max_slices, full_slices


# In[3]:

def load_3d_data(start, end, size = 168, size_3d = 128, normalize = False):
    
    mask_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/lung_mask/'.format(size)
    nodules_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/nodule_mask/'.format(size)
    patients = sorted([x for x in os.listdir(mask_path) if '.npy' in x])[start:end]
    print('Loading {} patients,'.format(len(patients)), 'Start: {}, end: {}'.format(start, end))
    if normalize:
        masks = np.full((len(patients), 1, size_3d, size, size), 0.019607).astype('float32')
        nodules = np.zeros((len(patients), 1, size_3d, size, size)).astype('float32')
    else:
        masks = np.full((len(patients), 1, size_3d, size, size), threshold_min).astype('float32')
        nodules = np.zeros((len(patients), 1, size_3d, size, size)).astype('float32')
        
    for i in range(len(patients)):
        mask = np.load(mask_path + patients[i]).astype('float32')
        mask = mask.swapaxes(1, 0)
        nod = np.load(nodules_path + patients[i]).astype('float32')
        nod = nod.swapaxes(1, 0)
        num_slices = mask.shape[1]
        offset = (size_3d - num_slices)
        if offset == 0:
            masks[i, :, :, :, :] = mask[:, :, :, :]
            nodules[i, :, :, :, :] = nod[:, :, :, :]
        if offset > 0:
            begin_offset = int(np.round(offset/2))
            end_offset = int(offset - begin_offset)
            masks[i, :, begin_offset:-end_offset, :, :] = mask[:, :, :, :]
            nodules[i, :, begin_offset:-end_offset, :, :] = nod[:, :, :, :]
        if offset < 0:
            offset = -(size_3d - num_slices)
            begin_offset = int(np.round(offset/2))
            end_offset = int(offset - begin_offset)
            masks[i, :, :, :, :] = mask[:, begin_offset:-end_offset, :, :]
            nodules[i, :, :, :, :] = nod[:, begin_offset:-end_offset, :, :]

    return masks, nodules

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# In[2]:


threshold_min = -2000
smooth = 1.0
end = 500

#max_slices, full_slices = get_max_slices(0, end)
max_slices = 136
width = 64
img_size = 168


img_rows = img_size
img_cols = img_size


# In[3]:

def model_load(name):
    check_model = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(name)
    model = load_model(check_model, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    return model


# In[4]:

model = model_load('3D_UNet_raw_800pats_gpu1')


# In[9]:

ms, nds = load_3d_data(start = 500, end = 525, size = 168, size_3d = 128, normalize = False)


# In[10]:

preds = model.predict(ms, batch_size = 2)


# In[13]:

preds[21][0]


# In[14]:

preds2 = preds.copy()
preds2[preds2 >= 0.01] = 1


# In[15]:

def check_pred(index1, index2):
    print_mask(ms[index1][0][index2], preds2[index1][0][index2])
    return

check_pred(5, 127)


# In[16]:

patient_max = {}
for i in range(len(preds)):
    val_max = {}
    zero = 1e-5
    for j in range(preds[i].shape[1]):
        current_max = np.max(preds[i, 0, j, :, :])
        if current_max > zero:
            val_max[j] = current_max
    patient_max[i] = val_max


# In[17]:

patient_max


# In[ ]:



