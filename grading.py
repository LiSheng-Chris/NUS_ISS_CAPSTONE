import os
import pickle
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import keras
from PIL import Image as pil_image

#from keras.applications.mobilenet import MobileNet, relu6, DepthwiseConv2D
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import DepthwiseConv2D
from keras.preprocessing import image
from tensorflow.keras.layers import AveragePooling2D, Conv2D, UpSampling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.resnet50 import preprocess_input

import itertools
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import tensorflow as tf

init_dim = 250
target_dim = 224
target_size = (target_dim, target_dim)
input_shape = (target_size[0], target_size[1], 3)
bs = 32
big_dim = 1024

# classes
class_labels = ['Benign', 'Gleason 3', 'Gleason 4', 'Gleason 5']
n_class = len(class_labels)
prefix = './'


def load_final_model(model_path):
    # load model
    model = load_model(model_path,custom_objects={'relu6': tf.keras.layers.ReLU(6.),'DepthwiseConv2D': DepthwiseConv2D})
    
    # Compute model predictions (pixel-level probability maps) 
    w_out, b_out = model.layers[-1].get_weights()
    w_out = w_out[np.newaxis,np.newaxis,:,:]
    # rescaling factor is 3
    base_model = MobileNet(include_top=False, weights=None,
                       input_shape=(big_dim, big_dim, 3),
                       alpha=.5, depth_multiplier=1, dropout=.2)
    block_name = 'conv_pw_13_relu'
    x_input = base_model.get_layer(block_name).output

    # average pooling instead of global pooling
    x = AveragePooling2D((7, 7), strides=(1,1), padding='same', name='avg_pool_top')(x_input)
    x = Conv2D(n_class, (1, 1), activation='softmax', padding='same')(x)
    x_out = UpSampling2D(size=(32, 32), name='upsample')(x)
    big_model = Model(base_model.input, x_out)
    big_model.load_weights(model_path, by_name=True)
    big_model.layers[-2].set_weights([w_out, b_out])
    return big_model
    
def assign_group(a, b, survival_groups=False):
    # if both cancer and benign tissue are predicted
    # ignore benign tissue for reporting, as pathologists do
    if (a > 0) and (b == 0):
        b = a
    if (b > 0) and (a == 0):
        a = b

    if not survival_groups:
        return a + b
    else:
        # get the actual Gleason pattern (range 3-5)
        a += 2
        b += 2
        if a+b <= 6:
            return 1
        elif a+b == 7:
            return 2
        else:
            return 3
            
    
def gleason_summary_wsum(y_pred, survival_groups=False, thres=None):
    gleason_scores = y_pred.copy()
    gleason_scores /= np.sum(gleason_scores)
    # remove outlier predictions
    if thres is not None:
        gleason_scores[gleason_scores < thres] = 0
    # and assign overall grade
    idx = np.argsort(gleason_scores)[::-1]
    primary_class = idx[0]
    secondary_class = idx[1] if gleason_scores[idx[1]] > 0 else idx[0]
    return assign_group(primary_class, secondary_class, survival_groups)
    
def pil_resize(img, target_size):
    hw_tuple = (target_size[1], target_size[0])
    if img.size != hw_tuple:
        img = img.resize(hw_tuple)
    return img

def pred(big_model,fname,image_file,seg_file):

    # get network predictions as heatmap
    img = image.load_img(image_file, grayscale=False, target_size=(big_dim, big_dim))
    X = image.img_to_array(img)
    X = preprocess_input(X)
    y_pred_prob = big_model.predict(X[np.newaxis,:,:,:], batch_size=1)[0]
    
    # get the (automatically generated) tissue mask
    tissue_mask = pil_image.open(seg_file)
    tissue_mask = np.array(pil_resize(tissue_mask, target_size=(big_dim, big_dim)))
    
    print(tissue_mask)

    # compute probability only at (predicted) tissue regions
    y_pred_prob[tissue_mask == n_class] = 0.
    y_pred_prob = y_pred_prob.reshape(-1, 4)
    w_sum = np.sum(y_pred_prob, axis=0)
    
    # Compute confusion matrices and Cohen's kappa statistic for Gleason score assignments on entire TMA spots
    csv_file = os.path.join(prefix,'dataset_TMA', 'tma_info', 'ZT80_gleason_scores.csv')
    df_patho = pd.read_csv(csv_file, sep='\t', index_col=0)

    
    # if the image was annotated by the pathologists
    a, b = df_patho.loc[fname][['patho1_class_primary', 'patho1_class_secondary']]
    x_gleason_annot_patho1 = assign_group(a, b)
    a, b = df_patho.loc[fname][['patho2_class_primary', 'patho2_class_secondary']]
    x_gleason_annot_patho2 = assign_group(a, b)
    x_gleason_cnn = gleason_summary_wsum(w_sum, thres=0.25)
    
    print('x_gleason_annot_patho1: %.2f' % x_gleason_annot_patho1)
    print('x_gleason_annot_patho2: %.2f' % x_gleason_annot_patho2)
    print('x_gleason_cnn: %.2f' % x_gleason_cnn)
    
    return str(x_gleason_cnn)


def run_pred(fname,seg_file):
    #model_weights = os.path.join(prefix,'model_weights/MobileNet_Gleason_weights.h5')
    #model_weights = os.path.join(prefix,'model_weights/best_model_weights.h5')
    #model_weights = os.path.join(prefix,'models/finetune_MobileNet_50/best_model_weights.h5')
    #model_weights = os.path.join(prefix,'models/finetune_VGG16/best_model_weights.h5')
    #model_weights = os.path.join(prefix,'models_aug/finetune_MobileNet_50/best_model_weights.h5')
    model_weights = os.path.join(prefix,'models_aug/finetune_VGG16/best_model_weights.h5')
    
    final_model=load_final_model(model_weights);
    #fname='ZT80_38_A_1_1'
    image_file = os.path.join(prefix,'dataset_TMA','TMA_images', fname+'.jpg')
    seg_file = os.path.join(prefix,'dataset_TMA','tissue_masks', 'mask_'+fname+'.png')
    #seg_file = os.path.join('./static/', 'mask_'+fname+'.png')
    print("mask:"+seg_file)
    return pred(final_model,fname,image_file,seg_file)