# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import warnings
import tensorflow as tf
import numpy as np

warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import os
from utils.inference_utils import InputReader
from utils.network.my_model import Model
from utils.unet3d.utils3d import THE20_single_dict
from utils.network.pre_trained_embedding import Settings

def run_Design(preparation_config, esm_contents):

    #==================================Model===================================
    test_reader = InputReader(data_list=preparation_config["filenames"], 
                              preparation_config=preparation_config)

    #============================Parameters====================================
    params = {}
    params["n_1d_feat"] = 256
    params["n_2d_feat"] = 128
    params["n_str_layers"] = 8
    params["n_structure_msa_feat"] = 384
    params["max_relative_distance"] = 32
    params["evofomer_config"] = Settings.CONFIG
    params["n_cycle"] = 3
    params["save_path"] = "./utils/models"
    #============================Models====================================
    model_1 = Model(model_params=params, esm_contents=esm_contents)
    model_1(input_1d=np.zeros((1,10,17 + 20 + 15*15*15*5)), 
            input_2d=np.zeros((1,10,10,130)), 
            residue_index=np.array([range(10)]), 
            L=10)
    model_1.load_model(name="1.h5")

    model_2 = Model(model_params=params, esm_contents=esm_contents)
    model_2(input_1d=np.zeros((1,10,17 + 20 + 15*15*15*5)), 
            input_2d=np.zeros((1,10,10,130)), 
            residue_index=np.array([range(10)]), 
            L=10)
    model_2.load_model(name="2.h5")

    model_3 = Model(model_params=params, esm_contents=esm_contents)
    model_3(input_1d=np.zeros((1,10,17 + 20 + 15*15*15*5)), 
            input_2d=np.zeros((1,10,10,130)), 
            residue_index=np.array([range(10)]), 
            L=10)
    model_3.load_model(name="3.h5")
    
    models = [model_1, model_2, model_3]
    for step, filenames_batch in enumerate(test_reader.dataset):

        filenames, x, x_trr, L = \
            test_reader.read_file_from_disk(filenames_batch)
        
        residue_index = np.array([range(L)])
        
        label_predictions = []
        for model in models:
            if L > 512:
                n = L // 512 + 1
                label_predictions_ = []
                for i in range(n):
                    residue_index_ = residue_index[:,i*512:(i+1)*512]
                    L_ = residue_index_.shape[1]
                    label_prediction_ = model(x[:,i*512:(i+1)*512,:], x_trr[:,i*512:(i+1)*512,i*512:(i+1)*512,:], 
                                              residue_index_, L_, training=False) 
                    label_predictions_.append(label_prediction_)
                    
                label_prediction = tf.concat(label_predictions_, 0)
            else:
                label_prediction = model(x, x_trr,
                                         residue_index, L, training=False)   
            
            label_predictions.append(label_prediction)
    
        label_predictions = np.array(label_predictions)
        label_predictions = np.mean(label_predictions, 0)
        labels = np.argmax(label_predictions, -1)
        
        fasta = ""
        for i in labels:
            fasta += THE20_single_dict[i]
        
        filename = filenames[0]
        fw = open(os.path.join(preparation_config["output_path"], filename+".fasta"), 'w')
        fw.writelines(">" + filename + "\n")
        fw.writelines(fasta)
        fw.close()
    #==================================Model===================================
    
    
    