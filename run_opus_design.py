# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""
import sys
sys.path.append("/work/home/xugang/projects/esm/esm-main")

import os
import warnings
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
import numpy as np
import time
import multiprocessing

import esm

from utils.inference_utils import mk_input1d, mk_input2d, cleanPDB
from utils.unet3d import utils3d, unet
from utils.inference import run_Design

def preparation(multi_iter):
    
    file_path, filename, preparation_config = multi_iter
    
    file_path2 = os.path.join(preparation_config["tmp_files_path"], filename + ".pdb")
    cleanPDB(file_path, file_path2)
    
    input1d_filename = filename + '.input1d.npz'
    if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], input1d_filename)):
        mk_input1d(file_path2, filename, preparation_config)
        
    input2d_filename = filename + '.input2d.npz'
    if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], input2d_filename)):
        mk_input2d(file_path2, filename, preparation_config)

if __name__ == '__main__':

    #============================Parameters====================================
    list_path = r"./testset/list_cameo"
    native_path = "./testset/cameo"
    
    files_path = []
    f = open(list_path)
    for i in f.readlines():
        filename = i.strip()
        path = os.path.join(native_path, filename + "_UNK.pdb")
        files_path.append(path)
    f.close()
    print (len(files_path))
    
    preparation_config = {}
    preparation_config["batch_size"] = 1
    preparation_config["tmp_files_path"] = os.path.join(os.path.abspath('.'), "tmp_files")
    preparation_config["output_path"] = os.path.join(os.path.abspath('.'), "predictions")
    preparation_config["mkdssp_path"] = os.path.join(os.path.abspath('.'), "utils/mkdssp/xssp-3.0.10/mkdssp")
    
    num_cpu = 56
    
    #============================Parameters====================================
    
    
    #============================Preparation===================================
    print('Preparation start...')
    start_time = time.time()
    
    multi_iters = []
    filenames = []
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        multi_iters.append([file_path, filename, preparation_config])
        filenames.append(filename)
        
    pool = multiprocessing.Pool(num_cpu)
    pool.map(preparation, multi_iters)
    pool.close()
    pool.join()

    preparation_config["filenames"] = filenames
        
    run_time = time.time() - start_time
    print('Preparation done..., time: %3.3f' % (run_time))  
    #============================Preparation===================================
    
    #============================ESM2===================================
    # Load ESM-2 model
    esm_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    esm_model.eval()  # disables dropout for deterministic results
    
    esm_contents = {}
    esm_contents["esm_model"] = esm_model
    esm_contents["batch_converter"] = batch_converter
    esm_contents["alphabet"] = alphabet
    #============================ESM2===================================
    
    #============================3D-Unet===================================
    print('Run 3DCNN module in OPUS-Design...')
    start_time = time.time()
    
    model1 = utils3d.U3DModel()
    model1.model(x=np.zeros((1,40,40,40,27)))
    model1.load_model(name= r"./utils/unet3d/models/1.h5")
    
    model2 = utils3d.U3DModel()
    model2.model(x=np.zeros((1,40,40,40,27)))
    model2.load_model(name= r"./utils/unet3d/models/2.h5")
    
    model3 = utils3d.U3DModel()
    model3.model(x=np.zeros((1,40,40,40,27)))
    model3.load_model(name= r"./utils/unet3d/models/3.h5")
    
    model4 = utils3d.U3DModel()
    model4.model(x=np.zeros((1,40,40,40,27)))
    model4.load_model(name= r"./utils/unet3d/models/4.h5")
    
    model5 = utils3d.U3DModel()
    model5.model(x=np.zeros((1,40,40,40,27)))
    model5.load_model(name= r"./utils/unet3d/models/5.h5")

    models = [model1, model2, model3, model4, model5]
    for file_path in files_path:
        filename = file_path.split('/')[-1].split('.')[0]
        u3d_filename = filename + '.3dcnn.npz'
        if not os.path.exists(os.path.join(preparation_config["tmp_files_path"], u3d_filename)):
            seq_len = np.load((os.path.join(preparation_config["tmp_files_path"], filename + ".input1d.npz")))['f'].shape[0]
            u3d = unet.U3DEng(file_path, models=models)
            u3d.reconstruct_protein(seq_len=seq_len, 
                                    output_path=os.path.join(preparation_config["tmp_files_path"], filename + '.3dcnn'),
                                    output_path2=os.path.join(preparation_config["output_path"], filename + '_tmp.fasta'),
                                    esm_contents=esm_contents)  
            
    run_time = time.time() - start_time
    print('3DCNN module done..., time: %3.3f' % (run_time))  
    #============================3D-Unet===================================
        
    #============================OPUS-Design===============================
    print('Run feature aggregation module in OPUS-Design...')
    start_time = time.time()    
    run_Design(preparation_config, esm_contents)
    run_time = time.time() - start_time
    print('Feature aggregation module done..., time: %3.3f' % (run_time))      
    #============================OPUS-Design===============================
    
    print('OPUS-Design done...')
    