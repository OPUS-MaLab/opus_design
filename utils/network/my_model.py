# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:41:38 2020

@author: xugang

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch

from utils.network.my_layer import TrackableLayer

import utils.network.pre_trained_embedding.model.EmbeddingModel as Pre_MSA_emb
import utils.network.pre_trained_embedding.model.EvoFormer as EvoFormer
from utils.network.my_ipa import StructureModule
from utils.unet3d.utils3d import THE20_single_dict
from utils.unet3d.unet import tranferSM

class AFStructureModule(TrackableLayer):
    def __init__(self, model_params, name='structure_module'):
        super(AFStructureModule, self).__init__()
        self.structure_layer = StructureModule(name=name,
                                               config=model_params,
                                               n_att_head=12,
                                               dropout=0.1,
                                               r_ff=2)
    def call(self, msa_3d, pair, training=True):
        return self.structure_layer(msa_3d, pair, training=training)
    
class AAEmbedding(TrackableLayer):
    def __init__(self, config):
        super(AAEmbedding, self).__init__()
        self.config = config
        self.emb = Pre_MSA_emb.Embedding(self.config)

    def call(self, inp_1d, residue_index):
        return self.emb(inp_1d, residue_index)

class AFEvoformerEnsemble(TrackableLayer):
    def __init__(self, config, name_layer, iter_layer, name='evoformer_iteration', iters=None):
        super(AFEvoformerEnsemble, self).__init__(name=name_layer+"_"+str(iter_layer))
        self.config = config
        self.evo_iterations = []
        for i in range(len(iters)):
            global_config = {'iter': iters[i]}

            self.evo_iteration = EvoFormer.Evoformer(config, name=name, global_config=global_config)
            self.evo_iterations.append(self.evo_iteration)

    def call(self, msa, pair, training=True):
        for i in range(len(self.evo_iterations)):
            msa, pair = self.evo_iterations[i](msa, pair, training=training)
        return msa, pair

class Recycle(keras.layers.Layer):
    def __init__(self):
        super(Recycle, self).__init__()
        self.prev_msa_norm = keras.layers.LayerNormalization(name='prev_msa_first_row_norm')
        self.prev_pair_norm = keras.layers.LayerNormalization(name='prev_pair_norm')
        
        self.last_label_encoding = keras.layers.Dense(256, name='last_label_linear')
        self.prev_label_norm = keras.layers.LayerNormalization(name='prev_label_norm')

    def call(self, msa, pair, label_last):
        # msa (1, L, 256), pair (L, L, 128), label_last (1, L, 20)
        msa = tf.stop_gradient(msa)
        pair = tf.stop_gradient(pair)
        label_last = tf.stop_gradient(label_last)

        pair_last = self.prev_pair_norm(pair)
        msa_last = self.prev_msa_norm(msa) + self.prev_label_norm(self.last_label_encoding(label_last))
        
        return msa_last, pair_last

class TRREmbedding(keras.layers.Layer):
    def __init__(self, name, n_feat):
        super(TRREmbedding, self).__init__()
        self.trr_emb = keras.layers.Conv2D(name=name, filters=n_feat, kernel_size=1, padding='SAME')
        self.norm = keras.layers.LayerNormalization()
        
    def call(self, feat):
        return self.norm(self.trr_emb(feat))

class CNN3dEmbedding(keras.layers.Layer):
    def __init__(self, num_layers, rate):
        super(CNN3dEmbedding, self).__init__()
        self.cov3d_1 = keras.layers.Conv3D(16, 3, strides = 1, padding = 'valid', activation = 'relu', name="3dcnn_1")
        self.cov3d_2 = keras.layers.Conv3D(1, 3, strides = 1, padding = 'valid', activation = 'relu', name="3dcnn_2")
        self.dnn = keras.layers.Dense(num_layers)
        self.dropout = keras.layers.Dropout(rate)
        self.norm = keras.layers.LayerNormalization()
        
    def call(self, x, training):
        length = tf.shape(x)[1]
        x = tf.reshape(x, (1, length, 15, 15, 15, 5))
        x = tf.squeeze(x, 0)
        x = self.cov3d_1(x)
        x = self.cov3d_2(x)
        x = tf.reshape(x, (length, 11*11*11))
        x = tf.expand_dims(x, 0)
        x = self.dnn(x)
        x = self.norm(x)
        x = self.dropout(x, training=training)
        return x 

class Model(keras.Model):

    def __init__(self, model_params, esm_contents):
        super(Model, self).__init__()
        
        self.model_params = model_params
        self.esm_contents = esm_contents
        #=========================== Embedding ===========================
        self.pre_msa_emb = AAEmbedding(self.model_params)

        self.evoformers = []
        for i in range(4):
            self.evoformers.append(AFEvoformerEnsemble(self.model_params["evofomer_config"]["evoformer"],
                                                       name_layer='evoformer_ensemble',
                                                       iter_layer=i,
                                                       iters=[4*i, 4*i+1, 4*i+2, 4*i+3]))

        self.trr_emb = TRREmbedding(name="trr_feat", n_feat=self.model_params["n_2d_feat"])
        self.cnn3d_emb = CNN3dEmbedding(num_layers=512, rate=0.5)
        self.structure_layer = AFStructureModule(model_params=model_params)
        
        #=========================== Output ===========================
        self.recycle = Recycle()
        self.label_output = keras.layers.Dense(20)
        
    def call(self, input_1d, input_2d, residue_index, L, training=False):
        
        msa_last = tf.zeros((1, L, self.model_params["n_1d_feat"]))
        pair_last = tf.zeros((L, L, self.model_params["n_2d_feat"]))
        pred_label_last = np.zeros((1, L, 20))
        msa_last, pair_last = self.recycle(msa_last, pair_last, pred_label_last)  # msa_last (1, L, 256), pair_last (L, L, 128)
        
        f_seq = input_1d[:,:,:37]
        assert f_seq.shape == (1, L, 37)
        
        f_3d_cnn = input_1d[:,:,37:]
        assert f_3d_cnn.shape == (1, L, 15*15*15*5)
        f_3d_cnn = self.cnn3d_emb(f_3d_cnn, training=training)
        assert f_3d_cnn.shape == (1, L, 512)
        
        f_2d_trr = self.trr_emb(input_2d)[0]
        assert f_2d_trr.shape == (L, L, 128)
        
        CYCLE = self.model_params["n_cycle"]
        
        inp_1d = tf.concat([f_seq, f_3d_cnn], -1)
        assert inp_1d.shape == (1, L, 37 + 512)

        label_logits = []
        for c in range(CYCLE):
            f_1d, f_2d = self.pre_msa_emb(inp_1d, residue_index) # (1, L, 256) (L, L, 128)
            
            # Inject trr130 feature
            f_2d += f_2d_trr
            
            # Inject previous outputs for recycling
            f_2d += pair_last
            f_1d += msa_last

            for i in range(len(self.evoformers)):
                f_1d, f_2d = self.evoformers[i](f_1d, f_2d, training=training)

            f_2d_ = tf.expand_dims(f_2d, axis=0)
            f_1d_ = self.structure_layer(f_1d, f_2d_, training=training)
                
            label_logit = self.label_output(f_1d_) # (1, L, 20)
            label_logits.append(label_logit)
            pred_label_last = tf.squeeze(tf.nn.softmax(label_logit, -1), 0)
 
            fasta = ""
            for i in pred_label_last:
                prob = np.max(i, -1)
                resid = np.argmax(i, -1)
                if prob >= 0.5:
                    fasta += THE20_single_dict[resid]      
                else:
                    fasta += '<mask>'
            data = [
                ("tmp", fasta),
            ]
            batch_labels, batch_strs, batch_tokens = self.esm_contents["batch_converter"](data)
            with torch.no_grad():
                results = self.esm_contents["esm_model"](batch_tokens, repr_layers=[33], return_contacts=False)
            pred_label_last_sm = torch.nn.functional.softmax(results['logits'][0][1:-1], -1).numpy()
            pred_label_last_sm = tranferSM(pred_label_last_sm)
            
            pred_label_last2 = []
            assert pred_label_last.shape == (L, 20)
            assert pred_label_last_sm.shape == (L, 20)
            for i in range(L):
                if np.max(pred_label_last[i], -1) >= 0.5:
                    pred_label_last2.append(pred_label_last[i])
                elif np.max(pred_label_last_sm[i], -1) >= 0.3:
                    pred_label_last2.append(pred_label_last_sm[i])
                else:
                    pred_label_last2.append(pred_label_last[i])              
            pred_label_last = np.array(pred_label_last2)
            assert pred_label_last.shape == (L, 20)
            
            if c != CYCLE - 1:
                pred_label_last = tf.expand_dims(pred_label_last, axis=0)
                msa_last, pair_last = self.recycle(f_1d, f_2d, pred_label_last)
                
        return pred_label_last
 
    def load_model(self, name):
        print ("load model:", os.path.join(self.model_params["save_path"], name))
        self.load_weights(os.path.join(self.model_params["save_path"], name))
                
