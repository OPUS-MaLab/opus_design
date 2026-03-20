##################################################################################
# This code is a part from DLPacker.

# @article {Misiura2021.05.23.445347,
#     author = {Misiura, Mikita and Shroff, Raghav and Thyer, Ross and Kolomeisky, Anatoly},
#     title = {DLPacker: Deep Learning for Prediction of Amino Acid Side Chain Conformations in Proteins},
#     elocation-id = {2021.05.23.445347},
#     year = {2021},
#     doi = {10.1101/2021.05.23.445347},
#     publisher = {Cold Spring Harbor Laboratory},
#     URL = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347},
#     eprint = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347.full.pdf},
#     journal = {bioRxiv}
#
#If you find use, please cite it.                                         
##################################################################################

import re
import numpy as np

import tensorflow as tf
from tensorflow import keras

from collections import defaultdict

THE20 = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5,\
         'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11,\
         'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16,\
         'TRP': 17, 'TYR': 18, 'VAL': 19}

THE20_dict = {}
for k in THE20:
    THE20_dict[THE20[k]] = k
    
THE20_single = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5,\
                'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11,\
                'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16,\
                'W': 17, 'Y': 18, 'V': 19}

THE20_single_dict = {}
for k in THE20_single:
    THE20_single_dict[THE20_single[k]] = k

THE20_a2aaa = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',\
               'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',\
               'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR',\
               'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

THE20_aaa2a = {}
for k in THE20_a2aaa:
    THE20_aaa2a[THE20_a2aaa[k]] = k
    
BB_ATOMS = ['C', 'CA', 'N', 'O']
BOX_SIZE = 10
GRID_SIZE = 40
SIGMA = 0.65

class ResIdentitylBlock(keras.layers.Layer):
    
    def __init__(self, f1, f2):
        super(ResIdentitylBlock, self).__init__()

        self.conv1 = keras.layers.Conv3D(f1, 3, strides = 1, padding = 'same', activation = 'relu')
        self.conv2 = keras.layers.Conv3D(f1, 1, strides = 1, padding = 'same',  activation = 'relu')
        self.conv3 = keras.layers.Conv3D(f2, 3, strides = 1, padding = 'same',  activation = 'relu')
    
    def call(self, x, training=False):
        
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + x_in
        return x

class MLP(keras.layers.Layer):
    
    def __init__(self):
        super(MLP, self).__init__()
        
        self.conv_1 = keras.layers.Conv3D(16, 3, strides = 2, padding = 'same', activation = 'relu')
        
        self.conv_2_label = keras.layers.Conv3D(1, 3, strides = 1, padding = 'same', activation = 'relu')
        self.dropout_label = keras.layers.Dropout(0.25)
        self.dnn_label = keras.layers.Dense(256, activation = 'relu')
        self.dropout_label2 = keras.layers.Dropout(0.25)
        self.dnn_label2 = keras.layers.Dense(20)
        
    def call(self, x, training):
        x = x[:, 5:-5, 5:-5, 5:-5, :]
        x = self.conv_1(x)

        label_feat = self.conv_2_label(x)
        label_feat = tf.reshape(label_feat, (-1, 15*15*15))
        label_feat = self.dropout_label(label_feat, training=training)
        label_feat = self.dnn_label(label_feat)
        label_feat = self.dropout_label2(label_feat, training=training)
        label_feat = self.dnn_label2(label_feat)       

        return label_feat

class UNETR_PP(tf.keras.Model):
    
    def __init__(self, depths=3, nres=8, width=128, grid_size=40):
        super(UNETR_PP, self).__init__()
        
        self.grid_size = grid_size
        
        self.conv_l1 = keras.layers.Conv3D(1 * width, 3, padding = 'same', strides = 2, activation = 'relu')
        self.blocks1 = []
        for _ in range(nres):
            self.blocks1.append(ResIdentitylBlock(1 * width, 1 * width))
            
        self.conv_l2 = keras.layers.Conv3D(2 * width, 3, padding = 'same', strides = 2, activation = 'relu')
        self.blocks2 = []
        for _ in range(nres):
            self.blocks2.append(ResIdentitylBlock(1 * width, 2 * width))
            
        self.conv_l3 = keras.layers.Conv3D(4 * width, 3, padding = 'same', strides = 2, activation = 'relu')
        self.blocks3 = []
        for _ in range(nres):
            self.blocks3.append(ResIdentitylBlock(2 * width, 4 * width))

        self.upsamp_1 = keras.layers.UpSampling3D(size = 2)
        self.conv_l4 = keras.layers.Conv3D(4 * width, 3, padding = 'same')

        self.upsamp_2 = keras.layers.UpSampling3D(size = 2)
        self.conv_l5 = keras.layers.Conv3D(2 * width, 3, padding = 'same')

        self.upsamp_3 = keras.layers.UpSampling3D(size = 2)
        self.conv_l6 = keras.layers.Conv3D(1 * width, 3, padding = 'same')
        
        self.blocks4 = []
        for _ in range(2):
            self.blocks4.append(ResIdentitylBlock(64, 1 * width))
        
        self.conv_l6_1 = keras.layers.Conv3D(4, 3, padding = 'same')
        self.conv_l6_2 = keras.layers.Conv3D(2, 3, padding = 'same')
        
        self.dihedral = MLP()

        self.ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    def call(self, x, training=False):
        l0 = x # (8)
        
        l1 = self.conv_l1(l0) # (4)
        for res_identity in self.blocks1:
            l1 = res_identity(l1, training=training)
            
        l2 = self.conv_l2(l1) # (2)
        for res_identity in self.blocks2:
            l2 = res_identity(l2, training=training)
            
        l3 = self.conv_l3(l2) # (2)
        for res_identity in self.blocks3:
            l3 = res_identity(l3, training=training)

        l = self.upsamp_1(l3)
        l = tf.concat((l, l2), axis=-1) # (2)
        l = self.conv_l4(l)
        l = tf.nn.leaky_relu(l, alpha=0.2)
        
        l = self.upsamp_2(l)
        l = tf.concat((l, l1), axis=-1) # (4)
        l = self.conv_l5(l)
        l = tf.nn.leaky_relu(l, alpha=0.2)
       
        l = self.upsamp_3(l)
        l = tf.concat((l, l0), axis=-1)  # (8)
        l = self.conv_l6(l)
        l = tf.nn.leaky_relu(l, alpha=0.2)
        
        for res_identity in self.blocks4:
            l = res_identity(l, training=training)
        
        l1 = self.conv_l6_1(l)
        l2 = self.conv_l6_2(l)
        
        logit_label = self.dihedral(tf.concat((x[..., :4], l1, l2, l), axis=-1), training=training)
        sm_label = tf.nn.softmax(logit_label, -1)
            
        return sm_label.numpy()[0], l1.numpy()[0]

class U3DModel():
    def __init__(self, ):
        self.model = UNETR_PP()
        
    def load_model(self, name=None):
        print ("load_weights", name)
        self.model.load_weights(name)

class InputBoxReader():
    def __init__(self, charges_filename:str = './utils/unet3d/charges.rtp'):
        
        self.grid_size = GRID_SIZE
        self.grid_spacing = BOX_SIZE * 2 / GRID_SIZE
        self.offset = 10 * GRID_SIZE // 40
        self.total_size = GRID_SIZE + 2 * self.offset

        size = round(SIGMA * 4)
        self.grid = np.mgrid[-size:size+self.grid_spacing:self.grid_spacing,\
                             -size:size+self.grid_spacing:self.grid_spacing,\
                             -size:size+self.grid_spacing:self.grid_spacing]

        kernel = np.exp(-np.sum(self.grid * self.grid, axis = 0) / SIGMA**2 / 2) 
        kernel /= (np.sqrt(2 * np.pi) * SIGMA)
        self.kernel = kernel[1:-1, 1:-1, 1:-1]
        self.norm = np.sum(self.kernel)
        
        self.charges = defaultdict(lambda: 0)
        with open(charges_filename, 'r') as f:
            for line in f:
                if line[0] == '[' or line[0] == ' ':
                    if re.match('\A\[ .{1,3} \]\Z', line[:-1]):
                        key = re.match('\A\[ (.{1,3}) \]\Z', line[:-1])[1]
                        self.charges[key] = defaultdict(lambda: 0)
                    else:
                        l = re.split(r' +', line[:-1])
                        self.charges[key][l[1]] = float(l[3])
    
    def __call__(self, box):
        amino_acids = set()
        for i, j in zip(box['segids'], box['resids']):
            amino_acids.add(i+"_"+str(j))
        length = len(amino_acids)
        
        target_seg_res_id = box['target']['segid']+"_"+str(box['target']['id'])
        amino_acids.remove(target_seg_res_id)
        assert length == len(amino_acids) + 1
        
        x = np.zeros([self.total_size, self.total_size, self.total_size, 27])
        
        centers = (np.array(box['positions']) + BOX_SIZE) / self.grid_spacing
        centers += self.offset
        cr = np.round(centers).astype(np.int32)
        offsets = cr - centers
        offsets = offsets[:, :, None, None, None]
        
        i0 = self.kernel.shape[0] // 2
        i1 = self.kernel.shape[0] - i0
        
        for ind, a in enumerate(box['types']):
            seg_res_id = box['segids'][ind] + "_" + str(box['resids'][ind])
            
            dist = self.grid + offsets[ind] * self.grid_spacing
            kernel = np.exp(-np.sum(dist * dist, axis = 0) / SIGMA**2 / 2)
            kernel = kernel[1:-1, 1:-1, 1:-1] * self.norm / np.sum(kernel)

            xa, xb = cr[ind][0]-i0, cr[ind][0]+i1
            ya, yb = cr[ind][1]-i0, cr[ind][1]+i1
            za, zb = cr[ind][2]-i0, cr[ind][2]+i1

            if a == 'C': ch = 0
            elif a == 'N': ch = 1
            elif a == 'O': ch = 2
            elif a == 'S': ch = 3
            else: ch = 4

            aa = box['resnames'][ind]
            an = box['names'][ind]

            if (an in BB_ATOMS and aa in THE20) or (not aa in THE20):
                x[xa:xb, ya:yb, za:zb, ch] += kernel
                if (aa in self.charges and seg_res_id in amino_acids):
                    charge = kernel * self.charges[aa][an]
                    x[xa:xb, ya:yb, za:zb, 5] += kernel * self.charges[aa][an]
                else:
                    charge = kernel * self.charges['RST'][an[:1]]
                    x[xa:xb, ya:yb, za:zb, 5] += charge

            if aa in THE20:
                if an in BB_ATOMS:
                    if seg_res_id in amino_acids:
                        x[xa:xb, ya:yb, za:zb, 6 + THE20[aa]] += kernel
                    else:
                        x[xa:xb, ya:yb, za:zb, 6 + 20] += kernel
            else:
                x[xa:xb, ya:yb, za:zb, 6 + 20] += kernel
                
                        
        b = self.offset
        x = x[b:-b, b:-b, b:-b, :]
        return x
    